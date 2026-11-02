#!/usr/bin/env python3
import os
import glob
import json
import math
import argparse

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from sklearn.model_selection import train_test_split

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']
MEAN = np.array([0.4914, 0.4822, 0.4465], np.float32)
STD = np.array([0.2470, 0.2435, 0.2616], np.float32)
IMG = 32
AUTOTUNE = tf.data.AUTOTUNE


def configure_runtime(seed):
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    policy = 'mixed_float16' if gpus else 'float32'
    print('GPUs:', gpus)
    print('TF:', tf.__version__, '| dtype policy:', policy)
    if not gpus:
        print('CPU-only: expect roughly 2 to 5 minutes per epoch for this model.')
    tf.keras.mixed_precision.set_global_policy(policy)
    tf.keras.utils.set_random_seed(seed)


def _numeric_key(path):
    return int(os.path.splitext(os.path.basename(path))[0])


def load_train_arrays(data_dir):
    paths = sorted(glob.glob(os.path.join(data_dir, 'train', '*.png')),
                   key=_numeric_key)
    if not paths:
        raise FileNotFoundError(
            f'no PNGs under {data_dir}/train/ - extract train.7z there first')
    labels = pd.read_csv(os.path.join(data_dir, 'trainLabels.csv'))
    labels['id'] = labels['id'].astype(int)
    labels = labels.set_index('id').loc[[_numeric_key(p) for p in paths]]
    name_to_idx = {c: i for i, c in enumerate(CLASSES)}
    y = labels['label'].map(name_to_idx).to_numpy(np.int32)
    return np.array(paths), y


def decode_labeled(path, label):
    img = tf.io.decode_png(tf.io.read_file(path), channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)      # [0,1]
    return tf.ensure_shape(img, [IMG, IMG, 3]), label


def augment(img, label):
    img = tf.image.resize_with_crop_or_pad(img, IMG + 4, IMG + 4)
    img = tf.image.random_crop(img, [IMG, IMG, 3])
    img = tf.image.random_flip_left_right(img)

    size = 8
    cy = tf.random.uniform([], 0, IMG, tf.int32)
    cx = tf.random.uniform([], 0, IMG, tf.int32)
    y1 = tf.clip_by_value(cy - size // 2, 0, IMG)
    x1 = tf.clip_by_value(cx - size // 2, 0, IMG)
    rows = tf.range(IMG)[:, None]
    cols = tf.range(IMG)[None, :]
    inside = (rows >= y1) & (rows < y1 + size) & (cols >= x1) & (cols < x1 + size)
    img = img * (1.0 - tf.cast(inside, tf.float32))[..., None]
    return img, label


def conv_bn(x, filters, drop=0.0):
    x = layers.Conv2D(filters, 3, padding='same', use_bias=False,
                      kernel_regularizer=regularizers.l2(5e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    if drop:
        x = layers.SpatialDropout2D(drop)(x)
    return x


def build_model():
    inp = layers.Input((IMG, IMG, 3))
    x = layers.Normalization(mean=MEAN, variance=STD ** 2)(inp)
    x = conv_bn(x, 64)
    x = conv_bn(x, 64)
    x = layers.MaxPooling2D()(x)                      # 16x16
    x = conv_bn(x, 128)
    x = conv_bn(x, 128, 0.1)
    x = layers.MaxPooling2D()(x)                      # 8x8
    x = conv_bn(x, 256)
    x = conv_bn(x, 256, 0.1)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(10, dtype='float32')(x)          # fp32 logits
    return models.Model(inp, x, name='cifar_cnn')


def build_datasets(paths, y, train_idx, val_idx, batch, seed):
    train = tf.data.Dataset.from_tensor_slices((paths[train_idx], y[train_idx]))
    train = (train
             .map(decode_labeled, AUTOTUNE)          # path -> (32,32,3) float32
             .cache()                                # ~176 MB, removes PNG decode per epoch
             .shuffle(len(train_idx), seed=seed, reshuffle_each_iteration=True)
             .map(augment, AUTOTUNE)                 # image -> image
             .batch(batch)
             .prefetch(AUTOTUNE))
    val = tf.data.Dataset.from_tensor_slices((paths[val_idx], y[val_idx]))
    val = (val.map(decode_labeled, AUTOTUNE)
              .batch(batch)
              .cache()
              .prefetch(AUTOTUNE))
    return train, val


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='data')
    p.add_argument('--out-dir', default='artifacts')
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--val-frac', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=1337)
    args = p.parse_args()

    configure_runtime(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    paths, y = load_train_arrays(args.data_dir)
    idx = np.arange(len(paths))
    train_idx, val_idx = train_test_split(
        idx, test_size=args.val_frac, stratify=y, random_state=args.seed)

    np.save(os.path.join(args.out_dir, 'val_indices.npy'), val_idx)
    with open(os.path.join(args.out_dir, 'class_names.json'), 'w') as f:
        json.dump(CLASSES, f)
    print(f'train {len(train_idx)} | val {len(val_idx)}')

    train_ds, val_ds = build_datasets(
        paths, y, train_idx, val_idx, args.batch_size, args.seed)

    model = build_model()
    model.summary()

    steps = math.ceil(len(train_idx) / args.batch_size)
    schedule = tf.keras.optimizers.schedules.CosineDecay(
        args.lr, decay_steps=args.epochs * steps)

    model.compile(
        optimizer=tf.keras.optimizers.AdamW(schedule, weight_decay=1e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, label_smoothing=0.1),
        metrics=['accuracy'])

    ckpt = os.path.join(args.out_dir, 'best.keras')
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            ckpt, monitor='val_accuracy', save_best_only=True, verbose=1),
        tf.keras.callbacks.CSVLogger(os.path.join(args.out_dir, 'train_log.csv')),
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=15, restore_best_weights=True),
    ]

    model.fit(train_ds, validation_data=val_ds,
              epochs=args.epochs, callbacks=callbacks)

    best = tf.keras.models.load_model(ckpt)
    loss, acc = best.evaluate(val_ds, verbose=0)
    print(f'\nbest checkpoint: val_loss={loss:.4f} val_acc={acc:.4f}')
    print('next: python3 modeltest.py --data-dir', args.data_dir,
          '--out-dir', args.out_dir)


if __name__ == '__main__':
    main()
