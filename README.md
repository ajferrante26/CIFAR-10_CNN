# CIFAR-10 Image Classification with a CNN

Trains a convolutional neural network from scratch on the Kaggle
[CIFAR-10 Object Recognition in Images](https://www.kaggle.com/c/cifar-10)
competition data, evaluates it on a held-out split of the training set,
visualizes the errors, and produces a `submission.csv` for the leaderboard.

Runs on a Jetson Orin Nano with NVIDIA's TensorFlow wheel, and on a plain
CPU-only machine for development.

## Repository layout

```
assignment4/
├── model.py            # training entry point
├── modeltest.py        # evaluation, plots, submission generation
├── data/
│   ├── train/          # 1.png ... 50000.png
│   ├── test/           # 1.png ... 300000.png
│   ├── trainLabels.csv
│   └── sample_submission.csv
├── artifacts/          # created by model.py
│   ├── best.keras
│   ├── train_log.csv
│   ├── val_indices.npy
│   ├── class_names.json
│   └── plots/
└── submission.csv      # created by modeltest.py
```

## Data

Download the competition data and extract it into `data/`:

```bash
kaggle competitions download -c cifar-10
7z x cifar-10.zip -odata
7z x data/train.7z -odata
7z x data/test.7z -odata
```

The layout that `model.py` expects is `data/train/*.png`,
`data/test/*.png`, and `data/trainLabels.csv` sitting directly inside
`data/`. Both scripts check this at startup and fail with a message
naming the missing path.

This is the competition archive, not `tf.keras.datasets.cifar10`. The
images are individual PNGs and the labels arrive as a CSV keyed by the
numeric image filename. Two consequences worth knowing:

* The 50,000 training images are split into 45,000 train and 5,000
  validation with a fixed seed. The validation split is the only labeled
  held-out data you have.
* The real test accuracy exists only on the Kaggle leaderboard, because
  `test.7z` ships 300,000 unlabeled images.

## Environment setup

### Windows, CPU only

TensorFlow dropped native Windows GPU support at version 2.11. A Windows
run always reports `GPUs: []` and always uses the CPU, regardless of the
installed CUDA toolkit.

```powershell
pip install "tensorflow==2.18.*" scikit-learn pandas matplotlib kaggle
```

For GPU work on Windows, use WSL2 with a CUDA toolkit inside it. The
`TensorFlow-DirectML` plugin is an alternative with partial operator
coverage.

### Jetson Orin Nano

I tried it, just unzipping the images took 6+ hours. Overall it fell on its face. Just don't do it. 

## Training

```bash
python3 model.py
```

Smoke test the pipeline end to end before committing to a long run:

```bash
python3 model.py --epochs 1
```

One epoch should finish with `val_accuracy` around 0.60 on a working
pipeline.

### model.py arguments

| Flag | Default | Meaning |
| --- | --- | --- |
| `--data-dir` | `data` | folder holding `train/`, `test/`, `trainLabels.csv` |
| `--out-dir` | `artifacts` | checkpoint and log destination |
| `--epochs` | `60` | training epochs |
| `--batch-size` | `128` | images per step |
| `--lr` | `1e-3` | peak learning rate for the cosine schedule |
| `--val-frac` | `0.1` | fraction of the 50k training set held out |
| `--seed` | `1337` | seed for split, shuffle, and weight init |

### What the model is

Eight convolutional layers in three stages (64, 128, 256 filters), each
convolution followed by batch normalization and ReLU, with 3x3 kernels,
L2 weight decay, two max-pool stages, global average pooling, and dropout
before a single dense layer. About 1.5 M parameters.

Training uses AdamW with a cosine learning rate decay, label smoothing of
0.1, and augmentation in the input pipeline: random 4-pixel pad and crop,
random horizontal flip, and 8x8 cutout. Mixed precision (`float16`) is
enabled automatically when a GPU is present and left at `float32` on CPU.

Input normalization by the CIFAR-10 channel mean and standard deviation
is a layer inside the model, so inference and submission paths cannot
apply it inconsistently.

### What to expect

| Hardware | Time per epoch | 60-epoch wall time |
| --- | --- | --- |
| Desktop CPU, oneDNN | ~2 to 5 min | ~2 to 5 h |
| Kaggle GPU (T4 or P100) | ~25 to 40 s | ~25 to 40 min |

Measure epoch 1 and extrapolate before walking away from a 60-epoch run.

## Evaluation and submission

```bash
python3 modeltest.py
```

This loads `artifacts/best.keras`, scores the fixed validation split with
horizontal-flip test-time augmentation, writes four plots, and runs
inference over all 300,000 test images.

Submit:

```bash
kaggle competitions submit -c cifar-10 -f submission.csv -m "cnn fp16 60ep"
```

### modeltest.py arguments

| Flag | Default | Meaning |
| --- | --- | --- |
| `--data-dir` | `data` | same layout as above |
| `--out-dir` | `artifacts` | where `best.keras` lives |
| `--batch-size` | `256` | inference batch size |
| `--no-tta` | off | disable horizontal-flip averaging |
| `--label-format` | `name` | `name` writes `cat`, `int` writes `3` |
| `--skip-test` | off | evaluation and plots only, skip test inference |

### Outputs

* `artifacts/plots/history.png` : train and validation loss and accuracy per epoch
* `artifacts/plots/confusion_matrix.png` : row-normalized confusion matrix
* `artifacts/plots/misclassified.png` : 25 misclassified validation images with true and predicted labels
* `artifacts/plots/confidence.png` : top-1 confidence distribution, correct vs incorrect
* `submission.csv` : `id,label` rows for the leaderboard

`modeltest.py` also prints the per-class precision and recall table and
the predicted class distribution over the test set. Each class should
land near 0.10. A class near 0.30 means the label mapping is wrong.

## Expected results

Reference numbers for this model on the 5,000-image validation split:

* Validation accuracy: 0.93 to 0.95 with flip TTA
* Dominant error mode: `cat` predicted as `dog` and the reverse, which
  is the usual confusion pair on CIFAR-10 at this resolution
* Kaggle public leaderboard typically sits a little lower than validation

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `must have either 3 or 4 dimensions` inside `augment` | training pipeline mapped `augment` over undecoded `(path, label)` pairs | use the `build_datasets` version that maps `decode_labeled` first |
| `GPUs: []` on Windows | TensorFlow 2.11+ has no native Windows GPU support | use WSL2, DirectML, or the Orin Nano |
| `GPUs: []` on the Jetson | PyPI wheel instead of NVIDIA's | reinstall from the Jetson Zoo index for your L4T version |
| `data layout problem` | missing `train/`, `test/`, or `trainLabels.csv` | extract the archives and download the labels CSV into `data/` |
| Submission scores near zero | label column format mismatch | `head -3 data/sample_submission.csv`, then match with `--label-format` |
| One test class above 25% of predictions | class index order disagrees with `trainLabels.csv` | check the printed class distribution |
| Step time flat as batch size grows | tf.data thread oversubscription | replace `AUTOTUNE` with `4` in the `.map()` calls |
| `trainLabels.csv` unavailable | archive extracted without it | rebuild from `tf.keras.datasets.cifar10`, which is the same 50,000 images in the same order, and verify before trusting it |

## Reproducibility notes

Seeds are set for Python, NumPy, and TensorFlow, and the validation split
indices are saved to `val_indices.npy` so evaluation always reuses the
exact rows from training. Bit-exact reruns are still not achievable:
cuDNN kernel selection and floating point reduction order vary between
runs, and mixed precision adds its own variance. Expect about plus or
minus 0.3 percentage points on validation accuracy between identical runs.

For a number that will hold up in a writeup, run the final configuration
three times and report the mean and standard deviation rather than the
best single run.

## Deliverables

* Validation accuracy and per-class table from `modeltest.py`
* The four plots in `artifacts/plots/`
* A screenshot of the Kaggle leaderboard row for the submitted `submission.csv`
