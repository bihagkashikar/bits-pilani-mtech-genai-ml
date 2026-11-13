# Dataset for the assignment

## Datasets and environment

Two datasets are free and download with a few lines of code and the third is generated in code; nothing else needs to be downloaded, and no pre-trained model is used.

### Dataset 1: UCI HAR (Problem 1 and Problem 2, Task A). Official page: Human Activity Recognition Using Smartphones. Download with the page's Download button, or in code:

```python
import urllib.request, zipfile
    url = 'https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip'`
    urllib.request.urlretrieve(url, 'har.zip')
    zipfile.ZipFile('har.zip').extractall('har')
```

# The archive may hold a second zip, 'UCI HAR Dataset.zip': extract that one too.

After unzipping, the folder UCI HAR Dataset holds everything used:
• train/Inertial Signals/ and test/Inertial Signals/: nine text files per split, each with one row per window of 128 steps.
• Channel order, fixed for everyone: body_acc_x, body_acc_y, body_acc_z, body_gyro_x, body_gyro_y, body_gyro_z, total_acc_x, total_acc_y, total_acc_z.
• train/y_train.txt and test/y_test.txt: labels 1–6 (walking, walking upstairs, walking downstairs, sitting, standing, laying).
• train/subject_train.txt and test/subject_test.txt: the subject ID of every window.
• Stacked shapes: train (7352, 128, 9), test (2947, 128, 9). The 21 training subjects and 9 test subjects do not overlap.
• Validation split: hold out 4 of the 21 training subjects, chosen with np.random.default_rng(seed).choice(train_subjects, 4, replace=False), using each of the student's three seeds in turn.

### Dataset 2: Tiny Shakespeare (Problem 2, Task B). Source: karpathy/tiny_shakespeare. Plain-text file, about 1.1 MB:

```python
    import urllib.request
    url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    text = urllib.request.urlopen(url).read().decode('utf-8')
```
Build the character vocabulary from the whole text (65 characters). Use the first 90% of the characters for training and the last 10% for validation, with no shuffling.

### Dataset 3 – adding problem (Problem 1, generated in code). No download is needed. Each input has two channels: a random value in [0, 1) and a marker. Exactly two steps are marked, one in each half of the sequence, and the target is the sum of the two marked values. Use this function so that every student has identical data:

```python
import numpy as np
def make_adding(n, T, seed):
    rng = np.random.default_rng(seed)
    x = np.zeros((n, T, 2), dtype=np.float32)
    x[:, :, 0] = rng.random((n, T))
    i1 = rng.integers(0, T // 2, n)
    i2 = rng.integers(T // 2, T, n)
    x[np.arange(n), i1, 1] = 1
    x[np.arange(n), i2, 1] = 1
    y = x[np.arange(n), i1, 0] + x[np.arange(n), i2, 0]
    return x, y.astype(np.float32)
```
For each T use 10,000 training sequences (seed 0), 1,000 validation (seed 1) and 1,000 test (seed 2). A model that always predicts 1.0 has MSE of about 0.167, which is the baseline to beat.
