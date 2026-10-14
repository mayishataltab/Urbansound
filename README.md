# Urban Sound Classification — ANN vs. SVM

**Author:** Mayishat Altab

## Overview
This repository presents a rigorous, end-to-end machine learning and deep learning pipeline for classifying environmental audio clips using the **UrbanSound8K** dataset. The project compares the performance of a hyperparameter-tuned **Artificial Neural Network (ANN)** against a traditional **Support Vector Machine (SVM)** baseline using Mel-Frequency Cepstral Coefficients (MFCCs) as acoustic features.

A primary focus of this implementation is **strict methodological integrity**—eliminating data leakage, enforcing end-to-end reproducibility, and ensuring realistic model evaluation on unseen audio.

---

## Key Methodological Highlights
* **Zero Data Leakage:** The dataset is partitioned using a **70% / 15% / 15%** stratified Train / Validation / Test split. The held-out test set is isolated prior to augmentation, scaling, and hyperparameter tuning, and is evaluated **exactly once** at the end of the pipeline.
* **Train-Only Audio Augmentation:** Acoustic data augmentation is applied exclusively to the training split (`6,112` base samples expanded to `12,224` training samples), preventing synthetic leakage into validation or test sets.
* **Temporal-Spectral Feature Representation:** Instead of relying solely on time-averaged MFCCs, each audio clip is represented by an **80-dimensional feature vector** concatenating the **40-band MFCC mean** and **40-band MFCC standard deviation**, capturing both average spectral envelope and temporal variance.
* **Leak-Free Feature Scaling:** `StandardScaler` is fitted strictly on the training features and applied via `.transform()` to validation, test, SVM, and single-file inference pipelines.
* **Automated Dataset Integrity Verification:** Pre-training header validation (`soundfile`) automatically checks all 8,732 audio files for missing paths, duplicate metadata entries, or corrupted audio streams.
* **End-to-End Reproducibility:** Global random seeds (`SEED = 42`) are enforced across Python `random`, `NumPy`, `TensorFlow`, and `PYTHONHASHSEED`.

---

## Dataset
The project uses the [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html) dataset, which contains **8,732 labeled sound excerpts** ($\le 4\text{ seconds}$) of urban sounds from 10 classes:
`air_conditioner`, `car_horn`, `children_playing`, `dog_bark`, `drilling`, `engine_idling`, `gun_shot`, `jackhammer`, `siren`, and `street_music`.

| Split | Base Files | Augmented Files | Total Samples | Proportion |
| :--- | :---: | :---: | :---: | :---: |
| **Training Set** | 6,112 | +6,112 | **12,224** | 70.0% (of base) |
| **Validation Set** | 1,310 | 0 | **1,310** | 15.0% |
| **Test Set** | 1,310 | 0 | **1,310** | 15.0% |

---

## Pipeline & Architecture

### 1. Audio Preprocessing & Augmentation
Audio files are loaded at a `22,050 Hz` sampling rate using `librosa`. Stochastic waveform augmentations applied to the training split include:
* **Gaussian Noise Injection** (`p = 0.5`, factor = `0.005`)
* **Random Gain Scaling** (`p = 0.3`, range = `0.8x – 1.2x`)
* **Time Shifting** (`p = 0.3`, roll up to `±20%`)
* **Pitch Shifting** (`p = 0.2`, `±2` semitones)
* **Time Stretching** (`p = 0.2`, rate = `0.8x – 1.2x`)

### 2. Deep Learning Model (ANN + KerasTuner)
* **Architecture:** Sequential feed-forward blocks of `Dense -> BatchNormalization -> ReLU -> Dropout`, regularized with L2 weight decay, terminating in a 10-unit `Softmax` output layer.
* **Hyperparameter Optimization:** Tuned via `keras_tuner.RandomSearch` (15 trials, validated strictly on the validation split).
* **Optimal Configuration:**
  * Hidden Layers: `2` (`480` units $\rightarrow$ `384` units)
  * Dropout Rate: `0.2`
  * Optimizer: `Adam` (`lr = 1e-4`)
  * Batch Size: `64`
  * Total Parameters: `230,890` (`229,162` trainable)
* **Training Callbacks:** `EarlyStopping` (`patience=10`, restoring best weights), `ReduceLROnPlateau` (`factor=0.5`, `patience=5`), and `ModelCheckpoint`.
* **Validation Performance:** Achieved **94.81% validation accuracy** during tuning and converged smoothly to **94.66% validation accuracy** during full training.

### 3. Classical Baseline (SVM)
* A Support Vector Classifier (`SVC`) trained on the identical scaled 80-dimensional MFCC feature set and evaluated alongside the ANN using comprehensive metrics (**Accuracy, Precision, Recall, F1-Score, and Confusion Matrices**).
