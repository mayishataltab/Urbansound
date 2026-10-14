# Urban Sound Classification: ANN vs SVM

A machine learning project that classifies urban environmental sounds using audio feature extraction and compares an Artificial Neural Network (ANN) with a Support Vector Machine (SVM).

**Author:** Mayishat Altab

## Overview

Urban environments contain a wide variety of sounds, including sirens, vehicle engines, dog barks, and street activity. Automatically classifying these sounds can support environmental sound analysis and audio-based recognition systems.

This project uses the **UrbanSound8K dataset** to develop and evaluate two classification approaches:

- **Artificial Neural Network (ANN):** A neural network with hyperparameter tuning using KerasTuner.
- **Support Vector Machine (SVM):** An RBF-kernel classifier with hyperparameter selection using GridSearchCV.

Both models use the same extracted and standardized audio features, enabling a more consistent comparison.

## Objectives

- Explore and analyze an urban environmental sound dataset.
- Extract meaningful audio features using Mel-Frequency Cepstral Coefficients (MFCCs).
- Compare ANN and RBF-SVM classification performance.
- Apply data augmentation to training data.
- Evaluate models using multiple classification metrics.
- Save trained models and preprocessing artifacts for future predictions.

## Dataset

**UrbanSound8K** is a dataset of urban environmental audio recordings organized into 10 sound classes.

The notebook expects the dataset's metadata CSV and audio recordings to be available locally.

Expected directory structure:

```text
project/
├── urban_sound_classification_clean.ipynb
├── data/
│   └── urbansound/
│       ├── UrbanSound8K.csv
│       ├── fold1/
│       ├── fold2/
│       ├── fold3/
│       ├── fold4/
│       ├── fold5/
│       ├── fold6/
│       ├── fold7/
│       ├── fold8/
│       ├── fold9/
│       └── fold10/
└── saved_models/
```

The dataset is not included in this repository. Obtain it from its official source and follow its terms of use.

## Methodology

### 1. Audio exploration

The notebook examines class distribution and visualizes audio using waveforms, spectrograms, Mel spectrograms, and MFCC representations.

### 2. Feature extraction

Audio is loaded and processed using Librosa. The notebook extracts **40 MFCC coefficients** and calculates their mean and standard deviation across time.

The two vectors are concatenated to produce an **80-dimensional feature representation** for each audio clip.

### 3. Data preparation

The workflow includes:

- A stratified 70% training, 15% validation, and 15% test split.
- Label encoding for target classes.
- Feature standardization with `StandardScaler`, fitted on the training data.
- Random seeds for Python, NumPy, and TensorFlow to support reproducibility.

### 4. Training-only data augmentation

The notebook defines audio augmentation operations including:

- Noise addition
- Gain adjustment
- Time shifting
- Pitch shifting
- Time stretching

Augmentation is applied only to training data, not to validation or test samples.

### 5. ANN training and tuning

The ANN uses TensorFlow/Keras. KerasTuner Random Search is used to select hyperparameters, after which the selected model is trained and evaluated.

### 6. SVM training and tuning

The SVM uses an RBF kernel. `GridSearchCV` with five-fold cross-validation searches over selected values of `C` and `gamma`.

The SVM uses the same standardized feature representation as the ANN.

### 7. Evaluation and comparison

The notebook compares the models using:

- Accuracy
- Macro precision and recall
- Macro and weighted F1-score
- Classification reports
- Confusion matrices
- Per-class accuracy
- Training and inference time
- Model parameter counts

The held-out test set is reserved for final evaluation and is not used for model selection.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Librosa | Audio processing and MFCC extraction |
| TensorFlow / Keras | ANN development and training |
| KerasTuner | ANN hyperparameter tuning |
| Scikit-learn | SVM, scaling, data splitting, and evaluation |
| NumPy and Pandas | Numerical computation and dataset handling |
| Matplotlib and Seaborn | Data visualization |
| SoundFile | Audio file inspection and reading |
| Joblib and Pickle | Saving preprocessing objects and training artifacts |

## Installation

Create a Python environment and install the required libraries:

```bash
pip install numpy pandas librosa matplotlib seaborn
pip install tensorflow keras-tuner scikit-learn
pip install soundfile joblib tqdm
```

Open the notebook:

```bash
jupyter notebook urban_sound_classification_clean.ipynb
```

You can also run it in Google Colab after uploading the notebook and making the dataset accessible.

**Dataset path note:** Before running, verify that `METADATA_PATH` points directly to `data/urbansound/UrbanSound8K.csv`. In the current notebook, its path expression may append `data/urbansound` twice; adjust it if necessary.

## Saved Artifacts

After successful execution, the notebook saves artifacts in `saved_models/`, including:

- `best_model.keras` — trained ANN
- `scaler.pkl` — fitted feature scaler
- `label_encoder.pkl` — class-label encoder
- `svm_model.pkl` — trained SVM model
- `history.pkl` — ANN training history
- `metrics.json` — evaluation metrics and configuration
- `ann_svm_comparison.csv` — model comparison table

These artifacts support reloading the ANN prediction pipeline and reproducing predictions without retraining the model.

## Making a Prediction

The notebook includes a prediction function that follows this pipeline:

**Audio file → MFCC features → StandardScaler → ANN → Predicted class and confidence**

The input must be an audio file accessible from the environment where the notebook is running.

## Reproducibility and Evaluation

The notebook incorporates several measures intended to improve experimental reliability:

- Fixed random seeds
- Stratified data splitting
- Training-only fitting of the scaler
- Training-only data augmentation
- Hyperparameter tuning before final test evaluation
- Multiple performance metrics rather than accuracy alone

Results may vary with software versions, hardware, dataset integrity, and training conditions. Report final performance using the metrics generated by a successful run of the notebook.

## Future Improvements

- Compare additional audio feature representations.
- Explore convolutional neural networks operating on spectrograms.
- Evaluate generalization using alternative dataset splits.
- Develop a user-facing audio classification application.

## Acknowledgment

This project uses the UrbanSound8K dataset. Please cite the original dataset publication when using the dataset in academic or research work.

**Reference:** Salamon, J., Jacoby, C., and Bello, J. P. “A Dataset and Taxonomy for Urban Sound Research.” *Proceedings of the 22nd ACM International Conference on Multimedia*, 2014.


