# Urban Sound Classification: ANN vs SVM

**Author:** Mayishat Altab (GitHub: [@mayishataltab](https://github.com/mayishataltab)

## Overview

This project classifies urban sound clips from the UrbanSound8K dataset into 10 classes (air conditioner, car horn, children playing, dog bark, drilling, engine idling, gun shot, jackhammer, siren, street music). It compares an Artificial Neural Network tuned with KerasTuner against a Support Vector Machine (RBF kernel) baseline.

The full work is in `UrbanNoise_updated.ipynb`.

## Method

- **Features:** 40 MFCCs per clip (audio loaded at 22,050 Hz), summarized by mean and standard deviation, giving 80 features per clip.
- **Split:** 70% train / 15% validation / 15% test (6,112 / 1,310 / 1,310 clips), stratified by class. The test set is used once, at the end.
- **Scaling:** `StandardScaler` fit on the training set only.
- **Augmentation:** noise, gain, time shift, pitch shift, and time stretch, applied to the training split only.
- **ANN:** hyperparameters tuned with KerasTuner (15 trials, best validation accuracy 94.8%).
- **SVM:** RBF kernel tuned with grid search (best C = 100, gamma = scale).
- **Reproducibility:** global random seeds set for `random`, `numpy`, and `tensorflow`.

## Results (held-out test set, 1,310 clips)

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | Training time | Inference time |
|---|---|---|---|---|---|---|
| ANN | 0.9481 | 0.9487 | 0.9467 | 0.9472 | 51 s | 0.73 s |
| SVM (RBF) | 0.9466 | 0.9431 | 0.9476 | 0.9451 | 177 s | 0.80 s |

The two models perform almost identically. The ANN is slightly ahead on accuracy and F1 and trained faster in this run. The difference (about 0.15 percentage points) is small and was measured on one split, so it should not be treated as a clear win.

Per-class results are in the notebook. The weakest class for the ANN is `car_horn` (F1 0.90, recall 0.86). The strongest are `air_conditioner` and `engine_idling` (F1 0.99 and 0.98).


## Dataset

This project uses the UrbanSound8K dataset, which was created by others and not collected by me.

> J. Salamon, C. Jacoby and J. P. Bello, "A Dataset and Taxonomy for Urban Sound Research,"
> 22nd ACM International Conference on Multimedia (ACM-MM'14), Orlando, FL, USA, Nov. 2014, pp. 1041-1044.

Dataset page: https://urbansounddataset.weebly.com/urbansound8k.html

## Tools

Python, TensorFlow/Keras, KerasTuner, scikit-learn, librosa, pandas, NumPy, Matplotlib, seaborn.
