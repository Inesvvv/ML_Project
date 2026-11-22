# Parkinson's Disease Detection: Model Comparison

Group machine-learning project. We compare six classifier families for detecting Parkinson's disease from 753 acoustic speech features. All of them use the same leakage-safe protocol, a **patient-grouped** train/test split with grouped 5-fold cross-validation, so no subject's recordings appear on both sides of a split.

## Data

*Parkinson's Disease Classification* dataset (Sakar et al., 2019, UCI): 756 voice recordings from 252 subjects (3 per subject). The classes are imbalanced (75% Parkinson's), so a majority-class baseline already reaches **72.5% accuracy**. Models are therefore compared on **ROC-AUC**, with thresholds tuned separately for the trade-off between sensitivity and specificity.

## Shared pipeline

1. Grouped 80/20 split by subject ID (603 train / 153 test recordings)
2. Highly correlated features removed (753 → 539), fit on training data only
3. Optional mutual-information feature selection or PCA, fit on training data only
4. Hyperparameter search with grouped 5-fold CV, optimising mean ROC-AUC
5. Retrain on the full training set, evaluate once on held-out test subjects, then analyse thresholds

## Results (held-out test subjects)

| Model | Tuning | Test ROC-AUC | Notebook |
|---|---|---|---|
| **Gradient boosting** | Grid: n_estimators × learning rate × depth | **0.954** | `gradient_boosting_FINAL.ipynb` |
| Random forest | Grid: n_estimators × depth | 0.928 | `random_forest_FINAL.ipynb` |
| Logistic regression + PCA (95% variance, 170 components) | C, penalty, class weight | 0.902 | `Logistic_Regression_PCA.ipynb`, `Logistic_Regression_Tuning.ipynb` |
| Neural network (4 layers) | Optuna TPE, 50 trials | 0.879 | `hyperparameter_tuning_neuralnet_FINAL.ipynb` |
| k-nearest neighbours | k, weighting, metric | 0.864 | `knn_Model_FINAL.ipynb` |
| Bagged decision trees | Grid: n_estimators × depth (CV ROC-AUC 0.809) | — | `random_forest_FINAL.ipynb` |

**Takeaways**
- Tree ensembles worked best on this tabular, high-dimensional, small-*n* dataset. Gradient boosting reached 0.954 ROC-AUC, and with threshold 0.80 it gets 92.8% accuracy with 96% recall on Parkinson's cases.
- The most informative features in both tree models are TQWT (tunable Q-factor wavelet) and MFCC features, which matches the original paper.
- The test set is only about 51 subjects, so differences of a few points in AUC between models are within sampling noise.

## Team

| Member | Contribution |
|---|---|
| Inès Vallot | Neural network: Optuna search, grouped CV and threshold analysis |
| Chloé Deux-Pancera | Random forest, gradient boosting, bagging |
| Apolline Weinstein | Logistic regression (tuning and PCA), k-NN |

Extended individual work on the neural network: [Parkinsons_MachineLearning](https://github.com/Inesvvv/Parkinsons_MachineLearning)

## Running

The notebooks were written for Google Colab and read `pd_speech_features.csv` from Google Drive. To run them locally, download the dataset from the [UCI repository](https://archive.ics.uci.edu/dataset/470/parkinson+s+disease+classification) and update the `read_csv` path.

**Tools:** Python · scikit-learn · TensorFlow/Keras · Optuna · pandas · Google Colab
