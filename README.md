
# Breast Cancer Classification using Machine Learning

## 1. Problem Statement

The objective of this project is to develop and compare multiple machine learning
classification models for predicting whether a breast tumor is benign or malignant.

Five supervised machine learning models are implemented and evaluated using the
same training and test data:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Random Forest

The models are compared using Accuracy, AUC, Precision, Recall, F1 Score, and
Matthews Correlation Coefficient (MCC).

---

## 2. Dataset Description

Dataset Source: UCI Machine Learning Repository

The dataset contains:

- 569 observations
- 30 numerical input features
- 1 target variable: Diagnosis
- 357 benign observations
- 212 malignant observations

The 30 features describe characteristics computed from digitized images of
fine needle aspirate (FNA) of breast masses.

The target variable is encoded as:

- 0 = Benign
- 1 = Malignant

There are no missing values in the features.

For model evaluation, the dataset was divided into:

- 80% training data
- 20% test data

The split was stratified to preserve the class distribution.

---

## 3. GitHub Repository

GitHub Repository: [TO BE UPDATED AFTER REPOSITORY CREATION]

---

## 4. Models Used

### Logistic Regression

A linear classification model used as a strong baseline for binary
classification.

### Decision Tree

A tree-based classifier that makes predictions using a sequence of
feature-based decision rules.

### K-Nearest Neighbors (KNN)

A distance-based classifier that predicts the class using the nearest
training observations.

### Gaussian Naive Bayes

A probabilistic classifier based on Bayes' theorem with a conditional
independence assumption and Gaussian distribution for continuous features.

### Random Forest

An ensemble classifier consisting of multiple decision trees whose
predictions are combined to obtain the final classification.

---

## 5. Model Performance Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9298 | 0.9246 | 0.9048 | 0.9048 | 0.9048 | 0.8492 |
| KNN | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Gaussian Naive Bayes | 0.9211 | 0.9891 | 0.9231 | 0.8571 | 0.8889 | 0.8292 |
| Random Forest | 0.9737 | 0.9929 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |

---

## 6. Observations on Model Performance

### Logistic Regression

Logistic Regression achieved strong overall performance with 96.49% accuracy
and 99.60% AUC. It achieved 97.50% precision, 92.86% recall, 95.12% F1
score, and 92.45% MCC. It also achieved the highest AUC among the five
models.

### Decision Tree

Decision Tree achieved 92.98% accuracy, 92.46% AUC, and 90.48% recall.
Its performance was lower than the other major models on most evaluation
metrics. Its confusion matrix contained four false positives and four
false negatives.

### KNN

KNN achieved 95.61% accuracy and 98.23% AUC. Its precision was high at
97.44%, while recall was 90.48%. It correctly classified most observations
but missed four malignant cases in the test set.

### Gaussian Naive Bayes

Gaussian Naive Bayes achieved 92.11% accuracy and 85.71% recall, which were
the lowest values among the five models. However, it achieved a relatively
high AUC of 98.91%, indicating strong class-ranking performance despite
weaker classification performance at the default decision threshold.

### Random Forest

Random Forest achieved the best overall classification performance with
97.37% accuracy, 100% precision, 92.86% recall, 96.30% F1 score, and
94.42% MCC. It produced zero false positives and three false negatives.

---

## 7. Overall Winner

Random Forest is selected as the overall winner for this dataset because it
achieved the highest Accuracy, Precision, F1 Score, and MCC, while matching
Logistic Regression in Recall.

Random Forest achieved:

- Accuracy: 97.37%
- Precision: 100.00%
- Recall: 92.86%
- F1 Score: 96.30%
- MCC: 94.42%

Logistic Regression achieved the highest AUC at 99.60%, slightly higher than
Random Forest's 99.29%. Therefore, Random Forest is the overall best
classifier based on the combined classification metrics, while Logistic
Regression has the strongest AUC.

---

## 8. Project Files

The repository contains:

- `app.py` - Streamlit application
- `requirements.txt` - Python dependencies
- `README.md` - Project documentation
- `test_data.csv` - Test dataset used by the application
- `model/` - Saved trained machine learning models and scaler

---

## 9. Streamlit Application

The Streamlit application provides:

- CSV test-data upload
- Model selection
- Model predictions
- Evaluation metrics
- Confusion matrix
- Results for the different trained models

Live Streamlit App: [TO BE UPDATED AFTER DEPLOYMENT]
