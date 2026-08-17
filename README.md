# Breast Cancer Classification using Machine Learning

## 1. Problem Statement

The objective of this project is to build and compare multiple machine-learning classification models for breast-tumor classification. The models classify observations into **Benign (B)** and **Malignant (M)** classes using the Breast Cancer Wisconsin (Diagnostic) dataset.

The project implements the classification models required in the BITS WILP Machine Learning Assignment - 2, evaluates them using multiple performance metrics, and provides an interactive Streamlit application for testing and comparing the trained models.

---

## 2. Dataset Description

### Dataset

**Breast Cancer Wisconsin (Diagnostic) Dataset**

The dataset was obtained from the **UCI Machine Learning Repository** (UCI dataset ID: 17).

### Dataset characteristics

* **Number of instances:** 569
* **Number of input features:** 30
* **Problem type:** Binary classification
* **Target variable:** `Diagnosis`
* **Classes:** `B` – Benign, `M` – Malignant
* **Class distribution:** 357 Benign, 212 Malignant

The 30 numerical features include measurements such as radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension.

### Data preparation

The target variable was encoded as `B → 0` and `M → 1`.

An **80:20 stratified train/test split** with `random\_state=42` was used:

* Training set: **455 samples × 30 features**
* Test set: **114 samples × 30 features**

`StandardScaler` was used for the models requiring scaled numerical features. The fitted scaler is stored as `model/scaler.pkl`.

---

## 3. GitHub Repository Link

**GitHub Repository:**  
https://github.com/arunpraveencse/breast-cancer-classification-ml

The repository contains the source code, dependencies, test data, README documentation, and saved model/scaler files required for running and deploying the application.

### Repository Structure

```text
breast-cancer-classification-ml/
│
├── app.py
├── requirements.txt
├── README.md
├── test\data.csv
│
└── model/
    ├── decision\tree.pkl
    ├── knn.pkl
    ├── logistic\regression.pkl
    ├── naive\bayes.pkl
    ├── random\forest.pkl
    └── scaler.pkl
```

\---

## 4. Models Used

The assignment explicitly lists the following classification models:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Random Forest (Ensemble)



### Evaluation Metrics

Each model was evaluated using:

* Accuracy
* AUC Score
* Precision
* Recall
* F1 Score
* Matthews Correlation Coefficient (MCC)

### Model Comparison

The following results were obtained on the supplied `test\_data.csv` through the deployed Streamlit application.

|ML Model|Accuracy|AUC|Precision|Recall|F1 Score|MCC|
|-|-:|-:|-:|-:|-:|-:|
|Logistic Regression|0.9649|0.9960|0.9750|0.9286|0.9512|0.9245|
|Decision Tree|0.9298|0.9246|0.9048|0.9048|0.9048|0.8492|
|KNN|0.9561|0.9823|0.9744|0.9048|0.9383|0.9058|
|Gaussian Naive Bayes|0.9211|0.9891|0.9231|0.8571|0.8889|0.8292|
|**Random Forest (Ensemble)**|**0.9737**|**0.9929**|**1.0000**|**0.9286**|**0.9630**|**0.9442**|

---

## 5. Observations on Model Performance

|ML Model|Observation about Model Performance|
|-|-|
|**Logistic Regression**|Strong overall performance with 96.49% accuracy and 0.9960 AUC. It provides high precision and recall and is a strong baseline for this classification problem.|
|**Decision Tree**|Achieves 92.98% accuracy. Its performance is lower than the stronger models on this test set, with an AUC of 0.9246 and MCC of 0.8492.|
|**KNN**|Performs well with 95.61% accuracy and 0.9823 AUC. Precision is high at 0.9744, although recall is lower than Logistic Regression and Random Forest.|
|**Gaussian Naive Bayes**|Achieves 92.11% accuracy and 0.9891 AUC. Although the AUC is high, recall and F1 score are lower than those of the stronger models evaluated here.|
|**Random Forest (Ensemble)**|Provides the best overall results on this test set. It achieves the highest accuracy (97.37%), perfect precision (1.0000), strong recall (0.9286), highest F1 score (0.9630), and highest MCC (0.9442).|

### Overall Winner

**Random Forest** is the overall winner for this dataset based on the evaluated test-set results.

It provides the highest accuracy, precision, F1 score, and MCC among the five implemented models while maintaining a high AUC and recall.

---

## 6. Streamlit Deployment

The project was deployed using **Streamlit Community Cloud**.

### Live Streamlit Application

**Streamlit App:**  
https://breast-cancer-classification-ml-iod5mdtdatvkzae7eanxcf.streamlit.app/

### Application Features

The Streamlit application provides the assignment-required functionality:

* Upload test data in CSV format.
* Select a machine-learning model from a dropdown.
* Evaluate the selected model on the uploaded test data.
* Display Accuracy, AUC, Precision, Recall, F1 Score, and MCC.
* Display a confusion matrix.
* Display prediction/evaluation results for the supplied test data.
* Compare the five implemented models by changing the model selection.

### Expected CSV Format

The uploaded test CSV should contain the **30 model input features** used during training.

If the `Diagnosis` column is included, the application calculates the evaluation metrics and confusion matrix.

---

## 7. Running the Project Locally

### Clone the repository

```bash
git clone https://github.com/arunpraveencse/breast-cancer-classification-ml.git
cd breast-cancer-classification-ml
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the Streamlit application

```bash
streamlit run app.py
```

---

## 8. Saved Model Files

The trained models are stored in the `model/` directory:

* `logistic_regression.pkl`
* `decision_tree.pkl`
* `knn.pkl`
* `naive_bayes.pkl`
* `random_forest.pkl`
* `scaler.pkl`

The trained models and scaler are stored in the repository so that the Streamlit application can load them directly for inference without retraining during application execution.

---

## 9. Assignment Submission Evidence

The project satisfies the required submission components:

* **GitHub Repository:** Public and accessible.
* **Source Code:** `app.py`
* **Dependencies:** `requirements.txt`
* **Documentation:** `README.md`
* **Test Data:** `test_data.csv`
* **Saved Models:** `model` directory
* **Live Deployment:** Streamlit Community Cloud
* **BITS Virtual Lab Evidence:** Screenshot of assignment execution in the BITS Virtual Lab.

The README content is intended to be included in the final PDF submission as required by the assignment.

---

## 10. Conclusion

Five classification models were implemented and evaluated on the same breast-cancer dataset using six evaluation metrics.

**Random Forest performed best overall on the supplied test data**, achieving:

* **Accuracy:** 97.37%
* **AUC:** 99.29%
* **Precision:** 100.00%
* **Recall:** 92.86%
* **F1 Score:** 96.30%
* **MCC:** 94.42%

The project demonstrates an end-to-end machine-learning workflow covering dataset preparation, model evaluation, interactive application development, and deployment on Streamlit Community Cloud.

