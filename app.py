
import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Breast Cancer Classification",
    page_icon="🩺",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🩺 Breast Cancer Classification")
st.write(
    "Compare five machine-learning models for benign and malignant "
    "breast-tumor classification using the provided test dataset."
)


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    models = {
        "Logistic Regression": joblib.load(
            "model/logistic_regression.pkl"
        ),

        "Decision Tree": joblib.load(
            "model/decision_tree.pkl"
        ),

        "KNN": joblib.load(
            "model/knn.pkl"
        ),

        "Gaussian Naive Bayes": joblib.load(
            "model/naive_bayes.pkl"
        ),

        "Random Forest": joblib.load(
            "model/random_forest.pkl"
        )
    }

    scaler = joblib.load("model/scaler.pkl")

    return models, scaler


models, scaler = load_models()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Model Configuration")

selected_model = st.sidebar.selectbox(
    "Select a Machine Learning Model",
    list(models.keys())
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Test CSV",
    type=["csv"]
)


# ============================================================
# INFORMATION
# ============================================================

st.sidebar.markdown("---")

st.sidebar.write("### Available Models")

for model_name in models.keys():
    st.sidebar.write(f"• {model_name}")


# ============================================================
# MAIN APPLICATION
# ============================================================

if uploaded_file is None:

    st.info(
        "Please upload the test_data.csv file using the sidebar "
        "to start the evaluation."
    )

    st.markdown("### Required CSV format")

    st.write(
        "The uploaded file should contain the 30 model features. "
        "If the `Diagnosis` column is present, the application "
        "will also calculate evaluation metrics."
    )

else:

    # --------------------------------------------------------
    # Read uploaded CSV
    # --------------------------------------------------------

    data = pd.read_csv(uploaded_file)

    st.success("Test data uploaded successfully.")

    st.subheader("Uploaded Dataset")

    st.write(
        f"Dataset contains **{data.shape[0]} rows** and "
        f"**{data.shape[1]} columns**."
    )

    st.dataframe(data.head())


    # --------------------------------------------------------
    # Separate target and features
    # --------------------------------------------------------

    target_column = "Diagnosis"

    if target_column in data.columns:

        y_true = data[target_column]

        X = data.drop(columns=[target_column])

    else:

        y_true = None

        X = data


    # --------------------------------------------------------
    # Validate feature columns
    # --------------------------------------------------------

    expected_features = list(
        scaler.feature_names_in_
    )

    missing_features = [
        feature for feature in expected_features
        if feature not in X.columns
    ]

    extra_features = [
        feature for feature in X.columns
        if feature not in expected_features
    ]


    if missing_features:

        st.error(
            "The uploaded dataset is missing required features:"
        )

        st.write(missing_features)

        st.stop()


    # Keep only the features used during model training
    X = X[expected_features]


    # --------------------------------------------------------
    # Select model
    # --------------------------------------------------------

    model = models[selected_model]


    # --------------------------------------------------------
    # Apply scaling when required
    # --------------------------------------------------------

    scaled_models = [
        "Logistic Regression",
        "KNN",
        "Gaussian Naive Bayes"
    ]

    if selected_model in scaled_models:

        X_model = scaler.transform(X)

    else:

        X_model = X


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    y_pred = model.predict(X_model)

    y_prob = None

    if hasattr(model, "predict_proba"):

        y_prob = model.predict_proba(X_model)[:, 1]


    # ========================================================
    # PREDICTIONS
    # ========================================================

    st.subheader(
        f"Predictions — {selected_model}"
    )

    prediction_output = X.copy()

    prediction_output["Predicted Diagnosis"] = y_pred

    prediction_output["Predicted Class"] = np.where(
        y_pred == 1,
        "Malignant",
        "Benign"
    )

    if y_prob is not None:

        prediction_output["Malignant Probability"] = y_prob


    st.dataframe(
        prediction_output,
        use_container_width=True
    )


    # ========================================================
    # EVALUATION
    # ========================================================

    if y_true is not None:

        st.subheader(
            f"Evaluation Metrics — {selected_model}"
        )

        accuracy = accuracy_score(
            y_true,
            y_pred
        )

        precision = precision_score(
            y_true,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_true,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_true,
            y_pred,
            zero_division=0
        )

        mcc = matthews_corrcoef(
            y_true,
            y_pred
        )

        if y_prob is not None:

            auc = roc_auc_score(
                y_true,
                y_prob
            )

        else:

            auc = np.nan


        # ----------------------------------------------------
        # Display metrics
        # ----------------------------------------------------

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Accuracy",
            f"{accuracy:.4f}"
        )

        col2.metric(
            "AUC",
            f"{auc:.4f}"
        )

        col3.metric(
            "Precision",
            f"{precision:.4f}"
        )


        col4, col5, col6 = st.columns(3)

        col4.metric(
            "Recall",
            f"{recall:.4f}"
        )

        col5.metric(
            "F1 Score",
            f"{f1:.4f}"
        )

        col6.metric(
            "MCC",
            f"{mcc:.4f}"
        )


        # ====================================================
        # CONFUSION MATRIX
        # ====================================================

        st.subheader("Confusion Matrix")

        cm = confusion_matrix(
            y_true,
            y_pred
        )

        fig, ax = plt.subplots()

        ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Benign", "Malignant"]
        ).plot(
            ax=ax,
            colorbar=False
        )

        ax.set_title(
            f"{selected_model} - Confusion Matrix"
        )

        st.pyplot(fig)

        plt.close(fig)


        # ====================================================
        # CLASSIFICATION REPORT
        # ====================================================

        st.subheader("Classification Report")

        report = classification_report(
            y_true,
            y_pred,
            target_names=["Benign", "Malignant"],
            output_dict=True,
            zero_division=0
        )

        report_df = pd.DataFrame(report).transpose()

        st.dataframe(
            report_df,
            use_container_width=True
        )


        # ====================================================
        # MODEL COMPARISON
        # ====================================================

        st.subheader("Model Comparison")

        comparison_data = {
            "Model": [
                "Logistic Regression",
                "Decision Tree",
                "KNN",
                "Gaussian Naive Bayes",
                "Random Forest"
            ],

            "Accuracy": [
                0.964912,
                0.929825,
                0.956140,
                0.921053,
                0.973684
            ],

            "AUC": [
                0.996032,
                0.924603,
                0.982308,
                0.989087,
                0.992890
            ],

            "Precision": [
                0.975000,
                0.904762,
                0.974359,
                0.923077,
                1.000000
            ],

            "Recall": [
                0.928571,
                0.904762,
                0.904762,
                0.857143,
                0.928571
            ],

            "F1": [
                0.951220,
                0.904762,
                0.938272,
                0.888889,
                0.962963
            ],

            "MCC": [
                0.924518,
                0.849206,
                0.905824,
                0.829162,
                0.944155
            ]
        }

        comparison_df = pd.DataFrame(
            comparison_data
        )

        st.dataframe(
            comparison_df,
            use_container_width=True
        )

