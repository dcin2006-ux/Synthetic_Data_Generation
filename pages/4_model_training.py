import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

st.set_page_config(
    page_title="Model Training",
    layout="wide"
)

st.title("🤖 Random Forest Model Training")

# --------------------------------------------------------
# Load Cleaned Dataset
# --------------------------------------------------------

if "clean_df" in st.session_state:
    df = st.session_state["clean_df"].copy()

elif "df" in st.session_state:
    df = st.session_state["df"].copy()

else:
    st.warning("⚠ Please upload and preprocess the dataset first.")
    st.stop()

# --------------------------------------------------------
# Check Target Column
# --------------------------------------------------------

target = "diagnosed_diabetes"

if target not in df.columns:
    st.error(f"'{target}' column not found.")
    st.stop()

# --------------------------------------------------------
# Features and Target
# --------------------------------------------------------

X = df.drop(columns=[target])

y = df[target]

# --------------------------------------------------------
# Dataset Information
# --------------------------------------------------------

st.subheader("📋 Dataset Information")

col1, col2, col3 = st.columns(3)

col1.metric("Rows", df.shape[0])
col2.metric("Columns", df.shape[1])
col3.metric("Target", target)

st.write("---")

# --------------------------------------------------------
# Train Test Split
# --------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

st.success("✅ Dataset Split Successfully")

st.write("Training Samples :", X_train.shape[0])
st.write("Testing Samples :", X_test.shape[0])

st.write("---")

# --------------------------------------------------------
# Train Model Button
# --------------------------------------------------------

train = st.button("🚀 Train Random Forest Model")

if train:

    with st.spinner("Training Random Forest Model... Please Wait"):

        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

    st.success("✅ Model Training Completed")

    # -----------------------------------------
    # Save Model
    # -----------------------------------------

    st.session_state["model"] = model

    # -----------------------------------------
    # Metrics
    # -----------------------------------------

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(y_test, y_pred)

    recall = recall_score(y_test, y_pred)

    f1 = f1_score(y_test, y_pred)

    st.header("📊 Model Performance")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Accuracy", f"{accuracy:.4f}")

    c2.metric("Precision", f"{precision:.4f}")

    c3.metric("Recall", f"{recall:.4f}")

    c4.metric("F1 Score", f"{f1:.4f}")

    st.write("---")

    # -----------------------------------------
    # Classification Report
    # -----------------------------------------

    st.subheader("📄 Classification Report")

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True
    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(report_df)

    # -----------------------------------------
    # Save for Next Page
    # -----------------------------------------

    st.session_state["X_test"] = X_test
    st.session_state["y_test"] = y_test
    st.session_state["y_pred"] = y_pred
    # --------------------------------------------------------
# Confusion Matrix
# --------------------------------------------------------

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay
)

st.write("---")
st.header("🧩 Confusion Matrix")

fig, ax = plt.subplots(figsize=(6,6))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    cmap="Blues",
    ax=ax
)

st.pyplot(fig)

# --------------------------------------------------------
# ROC Curve
# --------------------------------------------------------

st.write("---")
st.header("📈 ROC Curve")

fig, ax = plt.subplots(figsize=(7,5))

RocCurveDisplay.from_estimator(
    model,
    X_test,
    y_test,
    ax=ax
)

st.pyplot(fig)

# --------------------------------------------------------
# Precision Recall Curve
# --------------------------------------------------------

st.write("---")
st.header("📉 Precision Recall Curve")

fig, ax = plt.subplots(figsize=(7,5))

PrecisionRecallDisplay.from_estimator(
    model,
    X_test,
    y_test,
    ax=ax
)

st.pyplot(fig)

# --------------------------------------------------------
# Feature Importance
# --------------------------------------------------------

st.write("---")
st.header("⭐ Feature Importance")

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

st.dataframe(importance)

fig, ax = plt.subplots(figsize=(10,6))

sns.barplot(
    data=importance.head(10),
    x="Importance",
    y="Feature",
    ax=ax
)

plt.title("Top 10 Important Features")

st.pyplot(fig)

# --------------------------------------------------------
# Download Model
# --------------------------------------------------------

st.write("---")
st.header("💾 Download Trained Model")

joblib.dump(model, "random_forest.pkl")

with open("random_forest.pkl", "rb") as file:

    st.download_button(
        label="⬇ Download Random Forest Model",
        data=file,
        file_name="random_forest.pkl",
        mime="application/octet-stream"
    )

# --------------------------------------------------------
# Model Summary
# --------------------------------------------------------

st.write("---")
st.header("📋 Model Summary")

summary = pd.DataFrame({
    "Metric": [
        "Algorithm",
        "Training Samples",
        "Testing Samples",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ],
    "Value": [
        "Random Forest",
        X_train.shape[0],
        X_test.shape[0],
        round(accuracy,4),
        round(precision,4),
        round(recall,4),
        round(f1,4)
    ]
})

st.table(summary)

st.success("✅ Random Forest Model Training Completed Successfully!")