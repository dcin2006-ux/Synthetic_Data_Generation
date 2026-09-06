import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Model Training",
    layout="wide"
)

st.title("🤖 Random Forest Model Training")

st.write(
    "Train a Random Forest classifier to evaluate "
    "the prepared diabetes dataset."
)

# ============================================================
# LOAD DATASET
# ============================================================

if "clean_df" in st.session_state:

    df = st.session_state["clean_df"].copy()

elif "df" in st.session_state:

    df = st.session_state["df"].copy()

else:

    st.warning(
        "⚠️ Please upload and preprocess the dataset first."
    )

    st.stop()

# ============================================================
# TARGET COLUMN
# ============================================================

target = "diagnosed_diabetes"

if target not in df.columns:

    st.error(
        f"❌ Target column '{target}' was not found."
    )

    st.stop()

# ============================================================
# DATASET INFORMATION
# ============================================================

st.subheader("📋 Dataset Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Rows",
        f"{df.shape[0]:,}"
    )

with col2:
    st.metric(
        "Columns",
        df.shape[1]
    )

with col3:
    st.metric(
        "Target",
        target
    )

st.divider()

# ============================================================
# PREPARE X AND Y
# ============================================================

X = df.drop(columns=[target])
y = df[target]

# Convert target to numeric if required
try:
    y = pd.to_numeric(y)
except:
    st.error(
        "❌ Target column must contain numeric values."
    )
    st.stop()

# ============================================================
# CHECK FEATURE TYPES
# ============================================================

non_numeric_columns = X.select_dtypes(
    include=["object", "string", "category"]
).columns.tolist()

if len(non_numeric_columns) > 0:

    st.warning(
        "⚠️ Some categorical columns are still present. "
        "They will be automatically encoded."
    )

    X = pd.get_dummies(
        X,
        columns=non_numeric_columns,
        drop_first=True
    )

# Make sure everything is numeric
X = X.apply(
    pd.to_numeric,
    errors="coerce"
)

# Replace missing values
X = X.fillna(
    X.median(numeric_only=True)
)

# Any remaining missing values
X = X.fillna(0)

# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

st.subheader("✂️ Dataset Split")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Training Samples",
        f"{X_train.shape[0]:,}"
    )

with col2:
    st.metric(
        "Testing Samples",
        f"{X_test.shape[0]:,}"
    )

st.divider()

# ============================================================
# TRAIN MODEL BUTTON
# ============================================================

if st.button(
    "🚀 Train Random Forest Model",
    use_container_width=True
):

    # --------------------------------------------------------
    # TRAIN RANDOM FOREST
    # --------------------------------------------------------

    with st.spinner(
        "🤖 Training Random Forest Model..."
    ):

        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )

        model.fit(
            X_train,
            y_train
        )

        # IMPORTANT:
        # y_pred is created INSIDE the training block
        y_pred = model.predict(
            X_test
        )

    st.success(
        "✅ Random Forest Model Training Completed!"
    )

    # ========================================================
    # SAVE MODEL AND RESULTS
    # ========================================================

    st.session_state["model"] = model
    st.session_state["X_test"] = X_test
    st.session_state["y_test"] = y_test
    st.session_state["y_pred"] = y_pred

    # ========================================================
    # CALCULATE METRICS
    # ========================================================

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    # ========================================================
    # MODEL PERFORMANCE
    # ========================================================

    st.header("📊 Model Performance")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Accuracy",
            f"{accuracy:.4f}"
        )

    with c2:
        st.metric(
            "Precision",
            f"{precision:.4f}"
        )

    with c3:
        st.metric(
            "Recall",
            f"{recall:.4f}"
        )

    with c4:
        st.metric(
            "F1 Score",
            f"{f1:.4f}"
        )

    st.divider()

    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    st.subheader(
        "📄 Classification Report"
    )

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(
        report
    ).transpose()

    st.dataframe(
        report_df,
        width="stretch"
    )

    st.divider()

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    st.header("🧩 Confusion Matrix")

    fig, ax = plt.subplots(
        figsize=(6, 6)
    )

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        ax=ax
    )

    ax.set_title(
        "Random Forest Confusion Matrix"
    )

    st.pyplot(
        fig,
        clear_figure=True
    )

    plt.close(fig)

    st.divider()

    # ========================================================
    # ROC CURVE
    # ========================================================

    st.header("📈 ROC Curve")

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    RocCurveDisplay.from_estimator(
        model,
        X_test,
        y_test,
        ax=ax
    )

    ax.set_title(
        "Random Forest ROC Curve"
    )

    st.pyplot(
        fig,
        clear_figure=True
    )

    plt.close(fig)

    st.divider()

    # ========================================================
    # PRECISION-RECALL CURVE
    # ========================================================

    st.header(
        "📉 Precision-Recall Curve"
    )

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    PrecisionRecallDisplay.from_estimator(
        model,
        X_test,
        y_test,
        ax=ax
    )

    ax.set_title(
        "Precision-Recall Curve"
    )

    st.pyplot(
        fig,
        clear_figure=True
    )

    plt.close(fig)

    st.divider()

    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    st.header(
        "⭐ Feature Importance"
    )

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )

    st.dataframe(
        importance,
        width="stretch"
    )

    # --------------------------------------------------------
    # TOP 10 FEATURES
    # --------------------------------------------------------

    st.subheader(
        "🏆 Top 10 Important Features"
    )

    top_features = importance.head(10)

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    sns.barplot(
        data=top_features,
        x="Importance",
        y="Feature",
        ax=ax
    )

    ax.set_title(
        "Top 10 Feature Importance"
    )

    st.pyplot(
        fig,
        clear_figure=True
    )

    plt.close(fig)

    st.divider()

    # ========================================================
    # MODEL SUMMARY
    # ========================================================

    st.header(
        "📋 Model Summary"
    )

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
            str(X_train.shape[0]),
            str(X_test.shape[0]),
            f"{accuracy:.4f}",
            f"{precision:.4f}",
            f"{recall:.4f}",
            f"{f1:.4f}"
        ]
    })

    # Everything in Value is converted to string.
    # This avoids the ArrowTypeError seen in Cloud.

    st.dataframe(
        summary,
        width="stretch"
    )

    st.divider()

    # ========================================================
    # DOWNLOAD MODEL
    # ========================================================

    st.header(
        "💾 Download Trained Model"
    )

    import joblib

    model_bytes = joblib.dump(
        model,
        "random_forest.pkl"
    )

    with open(
        "random_forest.pkl",
        "rb"
    ) as file:

        model_data = file.read()

    st.download_button(
        label="⬇️ Download Random Forest Model",
        data=model_data,
        file_name="random_forest.pkl",
        mime="application/octet-stream",
        use_container_width=True
    )

    st.divider()

    st.success(
        "🎉 Model training and evaluation completed successfully!"
    )


# ============================================================
# SHOW PREVIOUS RESULTS
# ============================================================

elif (
    "model" in st.session_state
    and "y_test" in st.session_state
    and "y_pred" in st.session_state
):

    st.info(
        "✅ A trained Random Forest model is already available "
        "in this session."
    )

    y_test_saved = st.session_state["y_test"]
    y_pred_saved = st.session_state["y_pred"]

    accuracy_saved = accuracy_score(
        y_test_saved,
        y_pred_saved
    )

    precision_saved = precision_score(
        y_test_saved,
        y_pred_saved,
        zero_division=0
    )

    recall_saved = recall_score(
        y_test_saved,
        y_pred_saved,
        zero_division=0
    )

    f1_saved = f1_score(
        y_test_saved,
        y_pred_saved,
        zero_division=0
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Accuracy",
        f"{accuracy_saved:.4f}"
    )

    c2.metric(
        "Precision",
        f"{precision_saved:.4f}"
    )

    c3.metric(
        "Recall",
        f"{recall_saved:.4f}"
    )

    c4.metric(
        "F1 Score",
        f"{f1_saved:.4f}"
    )
    