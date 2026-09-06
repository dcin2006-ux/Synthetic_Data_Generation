import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Results",
    layout="wide"
)

st.title("📥 Results & Downloads")

# =========================================================
# CHECK DATA
# =========================================================

if "synthetic_df" not in st.session_state:
    st.warning("⚠ Please generate synthetic data first.")
    st.stop()

synthetic_df = st.session_state["synthetic_df"]

if "df" in st.session_state:
    original_df = st.session_state["df"].copy()
else:
    original_df = None


# =========================================================
# DATASET SUMMARY
# =========================================================

st.header("📊 Dataset Summary")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Synthetic Records",
    synthetic_df.shape[0]
)

col2.metric(
    "Synthetic Features",
    synthetic_df.shape[1]
)

if original_df is not None:
    col3.metric(
        "Original Records",
        original_df.shape[0]
    )


# =========================================================
# SYNTHETIC DATA PREVIEW
# =========================================================

st.header("🧬 Synthetic Dataset Preview")

st.dataframe(
    synthetic_df.head(20),
    use_container_width=True
)


# =========================================================
# DATA TYPES
# =========================================================

st.header("📋 Synthetic Dataset Information")

info_df = pd.DataFrame({
    "Column": synthetic_df.columns,
    "Data Type": synthetic_df.dtypes.astype(str).values,
    "Missing Values": synthetic_df.isnull().sum().values,
    "Unique Values": [
        synthetic_df[col].nunique()
        for col in synthetic_df.columns
    ]
})

st.dataframe(
    info_df,
    use_container_width=True
)


# =========================================================
# REAL VS SYNTHETIC MEAN
# =========================================================

if original_df is not None:

    st.header("📈 Original vs Synthetic Mean")

    numeric_columns = original_df.select_dtypes(
        include="number"
    ).columns

    common_numeric = [
        col for col in numeric_columns
        if col in synthetic_df.columns
    ]

    if len(common_numeric) > 0:

        mean_comparison = pd.DataFrame({
            "Original Mean": original_df[
                common_numeric
            ].mean(),

            "Synthetic Mean": synthetic_df[
                common_numeric
            ].mean()
        })

        st.dataframe(
            mean_comparison,
            use_container_width=True
        )


# =========================================================
# DISTRIBUTION COMPARISON
# =========================================================

if original_df is not None:

    st.header("📊 Distribution Comparison")

    numeric_columns = original_df.select_dtypes(
        include="number"
    ).columns.tolist()

    common_numeric = [
        col for col in numeric_columns
        if col in synthetic_df.columns
    ]

    if common_numeric:

        selected_feature = st.selectbox(
            "Select Feature",
            common_numeric
        )

        fig, ax = plt.subplots(
            figsize=(9, 5)
        )

        sns.histplot(
            original_df[selected_feature],
            kde=True,
            stat="density",
            alpha=0.5,
            label="Original"
        )

        sns.histplot(
            synthetic_df[selected_feature],
            kde=True,
            stat="density",
            alpha=0.5,
            label="Synthetic"
        )

        ax.set_title(
            f"Original vs Synthetic - {selected_feature}"
        )

        ax.legend()

        st.pyplot(fig)


# =========================================================
# DOWNLOAD SYNTHETIC DATA
# =========================================================

st.header("⬇ Download Synthetic Dataset")

synthetic_csv = synthetic_df.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="📥 Download Synthetic CSV",
    data=synthetic_csv,
    file_name="synthetic_diabetes_data.csv",
    mime="text/csv"
)


# =========================================================
# DOWNLOAD VALIDATION REPORT
# =========================================================

if original_df is not None:

    numeric_columns = original_df.select_dtypes(
        include="number"
    ).columns

    common_numeric = [
        col for col in numeric_columns
        if col in synthetic_df.columns
    ]

    if common_numeric:

        validation_report = pd.DataFrame({
            "Feature": common_numeric,

            "Original Mean": original_df[
                common_numeric
            ].mean().values,

            "Synthetic Mean": synthetic_df[
                common_numeric
            ].mean().values,

            "Original Std": original_df[
                common_numeric
            ].std().values,

            "Synthetic Std": synthetic_df[
                common_numeric
            ].std().values
        })

        st.header("📄 Validation Report")

        st.dataframe(
            validation_report,
            use_container_width=True
        )

        report_csv = validation_report.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="📥 Download Validation Report",
            data=report_csv,
            file_name="validation_report.csv",
            mime="text/csv"
        )


# =========================================================
# PROJECT COMPLETION
# =========================================================

st.divider()

st.success(
    "🎉 Synthetic Data Generation and Validation Completed Successfully!"
)

st.info(
    "The generated synthetic dataset can be used for "
    "downstream analysis and machine learning experimentation "
    "without directly exposing the original patient records."
)