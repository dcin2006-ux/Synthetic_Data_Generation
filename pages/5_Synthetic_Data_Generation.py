import streamlit as st
import pandas as pd
from ctgan import CTGAN

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Synthetic Data Generation",
    layout="wide"
)

st.title("🧬 Synthetic Data Generation using CTGAN")

st.write(
    "Generate synthetic healthcare records using "
    "Conditional Tabular GAN (CTGAN)."
)

# ============================================================
# LOAD DATASET
# ============================================================

if "clean_df" in st.session_state:

    df = st.session_state["clean_df"].copy()

elif "df" in st.session_state:

    df = st.session_state["df"].copy()

else:

    st.warning("⚠️ Please upload the dataset first.")
    st.stop()

# ============================================================
# ORIGINAL DATASET
# ============================================================

st.subheader("📊 Original Dataset")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Original Records",
        df.shape[0]
    )

with col2:
    st.metric(
        "Number of Features",
        df.shape[1]
    )

st.dataframe(
    df.head(10),
    use_container_width=True
)

st.write("---")

# ============================================================
# CTGAN SETTINGS
# ============================================================

st.subheader("⚙️ CTGAN Training Settings")

epochs = st.slider(
    "Select CTGAN Epochs",
    min_value=5,
    max_value=100,
    value=10,
    step=5
)

num_rows = st.number_input(
    "Number of Synthetic Records",
    min_value=100,
    max_value=200000,
    value=10000,
    step=100
)

st.info(
    "💡 For faster dashboard execution, CTGAN is trained on "
    "a representative sample of up to 10,000 records."
)

st.write("---")

# ============================================================
# TRAIN CTGAN
# ============================================================

if st.button("🚀 Train CTGAN", use_container_width=True):

    # --------------------------------------------------------
    # Select representative training sample
    # --------------------------------------------------------

    train_df = df.sample(
        n=min(10000, len(df)),
        random_state=42
    )

    st.info(
        f"CTGAN training sample: {len(train_df):,} records"
    )

    # --------------------------------------------------------
    # Train CTGAN
    # --------------------------------------------------------

    with st.spinner(
        "🧬 Training CTGAN... Please wait. "
        "This may take a few minutes."
    ):

        try:

            ctgan = CTGAN(
                epochs=epochs
            )

            ctgan.fit(train_df)

            st.success(
                "✅ CTGAN Model Trained Successfully!"
            )

            # Save model in Streamlit session
            st.session_state["ctgan_model"] = ctgan

        except Exception as e:

            st.error(
                "❌ CTGAN training failed."
            )

            st.exception(e)

            st.stop()

    st.write("---")

    # ========================================================
    # GENERATE SYNTHETIC DATA
    # ========================================================

    st.subheader("🧬 Generating Synthetic Dataset")

    with st.spinner(
        "Generating synthetic patient records..."
    ):

        try:

            synthetic_df = ctgan.sample(
                int(num_rows)
            )

            # Save synthetic dataset
            st.session_state["synthetic_df"] = synthetic_df

            st.success(
                "✅ Synthetic Dataset Generated Successfully!"
            )

        except Exception as e:

            st.error(
                "❌ Synthetic data generation failed."
            )

            st.exception(e)

            st.stop()

    st.write("---")

    # ========================================================
    # SYNTHETIC DATA PREVIEW
    # ========================================================

    st.subheader("🔬 Synthetic Dataset Preview")

    st.dataframe(
        synthetic_df.head(10),
        use_container_width=True
    )

    st.write(
        "Synthetic Dataset Shape:",
        synthetic_df.shape
    )

    st.write("---")

    # ========================================================
    # DATASET COMPARISON
    # ========================================================

    st.subheader("📊 Dataset Comparison")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Original Records",
            f"{df.shape[0]:,}"
        )

    with col2:

        st.metric(
            "Training Records",
            f"{len(train_df):,}"
        )

    with col3:

        st.metric(
            "Synthetic Records",
            f"{synthetic_df.shape[0]:,}"
        )

    st.write("---")

    # ========================================================
    # SYNTHETIC DATA INFORMATION
    # ========================================================

    st.subheader("📋 Synthetic Dataset Information")

    info_df = pd.DataFrame({
        "Column": synthetic_df.columns,
        "Data Type": synthetic_df.dtypes.astype(str).values,
        "Missing Values": synthetic_df.isnull().sum().values,
        "Unique Values": [
            synthetic_df[column].nunique()
            for column in synthetic_df.columns
        ]
    })

    st.dataframe(
        info_df,
        use_container_width=True
    )

    st.write("---")

    # ========================================================
    # DOWNLOAD SYNTHETIC DATASET
    # ========================================================

    st.subheader("⬇️ Download Synthetic Dataset")

    csv = synthetic_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="📥 Download Synthetic CSV",
        data=csv,
        file_name="synthetic_dataset.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.write("---")

    # ========================================================
    # COMPLETION MESSAGE
    # ========================================================

    st.success(
        "🎉 Synthetic data generation completed successfully!"
    )

    st.info(
        "The generated dataset can now be examined in the "
        "Validation and Results sections."
    )

# ============================================================
# SHOW PREVIOUSLY GENERATED DATA
# ============================================================

elif "synthetic_df" in st.session_state:

    st.write("---")

    st.subheader(
        "📦 Previously Generated Synthetic Dataset"
    )

    synthetic_df = st.session_state["synthetic_df"]

    st.dataframe(
        synthetic_df.head(10),
        use_container_width=True
    )

    st.write(
        "Synthetic Dataset Shape:",
        synthetic_df.shape
    )

    csv = synthetic_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="📥 Download Synthetic CSV",
        data=csv,
        file_name="synthetic_dataset.csv",
        mime="text/csv",
        use_container_width=True
    )