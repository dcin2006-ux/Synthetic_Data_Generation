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
# DATASET OVERVIEW
# ============================================================

st.subheader("📊 Original Dataset")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Original Records",
        f"{df.shape[0]:,}"
    )

with col2:
    st.metric(
        "Features",
        df.shape[1]
    )

st.dataframe(
    df.head(10),
    use_container_width=True
)

st.divider()

# ============================================================
# CTGAN SETTINGS
# ============================================================

st.subheader("⚙️ CTGAN Training Settings")

st.info(
    "💡 To keep the online dashboard stable, CTGAN uses a "
    "representative sample of up to 5,000 records."
)

epochs = st.slider(
    "CTGAN Epochs",
    min_value=5,
    max_value=10,
    value=5,
    step=1
)

num_rows = st.number_input(
    "Number of Synthetic Records",
    min_value=100,
    max_value=5000,
    value=5000,
    step=100
)

st.write("---")

# ============================================================
# TRAIN CTGAN
# ============================================================

if st.button(
    "🚀 Train CTGAN",
    use_container_width=True
):

    # --------------------------------------------------------
    # Select a smaller representative sample
    # --------------------------------------------------------

    train_size = min(5000, len(df))

    train_df = df.sample(
        n=train_size,
        random_state=42
    ).copy()

    st.info(
        f"📌 CTGAN will train using {train_size:,} records "
        f"for {epochs} epochs."
    )

    # --------------------------------------------------------
    # Train CTGAN
    # --------------------------------------------------------

    try:

        with st.spinner(
            "🧬 Training CTGAN... Please wait."
        ):

            ctgan = CTGAN(
                epochs=epochs
            )

            ctgan.fit(train_df)

        st.success(
            "✅ CTGAN Model Trained Successfully!"
        )

        # Save model
        st.session_state["ctgan_model"] = ctgan

    except Exception as e:

        st.error(
            "❌ CTGAN training could not be completed."
        )

        st.warning(
            "The deployed server may not have enough "
            "resources for CTGAN training."
        )

        st.code(
            str(e)
        )

        st.stop()

    st.divider()

    # ========================================================
    # GENERATE SYNTHETIC DATA
    # ========================================================

    st.subheader(
        "🧬 Generating Synthetic Dataset"
    )

    try:

        with st.spinner(
            "Generating synthetic records..."
        ):

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

        st.code(
            str(e)
        )

        st.stop()

    st.divider()

    # ========================================================
    # SYNTHETIC DATA PREVIEW
    # ========================================================

    st.subheader(
        "🔬 Synthetic Dataset Preview"
    )

    st.dataframe(
        synthetic_df.head(10),
        use_container_width=True
    )

    st.write(
        "Synthetic Dataset Shape:",
        synthetic_df.shape
    )

    st.divider()

    # ========================================================
    # DATASET COMPARISON
    # ========================================================

    st.subheader(
        "📊 Dataset Comparison"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Original Records",
            f"{df.shape[0]:,}"
        )

    with col2:
        st.metric(
            "CTGAN Training Records",
            f"{train_size:,}"
        )

    with col3:
        st.metric(
            "Synthetic Records",
            f"{synthetic_df.shape[0]:,}"
        )

    st.divider()

    # ========================================================
    # SYNTHETIC DATA INFORMATION
    # ========================================================

    st.subheader(
        "📋 Synthetic Dataset Information"
    )

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

    st.divider()

    # ========================================================
    # DOWNLOAD SYNTHETIC DATA
    # ========================================================

    st.subheader(
        "⬇️ Download Synthetic Dataset"
    )

    synthetic_csv = synthetic_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="📥 Download Synthetic CSV",
        data=synthetic_csv,
        file_name="synthetic_dataset.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.divider()

    # ========================================================
    # COMPLETION
    # ========================================================

    st.success(
        "🎉 Synthetic Data Generation Completed Successfully!"
    )

    st.info(
        "The generated synthetic dataset is now available "
        "for validation and downstream analysis."
    )


# ============================================================
# SHOW PREVIOUSLY GENERATED DATA
# ============================================================

elif "synthetic_df" in st.session_state:

    st.divider()

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

    synthetic_csv = synthetic_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="📥 Download Synthetic CSV",
        data=synthetic_csv,
        file_name="synthetic_dataset.csv",
        mime="text/csv",
        use_container_width=True
    )