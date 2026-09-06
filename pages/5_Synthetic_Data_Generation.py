import streamlit as st
import pandas as pd
from ctgan import CTGAN

st.set_page_config(page_title="Synthetic Data Generation", layout="wide")

st.title("🧬 Synthetic Data Generation using CTGAN")

# ----------------------------------------------------
# Load Dataset
# ----------------------------------------------------

if "clean_df" in st.session_state:
    df = st.session_state["clean_df"].copy()

elif "df" in st.session_state:
    df = st.session_state["df"].copy()

else:
    st.warning("Please upload the dataset first.")
    st.stop()

st.subheader("Original Dataset")

st.write(df.head())

st.write("Shape :", df.shape)

st.write("---")

# ----------------------------------------------------
# Train CTGAN
# ----------------------------------------------------

epochs = st.slider(
    "Select CTGAN Epochs",
    min_value=5,
    max_value=100,
    value=20
)

num_rows = st.number_input(
    "Number of Synthetic Records",
    min_value=100,
    max_value=200000,
    value=10000,
    step=100
)
if st.button("🚀 Train CTGAN"):

    with st.spinner("Training CTGAN... Please wait."):

        ctgan = CTGAN(epochs=epochs)

        ctgan.fit(df)

    st.success("✅ CTGAN Model Trained Successfully!")

    st.session_state["ctgan_model"] = ctgan

    st.write("---")

    st.subheader("Generating Synthetic Dataset...")

    synthetic_df = ctgan.sample(num_rows)

    st.session_state["synthetic_df"] = synthetic_df

    st.success("✅ Synthetic Dataset Generated Successfully!")

    st.write("---")

    st.subheader("Synthetic Dataset Preview")

    st.dataframe(synthetic_df.head())

    st.write("Synthetic Dataset Shape :", synthetic_df.shape)

    st.write("---")

    col1, col2 = st.columns(2)

    col1.metric(
        "Original Records",
        df.shape[0]
    )

    col2.metric(
        "Synthetic Records",
        synthetic_df.shape[0]
    )

    csv = synthetic_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇ Download Synthetic Dataset",
        csv,
        "synthetic_dataset.csv",
        "text/csv"
    )