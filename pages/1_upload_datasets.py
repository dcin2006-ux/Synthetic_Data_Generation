import streamlit as st
import pandas as pd

st.set_page_config(page_title="Upload Dataset", layout="wide")

st.title("📂 Upload Dataset")

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    # Store dataset globally
    st.session_state["df"] = df

    st.success("✅ Dataset Uploaded Successfully!")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    col1, col2 = st.columns(2)

    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])

    st.subheader("Column Names")
    st.write(df.columns.tolist())

    st.subheader("Data Types")
    st.dataframe(df.dtypes.astype(str))

    st.subheader("Missing Values")
    st.dataframe(df.isnull().sum())

    st.subheader("Duplicate Rows")
    st.write(df.duplicated().sum())