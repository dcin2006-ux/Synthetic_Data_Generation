import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Preprocessing", layout="wide")

st.title("🧹 Data Preprocessing")

# Check if dataset is uploaded
if "df" not in st.session_state:
    st.warning("⚠ Please upload a dataset first.")
    st.stop()

df = st.session_state["df"].copy()

st.subheader("📌 Original Dataset")
st.write(f"Rows: {df.shape[0]}")
st.write(f"Columns: {df.shape[1]}")

# -----------------------------
# Missing Values
# -----------------------------
st.subheader("1️⃣ Missing Values")

missing = df.isnull().sum()

st.dataframe(
    missing[missing > 0].reset_index().rename(
        columns={
            "index":"Column",
            0:"Missing Values"
        }
    )
)

numeric_cols = df.select_dtypes(include=['number']).columns
categorical_cols = df.select_dtypes(include=['object']).columns

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

st.success("✅ Missing values handled")

# -----------------------------
# Duplicate Values
# -----------------------------
st.subheader("2️⃣ Duplicate Values")

duplicates = df.duplicated().sum()

st.write("Duplicate Rows Before:", duplicates)

df = df.drop_duplicates()

st.write("Duplicate Rows After:", df.duplicated().sum())

# -----------------------------
# Remove Leakage Columns
# -----------------------------
st.subheader("3️⃣ Remove Leakage Columns")

leakage_columns = [
    "diabetes_stage",
    "diabetes_risk_score"
]

existing = [col for col in leakage_columns if col in df.columns]

if existing:
    df = df.drop(columns=existing)
    st.success(f"Removed: {existing}")
else:
    st.info("No leakage columns found.")

# -----------------------------
# Encode Categorical Columns
# -----------------------------
st.subheader("4️⃣ Encode Categorical Columns")

encoder = LabelEncoder()

cat_cols = df.select_dtypes(include="object").columns

for col in cat_cols:
    df[col] = encoder.fit_transform(df[col])

st.success("Categorical columns encoded")

# -----------------------------
# Final Dataset
# -----------------------------
st.subheader("5️⃣ Final Dataset Shape")

st.write(df.shape)

st.subheader("Preview")

st.dataframe(df.head())

# Save cleaned data
st.session_state["clean_df"] = df

# Download Button
csv = df.to_csv(index=False).encode()

st.download_button(
    "⬇ Download Cleaned Dataset",
    csv,
    "cleaned_dataset.csv",
    "text/csv"
)