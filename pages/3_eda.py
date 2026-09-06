import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="EDA", layout="wide")

st.title("📊 Exploratory Data Analysis (EDA)")

# -----------------------------
# Load Dataset
# -----------------------------
if "clean_df" in st.session_state:
    df = st.session_state["clean_df"]
elif "df" in st.session_state:
    df = st.session_state["df"]
else:
    st.warning("⚠ Please upload and preprocess the dataset first.")
    st.stop()

# -----------------------------
# Dataset Overview
# -----------------------------
st.header("📌 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Rows", df.shape[0])
col2.metric("Columns", df.shape[1])
col3.metric("Missing Values", int(df.isnull().sum().sum()))
col4.metric("Duplicate Rows", int(df.duplicated().sum()))

st.write("### First 5 Records")
st.dataframe(df.head())

st.write("### Data Types")
st.dataframe(
    df.dtypes.astype(str).reset_index().rename(
        columns={"index": "Column", 0: "Data Type"}
    )
)

# -----------------------------
# Missing Values
# -----------------------------
st.header("🧹 Missing Values")

missing = df.isnull().sum().sort_values(ascending=False)

st.dataframe(
    missing.reset_index().rename(
        columns={"index": "Column", 0: "Missing Values"}
    )
)

fig, ax = plt.subplots(figsize=(12,5))
missing.plot(kind="bar", ax=ax)
ax.set_ylabel("Missing Values")
ax.set_title("Missing Values per Column")
st.pyplot(fig)

# -----------------------------
# Statistical Summary
# -----------------------------
st.header("📈 Statistical Summary")

st.dataframe(df.describe())

st.header("📊 Univariate Analysis")

num_cols = df.select_dtypes(include="number").columns.tolist()

feature = st.selectbox(
    "Select Numerical Feature",
    num_cols,
    key="uni"
)

fig, ax = plt.subplots(figsize=(8,5))

sns.histplot(
    data=df,
    x=feature,
    kde=True,
    bins=30
)

plt.title(f"Distribution of {feature}")

st.pyplot(fig)



st.header("📈 Bivariate Analysis")

x = st.selectbox(
    "Select X-axis",
    num_cols,
    key="bi_x"
)

y = st.selectbox(
    "Select Y-axis",
    num_cols,
    key="bi_y"
)

fig, ax = plt.subplots(figsize=(8,6))

sns.scatterplot(
    data=df,
    x=x,
    y=y
)

plt.title(f"{x} vs {y}")

st.pyplot(fig)


st.header("📉 Multivariate Analysis")

selected_features = st.multiselect(
    "Select up to 4 numerical features",
    num_cols,
    default=num_cols[:4]
)

if len(selected_features) >= 2:
    sample = df[selected_features].sample(
        min(300, len(df)),
        random_state=42
    )

    fig = sns.pairplot(sample)
    st.pyplot(fig.figure)
else:
    st.info("Please select at least 2 features.")


st.header("📋 Categorical Analysis")

cat_cols = df.select_dtypes(include="object").columns.tolist()

if len(cat_cols) > 0:

    cat = st.selectbox(
        "Select Categorical Feature",
        cat_cols,
        key="cat"
    )

    fig, ax = plt.subplots(figsize=(8,5))

    sns.countplot(
        data=df,
        x=cat
    )

    plt.xticks(rotation=45)

    st.pyplot(fig)

else:
    st.info("No categorical columns available.")


st.header("📦 Outlier Analysis")

outlier_feature = st.selectbox(
    "Select Feature",
    num_cols,
    key="outlier"
)

fig, ax = plt.subplots(figsize=(8,5))

sns.boxplot(
    y=df[outlier_feature]
)

plt.title(f"Outlier Detection - {outlier_feature}")

st.pyplot(fig)


st.header("🏥 Disease-wise Analysis")

target_col = None

if "diagnosed_diabetes" in df.columns:
    target_col = "diagnosed_diabetes"
elif "diagnosis" in df.columns:
    target_col = "diagnosis"
elif "HeartDisease" in df.columns:
    target_col = "HeartDisease"

if target_col is not None:

    fig, ax = plt.subplots(figsize=(8,5))

    sns.countplot(
        data=df,
        x=target_col
    )

    plt.xticks(rotation=45)

    st.pyplot(fig)

    st.write("### Disease Counts")

    st.dataframe(df[target_col].value_counts())

else:
    st.info("No disease column found.")
# -----------------------------
# Correlation Heatmap
# -----------------------------
st.header("🔥 Correlation Matrix")

numeric_df = df.select_dtypes(include="number")

fig, ax = plt.subplots(figsize=(14,10))

sns.heatmap(
    numeric_df.corr(),
    cmap="coolwarm",
    annot=False,
    linewidths=0.5
)

st.pyplot(fig)

# -----------------------------
# Histogram
# -----------------------------
st.header("📊 Histogram")

num_cols = numeric_df.columns.tolist()

hist_col = st.selectbox(
    "Select Numerical Feature",
    num_cols
)

fig, ax = plt.subplots(figsize=(8,5))

sns.histplot(
    df[hist_col],
    kde=True,
    bins=30
)

plt.title(hist_col)

st.pyplot(fig)

# -----------------------------
# Distribution Plot
# -----------------------------
st.header("📈 Distribution Plot")

dist_col = st.selectbox(
    "Choose Feature",
    num_cols,
    key=100
)

fig, ax = plt.subplots(figsize=(8,5))

sns.kdeplot(
    df[dist_col],
    fill=True
)

plt.title(dist_col)

st.pyplot(fig)

# -----------------------------
# Box Plot
# -----------------------------
st.header("📦 Box Plot")

box_col = st.selectbox(
    "Select Feature",
    num_cols,
    key=101
)

fig, ax = plt.subplots(figsize=(8,5))

sns.boxplot(
    y=df[box_col]
)

plt.title(box_col)

st.pyplot(fig)

# -----------------------------
# Scatter Plot
# -----------------------------
st.header("📍 Scatter Plot")

x = st.selectbox(
    "X Axis",
    num_cols,
    key=200
)

y = st.selectbox(
    "Y Axis",
    num_cols,
    key=201
)

fig, ax = plt.subplots(figsize=(8,5))

sns.scatterplot(
    data=df,
    x=x,
    y=y
)

st.pyplot(fig)

# -----------------------------
# Count Plot
# -----------------------------
cat_cols = df.select_dtypes(include="object").columns.tolist()

if len(cat_cols) > 0:

    st.header("📊 Count Plot")

    cat = st.selectbox(
        "Select Categorical Feature",
        cat_cols
    )

    fig, ax = plt.subplots(figsize=(8,5))

    sns.countplot(
        data=df,
        x=cat
    )

    plt.xticks(rotation=45)

    st.pyplot(fig)

# -----------------------------
# Target Distribution
# -----------------------------
if "diagnosed_diabetes" in df.columns:

    st.header("🎯 Target Distribution")

    fig, ax = plt.subplots(figsize=(6,4))

    sns.countplot(
        data=df,
        x="diagnosed_diabetes"
    )

    st.pyplot(fig)

# -----------------------------
# Pair Plot
# -----------------------------
st.header("🔗 Pair Plot")

sample = numeric_df.sample(
    min(500, len(numeric_df)),
    random_state=42
)

pair = sns.pairplot(sample.iloc[:, :5])

st.pyplot(pair.figure)

# -----------------------------
# Correlation with Target
# -----------------------------
if "diagnosed_diabetes" in numeric_df.columns:

    st.header("⭐ Feature Correlation with Target")

    corr = numeric_df.corr()["diagnosed_diabetes"].sort_values(ascending=False)

    st.dataframe(corr)

    fig, ax = plt.subplots(figsize=(10,6))

    corr.plot(kind="bar", ax=ax)

    plt.ylabel("Correlation")

    st.pyplot(fig)

st.success("✅ Exploratory Data Analysis Completed Successfully.")