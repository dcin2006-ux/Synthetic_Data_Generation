import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Synthetic Data Generation Dashboard",
    page_icon="🏥",
    layout="wide"
)

# Sidebar
st.sidebar.title("🏥 Dashboard")
st.sidebar.success("Select a page from the sidebar.")

# Main Title
st.title("🏥 Synthetic Data Generation for Rare Disease Research")

st.markdown("---")

st.header("📌 Project Objective")

st.write("""
This dashboard demonstrates a complete data science pipeline for generating
privacy-preserving synthetic healthcare data using CTGAN.

The dashboard allows users to:

✔ Upload a healthcare dataset

✔ Perform preprocessing

✔ Perform Exploratory Data Analysis (EDA)

✔ Train a Random Forest model

✔ Evaluate model performance

✔ Generate synthetic patient data using CTGAN

✔ Compare Real vs Synthetic data

✔ Download generated synthetic datasets
""")

st.markdown("---")

st.header("🛠 Technologies Used")

col1, col2, col3 = st.columns(3)

with col1:
    st.info("""
    **Programming**
    - Python
    - Streamlit
    """)

with col2:
    st.info("""
    **Machine Learning**
    - Random Forest
    - Scikit-Learn
    """)

with col3:
    st.info("""
    **Synthetic Data**
    - CTGAN
    - Pandas
    """)

st.markdown("---")

st.header("📊 Project Workflow")

st.write("""
1. Upload Dataset

2. Data Preprocessing

3. Exploratory Data Analysis

4. Random Forest Model Training

5. Model Evaluation

6. Synthetic Data Generation (CTGAN)

7. Validation

8. Download Results
""")

st.markdown("---")

st.success("✅ Dashboard is Ready. Select a page from the sidebar to continue.")