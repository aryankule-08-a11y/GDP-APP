import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="CSV Analyzer", layout="wide")

st.title("CSV Analyzer — Works on Streamlit & GitHub")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"Loaded successfully! Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    except Exception as e:
        st.error("Error reading CSV file. Please upload a valid file.")
        st.stop()

    st.subheader("Preview Data")
    st.dataframe(df.head())

    st.subheader("Basic Statistics")
    st.write(df.describe(include="all"))

    st.subheader("Column Selection for Plot")
    numeric_columns = df.select_dtypes(include='number').columns.tolist()

    if len(numeric_columns) == 0:
        st.warning("No numeric columns found for plotting.")
    else:
        x_col = st.selectbox("X-axis", numeric_columns)
        y_col = st.selectbox("Y-axis", numeric_columns)

        fig = px.scatter(df, x=x_col, y=y_col, title=f"Scatter Plot: {x_col} vs {y_col}")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Download Processed CSV")
    st.download_button(
        label="Download CSV",
        data=df.to_csv(index=False).encode(),
        file_name="processed_data.csv",
        mime="text/csv"
    )

else:
    st.info("Please upload a CSV file to begin.")



