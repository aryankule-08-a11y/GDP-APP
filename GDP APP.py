# -----------------------------------------------------------
# Streamlit Data Analysis Web App
# -----------------------------------------------------------
# Features:
# • Upload CSV
# • Auto-detect numerical & categorical columns
# • Data Cleaning + Missing value handling
# • Summary statistics
# • Trend Line Chart
# • Bar chart for categorical comparison
# • Scatter Plot
# • Correlation Heatmap
# • Sidebar filters (dynamic)
# • Download cleaned CSV
# -----------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="Data Analysis App", layout="wide")

# -----------------------------------------------------------
# TITLE
# -----------------------------------------------------------
st.title("📊 Universal Data Analysis App")
st.write("Upload your CSV file and explore data insights, charts, and relationships.")

# -----------------------------------------------------------
# FILE UPLOADER
# -----------------------------------------------------------
uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📁 Raw Dataset Preview")
    st.dataframe(df.head())

    # -----------------------------------------------------------
    # DATA CLEANING
    # -----------------------------------------------------------
    st.subheader("🧹 Data Cleaning")

    st.write("**Handling Missing Values:**")
    st.write("Missing values are filled using forward-fill and then backward-fill.")

    df_clean = df.copy()
    df_clean = df_clean.ffill().bfill()

    st.write("✅ Missing values handled successfully.")
    st.write("### Cleaned Dataset Preview")
    st.dataframe(df_clean.head())

    # -----------------------------------------------------------
    # DATA TYPES + AUTO DETECTION
    # -----------------------------------------------------------
    st.subheader("🔍 Column Type Detection")

    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df_clean.select_dtypes(exclude=[np.number]).columns.tolist()

    st.write("**Numeric Columns:**", numeric_cols)
    st.write("**Categorical Columns:**", categorical_cols)

    # -----------------------------------------------------------
    # SUMMARY STATISTICS
    # -----------------------------------------------------------
    st.subheader("📈 Summary Statistics")
    st.dataframe(df_clean.describe())

    # -----------------------------------------------------------
    # SIDEBAR FILTERS
    # -----------------------------------------------------------
    st.sidebar.header("Filter Data")

    filtered_df = df_clean.copy()

    for col in categorical_cols:
        unique_values = filtered_df[col].dropna().unique().tolist()
        selected = st.sidebar.multiselect(f"Filter by {col}", unique_values, default=unique_values)
        filtered_df = filtered_df[filtered_df[col].isin(selected)]

    # -----------------------------------------------------------
    # LINE CHART (TRENDS)
    # -----------------------------------------------------------
    st.subheader("📉 Trend Line Chart")

    if len(numeric_cols) >= 2:
        x_col = st.selectbox("Select X-axis (usually Year/Time)", numeric_cols)
        y_col = st.selectbox("Select Y-axis for Trend", numeric_cols)

        fig_line = px.line(filtered_df, x=x_col, y=y_col, title=f"Trend of {y_col} over {x_col}")
        st.plotly_chart(fig_line, use_container_width=True)

        st.write(f"**Insight:** This chart shows how `{y_col}` changes with `{x_col}` over time or sequence.")

    # -----------------------------------------------------------
    # BAR CHART (CATEGORICAL COMPARISON)
    # -----------------------------------------------------------
    st.subheader("📊 Bar Chart – Categorical Comparison")

    if categorical_cols and numeric_cols:
        cat_col = st.selectbox("Categorical Column", categorical_cols)
        num_col = st.selectbox("Numeric Column for Comparison", numeric_cols)

        fig_bar = px.bar(
            filtered_df.groupby(cat_col)[num_col].mean().reset_index(),
            x=cat_col, y=num_col,
            title=f"Average {num_col} by {cat_col}"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        st.write(f"**Insight:** This chart compares average `{num_col}` across categories of `{cat_col}`.")

    # -----------------------------------------------------------
    # SCATTER PLOT (RELATIONSHIPS)
    # -----------------------------------------------------------
    st.subheader("⚖️ Scatter Plot – Relationship Between Variables")

    if len(numeric_cols) >= 2:
        x_scatter = st.selectbox("Scatter X-axis", numeric_cols, key="scatter_x")
        y_scatter = st.selectbox("Scatter Y-axis", numeric_cols, key="scatter_y")

        fig_scatter = px.scatter(filtered_df, x=x_scatter, y=y_scatter, trendline="ols",
                                 title=f"Scatter Plot of {x_scatter} vs {y_scatter}")
        st.plotly_chart(fig_scatter, use_container_width=True)

        st.write(f"**Insight:** This plot shows how `{y_scatter}` changes with `{x_scatter}`. "
                 "The OLS line helps visualize correlation.")

    # -----------------------------------------------------------
    # CORRELATION HEATMAP
    # -----------------------------------------------------------
    st.subheader("🔥 Correlation Heatmap")

    if len(numeric_cols) >= 2:
        corr = filtered_df[numeric_cols].corr()

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

        st.write("**Insight:** Darker colors represent stronger relationships between numeric variables.")

    # -----------------------------------------------------------
    # DOWNLOAD CLEANED CSV
    # -----------------------------------------------------------
    st.subheader("⬇️ Download Cleaned Dataset")

    csv_buffer = io.BytesIO()
    df_clean.to_csv(csv_buffer, index=False)
    st.download_button("Download Cleaned CSV", data=csv_buffer.getvalue(),
                       file_name="cleaned_dataset.csv", mime="text/csv")

else:
    st.info("👆 Upload a CSV file to begin analysis.")
