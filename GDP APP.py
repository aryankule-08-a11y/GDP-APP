import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ============ PAGE CONFIG ============
st.set_page_config(
    page_title="Indian Budget Analysis 2014–2025",
    page_icon="📊",
    layout="wide"
)

# ============ HEADER UI ============
st.title("📊 Budget Analysis Dashboard (2014–2025)")
st.markdown("""
This dashboard provides **deep insights**, **trends**, **comparisons**, and **correlations** 
from India's department-wise budget allocation for the period **2014–2025**.
---
""")

# ============ LOAD DATA ============
uploaded = st.file_uploader("Upload Budget CSV File", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)

    # Rename column for consistency
    df.rename(columns={"Department Name": "Department"}, inplace=True)

    year_cols = [col for col in df.columns if col != "Department"]

    # Sidebar filters
    st.sidebar.header("🔍 Filters")

    selected_departments = st.sidebar.multiselect(
        "Select Departments",
        df["Department"].unique(),
        default=df["Department"].unique()
    )

    df_filtered = df[df["Department"].isin(selected_departments)]

    st.sidebar.write("### Display Options")
    show_raw = st.sidebar.checkbox("Show Raw Data")

    # ================= RAW DATA ==================
    if show_raw:
        st.subheader("📄 Raw Uploaded Data")
        st.dataframe(df_filtered, use_container_width=True)

    # ================= TOTAL BUDGET OVER YEARS ==================
    st.subheader("📈 Total Budget Trend (2014–2025)")

    total_by_year = df_filtered[year_cols].sum()

    fig = px.line(
        x=total_by_year.index,
        y=total_by_year.values,
        markers=True,
        title="Total Budget Over Years",
        labels={"x": "Year", "y": "Total Budget (Cr ₹)"}
    )
    st.plotly_chart(fig, use_container_width=True)

    # ================= YOY GROWTH ==================
    st.subheader("📉 Year-on-Year (YoY) Growth")

    yoy = total_by_year.pct_change() * 100
    yoy_df = pd.DataFrame({"Year": yoy.index, "YoY Growth %": yoy.values})

    fig_yoy = px.bar(
        yoy_df,
        x="Year",
        y="YoY Growth %",
        title="Year-on-Year Growth in Total Budget",
        color="YoY Growth %",
        text="YoY Growth %"
    )
    st.plotly_chart(fig_yoy, use_container_width=True)

    # ================= DEPARTMENT WISE TREND ==================
    st.subheader("🏛 Department-wise Budget Trend")

    dept_choice = st.selectbox("Select Department", df["Department"].unique())

    dept_data = df[df["Department"] == dept_choice].iloc[0][year_cols]

    fig_dept = px.line(
        x=year_cols,
        y=dept_data.values,
        markers=True,
        title=f"{dept_choice} — Budget Trend",
        labels={"x": "Year", "y": "Budget (Cr ₹)"}
    )
    st.plotly_chart(fig_dept, use_container_width=True)

    # ================= TOP SPENDING DEPARTMENTS ==================
    st.subheader("🏆 Top 10 Highest Budget Departments (2025)")

    df["2025"] = df["2025"].astype(float)
    top10 = df.sort_values("2025", ascending=False).head(10)

    fig_top = px.bar(
        top10,
        x="Department",
        y="2025",
        title="Top 10 Departments by Budget (2025)",
        text="2025",
        color="2025"
    )
    st.plotly_chart(fig_top, use_container_width=True)

    # ================= CORRELATION HEATMAP ==================
    st.subheader("🔗 Correlation Heatmap (Budget Relationship between Departments)")

    corr = df[year_cols].T.corr()

    fig_corr = px.imshow(
        corr,
        color_continuous_scale="Viridis",
        title="Budget Correlation Heatmap"
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    # ================= INSIGHTS SECTION ==================
    st.subheader("🧠 Automated Insights")

    highest_2025 = df.loc[df["2025"].idxmax()]
    lowest_2025 = df.loc[df["2025"].idxmin()]

    st.markdown(f"""
### 📌 Key Insights
- **Highest allocation (2025)**: `{highest_2025['Department']}` — **₹{highest_2025['2025']:.2f} Cr**
- **Lowest allocation (2025)**: `{lowest_2025['Department']}` — **₹{lowest_2025['2025']:.2f} Cr**
- **Strongest growth year**: `{yoy.idxmax()}` — {yoy.max():.2f}%
- **Weakest growth year**: `{yoy.idxmin()}` — {yoy.min():.2f}%
""")

else:
    st.info("Please upload your **Budget 14-25 CSV file** to begin analysis.")
