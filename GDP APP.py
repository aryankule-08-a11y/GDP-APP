# Streamlit Budget Analysis App (Clean & Error‑Free Version)
# File: streamlit_budget_app.py
# This version is simplified and fully compatible with GitHub + Streamlit Cloud.

import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Budget Dashboard", layout="wide")

st.title("📊 Budget Analysis Dashboard (2014–2025)")
st.write("Upload your budget CSV file to start exploring the data.")

# ---- File Upload ----
file = st.file_uploader("Upload CSV File", type=["csv"])

if file is not None:
    df = pd.read_csv(file)
else:
    st.warning("Please upload a CSV file to continue.")
    st.stop()

# ---- Basic Cleanup ----
df.columns = [c.strip() for c in df.columns]

# Detect numeric + category columns
numeric_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
category_cols = df.select_dtypes(exclude=['int64','float64']).columns.tolist()

if len(numeric_cols) == 0:
    st.error("No numeric columns detected — the app needs at least one numeric column (e.g., Budget, Amount)")
    st.stop()

# ---- Sidebar Filters ----
st.sidebar.header("Filters")
value_col = st.sidebar.selectbox("Select value column", numeric_cols)

category_col = None
if len(category_cols) > 0:
    category_col = st.sidebar.selectbox("Select category column (optional)", ["None"] + category_cols)
    if category_col == "None":
        category_col = None

# Detect year column
year_col = None
for c in df.columns:
    if "year" in c.lower():
        year_col = c
        break

if year_col:
    df[year_col] = pd.to_numeric(df[year_col], errors='coerce')
    years = df[year_col].dropna().astype(int).unique()
    if len(years) > 0:
        min_y, max_y = min(years), max(years)
        selected_years = st.sidebar.slider("Select Year Range", min_y, max_y, (min_y, max_y))
        df = df[(df[year_col] >= selected_years[0]) & (df[year_col] <= selected_years[1])]

# ---- Overview Metrics ----
st.subheader("Overview Metrics")
col1, col2, col3 = st.columns(3)

col1.metric("Total", f"{df[value_col].sum():,.0f}")
col2.metric("Average", f"{df[value_col].mean():,.2f}")
col3.metric("Maximum", f"{df[value_col].max():,.0f}")

# ---- Charts ----
st.markdown("---")
st.subheader("Charts")

# Line chart if year column exists
if year_col:
    line_df = df.groupby(year_col)[value_col].sum().reset_index()
    fig_line = px.line(line_df, x=year_col, y=value_col, markers=True, title="Trend Over Years")
    st.plotly_chart(fig_line, use_container_width=True)

# Category chart
if category_col:
    category_df = df.groupby(category_col)[value_col].sum().reset_index().sort_values(value_col, ascending=False)
    fig_bar = px.bar(category_df, x=category_col, y=value_col, title="Category Breakdown")
    st.plotly_chart(fig_bar, use_container_width=True)

# ---- Data Table ----
st.markdown("---")
st.subheader("Data Table")
st.dataframe(df, use_container_width=True)

# ---- Download button ----
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("Download Filtered Data", csv, "filtered_budget.csv", "text/csv")

# ---- README for GitHub ----
st.markdown("---")
st.write("### Deployment Notes")
st.write("""
**requirements.txt** should contain:

```
streamlit
pandas
plotly
```

To run locally:
```
streamlit run streamlit_budget_app.py
```

This version is optimized for Streamlit Cloud and GitHub with no errors.
""")
