# app.py
"""
Streamlit Data Analysis App
- Loads a CSV from user upload (or falls back to a default local path)
- Performs cleaning (simple, documented strategy)
- Shows summary stats, charts, insights, and provides a downloadable cleaned CSV
- Uses pandas, plotly, seaborn, matplotlib
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy import stats

# ---------------------------
# Helper functions
# ---------------------------

def load_data(uploaded_file, fallback_path=None):
    """
    Load CSV from uploaded_file (Streamlit's UploadedFile) or fallback_path.
    Returns: DataFrame
    """
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("Loaded uploaded CSV.")
            return df
        except Exception as e:
            st.error(f"Could not read uploaded file: {e}")
            return None
    else:
        if fallback_path:
            try:
                df = pd.read_csv(fallback_path)
                st.info(f"Loaded dataset from fallback path: {fallback_path}")
                return df
            except Exception as e:
                st.error(f"Could not read fallback file: {e}")
                return None
        else:
            st.warning("No file uploaded and no fallback path provided.")
            return None

def detect_columns(df):
    """
    Detect numerical and categorical columns.
    Returns: (numerical_cols, categorical_cols)
    """
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    return numerical_cols, categorical_cols

def clean_dataframe(df):
    """
    Simple documented cleaning:
    - Numerical columns: fill missing values with median
    - Categorical columns: fill missing with mode if exists else 'Missing'
    - Strip whitespace from column names
    Returns cleaned DataFrame
    """
    df = df.copy()
    # Trim column names
    df.columns = [c.strip() for c in df.columns]
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    for col in num_cols:
        median = df[col].median()
        df[col] = df[col].fillna(median)

    for col in cat_cols:
        try:
            mode = df[col].mode(dropna=True)[0]
        except Exception:
            mode = 'Missing'
        df[col] = df[col].fillna(mode)

    return df

def detect_anomalies(df, numerical_cols, z_thresh=3.0):
    """
    Detect anomaly rows using z-score on numerical columns.
    Returns DataFrame of anomalies (may be empty).
    """
    if not numerical_cols:
        return pd.DataFrame()
    z = np.abs(stats.zscore(df[numerical_cols], nan_policy='omit'))
    z_df = pd.DataFrame(z, columns=numerical_cols, index=df.index)
    anomaly_idx = z_df[(z_df > z_thresh).any(axis=1)].index
    return df.loc[anomaly_idx]

def corr_heatmap_figure(corr):
    """
    Create a matplotlib figure for the correlation heatmap (used for st.pyplot).
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, square=True,
                cbar_kws={"shrink": 0.8})
    plt.tight_layout()
    return fig

def make_download_button(df, filename="cleaned_data.csv", button_text="Download cleaned CSV"):
    """
    Create a bytes buffer and return for st.download_button
    """
    csv = df.to_csv(index=False).encode('utf-8')
    return st.download_button(label=button_text, data=csv, file_name=filename, mime='text/csv')

# ---------------------------
# Streamlit page config
# ---------------------------

st.set_page_config(page_title="Data Analysis App", layout="wide", initial_sidebar_state="expanded")
st.title("📊 Streamlit Data Analysis App")
st.markdown("Upload a CSV or use the provided dataset. The app detects numeric/categorical columns, cleans data, "
            "shows statistics, charts, correlations, and provides a cleaned CSV for download.")

# ---------------------------
# Sidebar - file upload & options
# ---------------------------

st.sidebar.header("Upload / Options")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

# Fallback path - helpful for local testing or GitHub deployment where dataset is included
FALLBACK_PATH = "/mnt/data/India_GDP_Data.csv"  # change/remove if not applicable

df_raw = load_data(uploaded_file, fallback_path=FALLBACK_PATH)

if df_raw is None:
    st.stop()

# ---------------------------
# Data cleaning & basic info
# ---------------------------

st.header("Data preview and cleaning")
with st.expander("Raw data (first 10 rows)"):
    st.write(df_raw.head(10))

# Clean dataframe (strip column whitespace, fill missing, documented)
df_clean = clean_dataframe(df_raw)

st.subheader("Detected columns and types")
numerical_cols, categorical_cols = detect_columns(df_clean)
col_info = pd.DataFrame({
    "column": df_clean.columns,
    "dtype": df_clean.dtypes.astype(str).values
})
st.table(col_info)

st.write(f"**Numerical columns detected:** {numerical_cols}")
st.write(f"**Categorical columns detected:** {categorical_cols} (empty list means none detected)")

# Show missing value summary
st.subheader("Missing values (counts and %)")
miss_count = df_clean.isnull().sum()
miss_pct = (df_clean.isnull().mean() * 100).round(2)
miss_df = pd.DataFrame({"missing_count": miss_count, "missing_percent": miss_pct})
st.dataframe(miss_df.style.format({"missing_percent": "{:.2f}%"}), height=200)

# Show summary statistics
if numerical_cols:
    st.subheader("Numerical summary (describe)")
    st.dataframe(df_clean[numerical_cols].describe().T)
else:
    st.info("No numerical columns detected to show summary statistics.")

if categorical_cols:
    st.subheader("Categorical summary")
    st.dataframe(df_clean[categorical_cols].describe().T)

# Download cleaned CSV
st.markdown("---")
st.subheader("Download cleaned CSV")
make_download_button(df_clean, filename="India_GDP_Data_cleaned.csv")

# ---------------------------
# Sidebar - dynamic filters
# ---------------------------

st.sidebar.header("Filters")
# If Year exists and numeric, provide a year range filter
if 'Year' in df_clean.columns and pd.api.types.is_numeric_dtype(df_clean['Year']):
    years = df_clean['Year'].sort_values().unique()
    year_min = int(df_clean['Year'].min())
    year_max = int(df_clean['Year'].max())
    year_range = st.sidebar.slider("Year range", min_value=year_min, max_value=year_max,
                                   value=(year_min, year_max), step=1)
    df_filtered = df_clean[(df_clean['Year'] >= year_range[0]) & (df_clean['Year'] <= year_range[1])]
else:
    df_filtered = df_clean.copy()

# For other numerical columns, allow min/max sliders dynamically (optional)
for col in numerical_cols:
    if col == 'Year':
        continue
    col_min = float(df_clean[col].min())
    col_max = float(df_clean[col].max())
    # Display with the original precision scaled
    v = st.sidebar.slider(f"{col} range", min_value=col_min, max_value=col_max,
                          value=(col_min, col_max))
    df_filtered = df_filtered[(df_filtered[col] >= v[0]) & (df_filtered[col] <= v[1])]

# For categorical columns, allow multi-select filter
for col in categorical_cols:
    vals = df_clean[col].unique().tolist()
    sel = st.sidebar.multiselect(f"Filter {col}", options=vals, default=vals)
    df_filtered = df_filtered[df_filtered[col].isin(sel)]

st.sidebar.markdown("---")
st.sidebar.write("Rows after filtering: ", len(df_filtered))

# ---------------------------
# Insights and anomalies
# ---------------------------

st.header("Computed insights & anomaly detection")
st.subheader("Automatically computed facts")
st.write(f"- Dataset shape (rows × cols): {df_clean.shape}")
st.write(f"- Numerical columns: {numerical_cols}")
st.write(f"- Categorical columns: {categorical_cols}")
st.write("- Missing values: see table above.")

# Correlations
if numerical_cols:
    corr = df_filtered[numerical_cols].corr()
    st.write("Top absolute correlations (numerical columns):")
    # show top 5 absolute correlations (excluding self)
    corr_unstack = corr.abs().unstack().reset_index()
    corr_unstack.columns = ['var1', 'var2', 'abs_corr']
    corr_unstack = corr_unstack[corr_unstack['var1'] != corr_unstack['var2']]
    corr_unstack = corr_unstack.sort_values('abs_corr', ascending=False)
    # deduplicate pairs (a,b) and (b,a)
    seen = set()
    unique_pairs = []
    for _, r in corr_unstack.iterrows():
        pair = tuple(sorted([r['var1'], r['var2']]))
        if pair not in seen:
            seen.add(pair)
            unique_pairs.append((r['var1'], r['var2'], r['abs_corr']))
        if len(unique_pairs) >= 5:
            break
    for a, b, val in unique_pairs:
        st.write(f"- {a} ↔ {b}: {val:.3f}")

# Anomaly detection: z-score > 3 on numerical columns
anomalies = detect_anomalies(df_filtered, numerical_cols, z_thresh=3.0)
st.write(f"Number of anomalous rows detected (z-score > 3 on any numeric column): **{len(anomalies)}**")
if not anomalies.empty:
    st.dataframe(anomalies)
    st.warning("Investigate these rows — they are extreme relative to the dataset (z-score > 3).")

# ---------------------------
# Charts
# ---------------------------

st.header("Charts & Visualizations")
st.markdown("Below each chart you will find a short explanation automatically generated from the data shown.")

# 1) Line chart for trends (Year vs numeric metrics)
st.subheader("Line chart — Trends over Years")
if 'Year' in df_filtered.columns and pd.api.types.is_numeric_dtype(df_filtered['Year']):
    # let user pick which numeric columns to plot
    st.markdown("Select numeric columns to plot as trends over Year:")
    trend_cols = st.multiselect("Trend columns", options=numerical_cols, default=[c for c in numerical_cols if c != 'Year'])
    if not trend_cols:
        st.info("Select at least one numeric column (other than Year) to show trends.")
    else:
        # Use plotly for interactive line chart
        df_plot = df_filtered[['Year'] + trend_cols].sort_values('Year')
        fig = px.line(df_plot, x='Year', y=trend_cols, markers=True,
                      title="Trends over Year")
        fig.update_layout(legend_title_text='Metric')
        st.plotly_chart(fig, use_container_width=True)
        # explanation
        st.markdown("**Explanation:** This line chart shows how the selected numeric metrics evolve over years. "
                    "If a selected metric shows a steady upward trend, it indicates sustained increase across the time range.")
else:
    st.info("No numeric 'Year' column available for a time-based line chart.")

st.markdown("---")

# 2) Bar chart for categorical comparisons
st.subheader("Bar chart — Categorical comparisons or top-N by numeric")
if categorical_cols:
    st.markdown("Bar chart compares selected categorical column's aggregated numeric value.")
    cat_col = st.selectbox("Choose categorical column", options=categorical_cols)
    agg_col = st.selectbox("Choose numeric column to aggregate", options=numerical_cols)
    agg_func = st.selectbox("Aggregation", options=['sum', 'mean', 'median', 'count'], index=1)
    if agg_func == 'count':
        bar_df = df_filtered.groupby(cat_col).size().reset_index(name='count').sort_values('count', ascending=False)
        fig2 = px.bar(bar_df, x=cat_col, y='count', title=f"Count by {cat_col}")
    else:
        grp = df_filtered.groupby(cat_col)[agg_col]
        if agg_func == 'sum':
            bar_df = grp.sum().reset_index().sort_values(agg_col, ascending=False)
        elif agg_func == 'mean':
            bar_df = grp.mean().reset_index().sort_values(agg_col, ascending=False)
        else:
            bar_df = grp.median().reset_index().sort_values(agg_col, ascending=False)
        fig2 = px.bar(bar_df, x=cat_col, y=agg_col, title=f"{agg_func.title()} of {agg_col} by {cat_col}")
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown("**Explanation:** This bar chart compares categories by the chosen aggregate metric.")
else:
    # No categorical columns - fallback: show top N years by GDP (or top N by selected numeric)
    st.markdown("No categorical columns detected. Showing top-N rows by a numeric metric instead (useful alternative).")
    metric_for_bar = st.selectbox("Pick numeric metric for top-N bar chart", options=[c for c in numerical_cols if c != 'Year'] or numerical_cols)
    top_n = st.slider("Top N", min_value=3, max_value=min(20, len(df_filtered)), value=10)
    bar_df = df_filtered.sort_values(metric_for_bar, ascending=False).head(top_n)
    fig2 = px.bar(bar_df, x='Year' if 'Year' in bar_df.columns else bar_df.index.astype(str),
                  y=metric_for_bar, title=f"Top {top_n} Years by {metric_for_bar}")
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown("**Explanation:** Since the dataset has no categorical variables, we present a top-N bar chart of years "
                f"ranked by `{metric_for_bar}` to allow easy compariso
