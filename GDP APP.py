# app.py
import streamlit as st
import pandas as pd
import numpy as np
import io
from utils import (
    safe_read_csv,
    dataframe_overview,
    detect_variable_types,
    make_summary_table,
    plot_histogram,
    plot_box,
    plot_scatter,
    plot_correlation_heatmap,
    download_link_for_df,
    fill_missing_preview
)
import plotly.express as px

st.set_page_config(page_title="CSV Analyzer — Clean UX", layout="wide", initial_sidebar_state="expanded")

# ---- SIDEBAR ----
st.sidebar.title("CSV Analyzer")
st.sidebar.markdown("Upload a CSV and explore it with interactive visualizations and simple cleaning tools.")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv", "txt"], accept_multiple_files=False)

# quick sample dataset option
use_sample = st.sidebar.checkbox("Load sample dataset (Iris)", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown("View & download cleaned data after applying transformations.")
st.sidebar.markdown("Built for: fast EDA, basic cleaning, interactive plots, and export-ready CSVs.")

# ---- MAIN ----
st.title("CSV Analyzer — Interactive EDA & Clean UX")
st.caption("Upload any CSV. The app automatically detects data types and offers plots & simple cleaning options.")

# Load data
if use_sample and uploaded_file is None:
    df = px.data.iris()  # small sample from plotly
    st.info("Loaded sample Iris dataset.")
elif uploaded_file is not None:
    try:
        df = safe_read_csv(uploaded_file)
        st.success(f"Loaded `{uploaded_file.name}` — {df.shape[0]} rows x {df.shape[1]} cols")
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        st.stop()
else:
    st.info("Upload a CSV or check 'Load sample dataset' to try the app.")
    st.stop()

# Basic controls for preview
st.sidebar.markdown("### Preview options")
n_preview = st.sidebar.slider("Rows to preview", min_value=5, max_value=100, value=10, step=5)
show_head = st.sidebar.checkbox("Show head (vs tail)", value=True)

# Overview & types
with st.expander("Dataset overview", expanded=True):
    st.subheader("Overview")
    overview = dataframe_overview(df)
    st.metric("Rows", overview["rows"])
    st.metric("Columns", overview["cols"])
    st.write("Memory usage:", overview["memory"])
    st.write("Column types and missing values:")
    types_table = detect_variable_types(df)
    st.dataframe(types_table, height=300)

# Preview table
with st.expander("Data preview", expanded=False):
    st.subheader("Preview")
    if show_head:
        st.dataframe(df.head(n_preview))
    else:
        st.dataframe(df.tail(n_preview))

# Summary statistics
with st.expander("Summary statistics", expanded=True):
    st.subheader("Summary (numeric & categorical)")
    summary = make_summary_table(df)
    st.dataframe(summary, height=350)
    st.download_button("Download summary (CSV)", data=summary.to_csv(index=False).encode('utf-8'),
                       file_name="summary_statistics.csv", mime="text/csv")

# Missing values & simple cleaning UI
with st.expander("Missing values & simple cleaning", expanded=True):
    st.subheader("Missing values")
    mv = df.isna().sum().sort_values(ascending=False)
    mv = mv[mv > 0]
    if mv.empty:
        st.success("No missing values detected.")
    else:
        st.write("Columns with missing values:")
        st.table(mv)

        # Cleaning controls
        col_to_fill = st.selectbox("Choose a column to preview fill", options=list(mv.index), index=0)
        fill_strategy = st.selectbox("Fill strategy", options=["Drop rows", "Fill with mean (numeric)", "Fill with median (numeric)", "Fill with mode", "Fill with custom value"])
        custom_value = None
        if fill_strategy == "Fill with custom value":
            custom_value = st.text_input("Custom value (will be interpreted with column dtype)")

        if st.button("Preview filling (does not modify original)"):
            preview = fill_missing_preview(df, col_to_fill, fill_strategy, custom_value)
            st.write("Preview of first 10 rows where missing occurred (filled values shown):")
            st.dataframe(preview.head(10))

        if st.button("Apply fill to dataframe"):
            df = fill_missing_preview(df, col_to_fill, fill_strategy, custom_value, apply=True)
            st.success(f"Applied `{fill_strategy}` to `{col_to_fill}`. New missing count: {df[col_to_fill].isna().sum()}")

# Interactive plotting controls
with st.expander("Interactive plots", expanded=True):
    st.subheader("Plots")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    all_cols = df.columns.tolist()

    st.write("Choose plot type and variables. Plots are interactive (zoom, pan).")

    plot_type = st.selectbox("Plot type", options=["Histogram", "Box", "Scatter", "Correlation heatmap"])

    if plot_type == "Histogram":
        if not numeric_cols:
            st.warning("No numeric columns available for histogram.")
        else:
            col = st.selectbox("Numeric column", numeric_cols)
            bins = st.slider("Bins", 5, 200, 30)
            fig = plot_histogram(df, col, bins=bins)
            st.plotly_chart(fig, use_container_width=True)

    elif plot_type == "Box":
        if not numeric_cols:
            st.warning("No numeric columns available for box plot.")
        else:
            col = st.selectbox("Numeric column", numeric_cols)
            fig = plot_box(df, col)
            st.plotly_chart(fig, use_container_width=True)

    elif plot_type == "Scatter":
        if len(numeric_cols) < 2:
            st.warning("Need at least two numeric columns for scatter.")
        else:
            x = st.selectbox("X-axis", numeric_cols, index=0)
            y = st.selectbox("Y-axis", numeric_cols, index=1)
            color = st.selectbox("Optional color (categorical)", options=[None] + [c for c in all_cols if df[c].nunique() < 20], index=0)
            fig = plot_scatter(df, x, y, color)
            st.plotly_chart(fig, use_container_width=True)

    else:  # Correlation heatmap
        if len(numeric_cols) < 2:
            st.warning("Not enough numeric columns for correlation heatmap.")
        else:
            corr_fig = plot_correlation_heatmap(df[numeric_cols])
            st.plotly_chart(corr_fig, use_container_width=True)

# Column inspector and transformation
with st.expander("Column inspector & transformations", expanded=False):
    st.subheader("Column inspector")
    col = st.selectbox("Select column", options=all_cols)
    st.write("First 10 values:")
    st.write(df[col].head(10))
    st.write("Unique values (up to 100):")
    st.write(df[col].dropna().unique()[:100])

    if st.button("Convert to numeric (coerce errors)"):
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            st.success(f"Column `{col}` converted to numeric.")
        except Exception as e:
            st.error(f"Conversion failed: {e}")

# Download cleaned CSV
with st.expander("Export / Download", expanded=True):
    st.subheader("Export cleaned CSV")
    csv_bytes = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", data=csv_bytes, file_name="cleaned_data.csv", mime="text/csv")
    st.markdown("---")
    st.markdown("If you'd like a GitHub-ready folder with this app, include these files in your repo:\n\n- app.py\n- utils.py\n- requirements.txt\n- README.md")

# Footer / help
st.markdown("---")
st.caption("Hints: use the sidebar to set preview size. Use 'Missing values' to quickly impute simple fixes. For more advanced modeling, export the cleaned CSV and use a notebook or pipeline.")

# utils.py
import pandas as pd
import numpy as np
import io
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

def safe_read_csv(uploaded_file):
    """
    Read a CSV uploaded via Streamlit file_uploader.
    Tries multiple encodings and separators for robustness.
    """
    # uploaded_file is a BytesIO or similar
    content = uploaded_file.read()
    # try utf-8 first
    try:
        df = pd.read_csv(io.BytesIO(content))
        return df
    except Exception:
        # try common alternatives
        tries = [
            {"encoding":"latin1"},
            {"sep":";"},
            {"sep":"\t"},
            {"encoding":"latin1","sep":";"},
            {"encoding":"utf-8","engine":"python"}
        ]
        for params in tries:
            try:
                df = pd.read_csv(io.BytesIO(content), **params)
                return df
            except Exception:
                continue
    # last resort: let pandas infer with python engine
    return pd.read_csv(io.BytesIO(content), engine="python")

def dataframe_overview(df):
    rows, cols = df.shape
    mem = f"{df.memory_usage(deep=True).sum()/1024**2:.2f} MB"
    return {"rows": rows, "cols": cols, "memory": mem}

def detect_variable_types(df):
    """
    Returns a DataFrame describing each column type and missing counts, unique counts.
    """
    cols = []
    for c in df.columns:
        dtype = str(df[c].dtype)
        n_missing = int(df[c].isna().sum())
        n_unique = int(df[c].nunique(dropna=True))
        sample = df[c].dropna().unique()[:5].tolist()
        cols.append({"column":c, "dtype":dtype, "missing":n_missing, "unique":n_unique, "sample_values": sample})
    return pd.DataFrame(cols).sort_values(by="missing", ascending=False)

def make_summary_table(df):
    """
    Combines numeric describe and basic categorical counts into a single table for display.
    """
    numeric = df.select_dtypes(include=[np.number])
    cat = df.select_dtypes(exclude=[np.number])

    num_desc = numeric.describe().T.reset_index().rename(columns={"index":"column"})
    cat_summary = []
    for c in cat.columns:
        top = cat[c].mode(dropna=True)
        top_val = top.iloc[0] if not top.empty else None
        cat_summary.append({"column":c, "top": top_val, "unique": int(cat[c].nunique(dropna=True)), "missing": int(cat[c].isna().sum())})
    cat_summary = pd.DataFrame(cat_summary)

    # unify
    if not num_desc.empty:
        num_desc = num_desc[["column","count","mean","std","min","25%","50%","75%","max"]]
    summary = num_desc
    if not cat_summary.empty:
        # append categorical info as new rows (fill numeric cols with NaN)
        for col in ["count","mean","std","min","25%","50%","75%","max"]:
            if col not in cat_summary.columns:
                cat_summary[col] = np.nan
        summary = pd.concat([num_desc, cat_summary.rename(columns={"top":"50%"})], sort=False, ignore_index=True).fillna("")
    return summary

# Plotting helpers using plotly for interactivity
def plot_histogram(df, col, bins=30):
    fig = px.histogram(df, x=col, nbins=bins, marginal="rug", title=f"Histogram of {col}")
    fig.update_layout(height=400)
    return fig

def plot_box(df, col):
    fig = px.box(df, y=col, points="all", title=f"Box plot of {col}")
    fig.update_layout(height=350)
    return fig

def plot_scatter(df, x, y, color=None):
    if color:
        fig = px.scatter(df, x=x, y=y, color=color, title=f"Scatter: {y} vs {x} colored by {color}")
    else:
        fig = px.scatter(df, x=x, y=y, title=f"Scatter: {y} vs {x}")
    fig.update_layout(height=450)
    return fig

def plot_correlation_heatmap(df_numeric):
    corr = df_numeric.corr()
    fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, colorbar=dict(title="corr")))
    fig.update_layout(title="Correlation matrix", height=500)
    return fig

def download_link_for_df(df, filename="data.csv"):
    return df.to_csv(index=False).encode('utf-8')

def fill_missing_preview(df, column, strategy, custom_value=None, apply=False):
    """
    Returns a df copy with missing values in `column` filled with the chosen strategy.
    If apply=True, modifies the original df (returns the modified df).
    """
    df_copy = df.copy()
    if strategy == "Drop rows":
        df_res = df_copy.dropna(subset=[column])
    elif strategy == "Fill with mean (numeric)":
        mean_val = df_copy[column].mean()
        df_res = df_copy.copy()
        df_res[column] = df_res[column].fillna(mean_val)
    elif strategy == "Fill with median (numeric)":
        med = df_copy[column].median()
        df_res = df_copy.copy()
        df_res[column] = df_res[column].fillna(med)
    elif strategy == "Fill with mode":
        mode = df_copy[column].mode(dropna=True)
        mode_val = mode.iloc[0] if not mode.empty else None
        df_res = df_copy.copy()
        df_res[column] = df_res[column].fillna(mode_val)
    elif strategy == "Fill with custom value":
        # try to cast custom_value to the column dtype if possible
        df_res = df_copy.copy()
        if custom_value is not None and custom_value != "":
            try:
                casted = df_res[column].astype(type(df_res[column].dropna().iloc[0]))
                df_res[column] = df_res[column].fillna(custom_value)
            except Exception:
                # fallback to raw string fill
                df_res[column] = df_res[column].fillna(custom_value)
        else:
            raise ValueError("Custom value cannot be empty.")
    else:
        raise ValueError("Unknown strategy")

    if apply:
        return df_res  # caller will replace original df
    else:
        return df_res

