# app/streamlit_app.py  (or streamlit_app.py at repo root)
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="OSBM Survey Explorer", layout="wide")
st.title("OSBM Employee Satisfaction — FY ’25–26")

st.markdown(
    "Upload the Excel export (e.g., **OSBM Employee Satisfaction Survey FY '25-26(1-72).xlsx**) "
    "or place it locally at `data/…xlsx` to auto-load."
)

# --- Local default (dev convenience) ---
default_path = Path("data/OSBM Employee Satisfaction Survey FY '25-26(1-72).xlsx")

df = None
if default_path.exists():
    # Local development: auto-load from data/
    df = pd.read_excel(default_path)
    st.info(f"Loaded default dataset from `{default_path}`.")
else:
    # Deployed (or no local file): use uploader
    file = st.file_uploader("Upload Excel (.xlsx or .xls)", type=["xlsx", "xls"])
    if not file:
        st.info("Choose the FY ’25–26 Excel file to begin.")
        st.stop()
    df = pd.read_excel(file)

# Tidy columns
df.columns = [str(c).strip() for c in df.columns]

# Preview
st.subheader("Preview (first 10 rows)")
st.dataframe(df.head(10), use_container_width=True)

# Quick chart
st.subheader("Quick chart")
numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
if not numeric_cols:
    st.warning("No numeric columns detected yet—tables will still work.")
else:
    col = st.selectbox("Pick a numeric column for a histogram", options=numeric_cols)
    fig = px.histogram(df, x=col, nbins=20, title=f"Histogram — {col}")
    st.plotly_chart(fig, use_container_width=True)

# Optional: download the preview
st.download_button(
    "Download preview as CSV",
    df.head(10).to_csv(index=False).encode("utf-8"),
    file_name="preview_first_10_rows.csv",
    mime="text/csv",
)