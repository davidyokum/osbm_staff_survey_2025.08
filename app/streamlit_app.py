# app/streamlit_app.py  (run from repo root with:  streamlit run app/streamlit_app.py)
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="OSBM Survey Explorer", layout="wide")
st.title("OSBM Employee Satisfaction — FY ’25–26")

st.markdown(
    "This app auto-loads your local file if present at "
    "`data/OSBM Employee Satisfaction Survey FY '25-26(1-72).xlsx`, "
    "otherwise it will prompt you to upload."
)

# ---------- Load data (simple, with your exact filename) ----------
LOCAL_FILE = Path("data/OSBM Employee Satisfaction Survey FY '25-26(1-72).xlsx")

if LOCAL_FILE.exists():
    st.info(f"Loaded local dataset from `{LOCAL_FILE}`")
    df = pd.read_excel(LOCAL_FILE)
else:
    file = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])
    if not file:
        st.info("Place the file in `data/…` for auto-load, or upload it here to begin.")
        st.stop()
    df = pd.read_excel(file)

# Tidy columns
df.columns = [str(c).strip() for c in df.columns]

# ---------- Preview ----------
st.subheader("Preview (first 10 rows)")
st.dataframe(df.head(10), use_container_width=True)

# ---------- Quick chart: histogram of a numeric column ----------
st.subheader("Quick chart")
numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
if not numeric_cols:
    st.warning("No numeric columns detected yet—tables will still work.")
else:
    col = st.selectbox("Pick a numeric column for a histogram", options=numeric_cols)
    fig = px.histogram(df, x=col, nbins=20, title=f"Histogram — {col}")
    st.plotly_chart(fig, use_container_width=True)

# ---------- Optional: simple cross-tab for categorical columns ----------
st.subheader("Cross-tab (stacked bar)")
cat_cols = [c for c in df.columns if (df[c].nunique() <= 12 and not pd.api.types.is_numeric_dtype(df[c]))]
if len(cat_cols) >= 2:
    a = st.selectbox("Row variable", options=cat_cols, index=0, key="ct1")
    b = st.selectbox("Color variable", options=[c for c in cat_cols if c != a], index=0, key="ct2")
    cross = df.groupby([a, b]).size().reset_index(name="n")
    fig2 = px.bar(cross, x=a, y="n", color=b, barmode="stack", title=f"{a} × {b}")
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.caption("Tip: Cross-tab appears when the data has at least two small cardinality categorical columns.")

# ---------- Download preview ----------
st.download_button(
    "Download preview (first 10 rows) as CSV",
    df.head(10).to_csv(index=False).encode("utf-8"),
    file_name="preview_first_10_rows.csv",
    mime="text/csv",
)