# streamlit_app.py
import streamlit as st
import pandas as pd
import plotly.express as px

# Page settings
st.set_page_config(page_title="OSBM Survey Explorer", layout="wide")
st.title("OSBM Employee Satisfaction — FY ’25–26")

st.markdown(
    "Upload the Excel export (e.g., **OSBM Employee Satisfaction Survey FY '25-26(1-72).xlsx**) to explore."
)

# 1) Upload file
file = st.file_uploader("Upload Excel (.xlsx or .xls)", type=["xlsx", "xls"])

# Stop here until a file is uploaded
if not file:
    st.info("Choose the FY ’25–26 Excel file to begin.")
    st.stop()

# 2) Read the Excel into a DataFrame
# Requires 'openpyxl' (listed in requirements.txt)
df = pd.read_excel(file)
df.columns = [str(c).strip() for c in df.columns]  # tidy column names

# 3) Preview
st.subheader("Preview (first 10 rows)")
st.dataframe(df.head(10), use_container_width=True)

# 4) Choose a numeric column for a quick chart
st.subheader("Quick chart")
numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

if not numeric_cols:
    st.warning("No numeric columns detected yet—still fine for tables, but charts need numbers.")
else:
    col = st.selectbox("Pick a numeric column for a histogram", options=numeric_cols)
    fig = px.histogram(df, x=col, nbins=20, title=f"Histogram — {col}")
    st.plotly_chart(fig, use_container_width=True)

# (Optional) Save current view as CSV
st.download_button(
    "Download preview as CSV",
    df.head(10).to_csv(index=False).encode("utf-8"),
    file_name="preview_first_10_rows.csv",
    mime="text/csv",
)