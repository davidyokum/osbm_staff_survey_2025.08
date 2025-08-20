# app/streamlit_app.py
# Run from repo root:  streamlit run app/streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, List, Tuple

# -------------------- Global styling --------------------
st.markdown(
    """
    <style>
    /* Make subheaders (used for Tables) the same size as headers (used for Figures) */
    .stMarkdown h3 {
        font-size: 1.5rem !important;
    }
    .stMarkdown h2 {
        font-size: 1.5rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------- Page setup --------------------
st.set_page_config(page_title="OSBM Survey Report", layout="wide")
st.title("OSBM Employee Satisfaction — 2024-2025")

# -------------------- Paths & helpers --------------------
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent if APP_DIR.name == "app" else APP_DIR
DATA_DIR = PROJECT_ROOT / "data"

@st.cache_data
def read_excel_path(p: Path) -> pd.DataFrame:
    df = pd.read_excel(p)
    df.columns = [str(c).strip() for c in df.columns]
    return df

@st.cache_data
def read_excel_bytes(b: bytes) -> pd.DataFrame:
    df = pd.read_excel(BytesIO(b))
    df.columns = [str(c).strip() for c in df.columns]
    return df

def find_q1_column(d: pd.DataFrame) -> Optional[str]:
    """Return the column name for Q1 (happiness 0–10), robust to odd headers like 'Column1'."""
    cols = [str(c).strip() for c in d.columns]
    low  = [c.lower() for c in cols]
    # header-based matches first
    for i, c in enumerate(low):
        if ("happy" in c or "happiness" in c) and ("0-10" in c or "0–10" in c or "work" in c):
            return cols[i]
    for i, c in enumerate(low):
        if ("happy" in c or "happiness" in c):
            return cols[i]
    # fallback: detect a numeric 0–10 scale column
    candidates = []
    for name in cols:
        s = pd.to_numeric(d[name], errors="coerce")
        valid = s.dropna()
        if valid.empty:
            continue
        if valid.between(0, 10).mean() > 0.9 and valid.nunique() <= 11:
            candidates.append((name, valid.count()))
    if candidates:
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]
    return None

# -------------------- Load CURRENT YEAR (explicit) --------------------
current_path = DATA_DIR / "osbm_staff_survey_2025.xlsx"
if current_path.exists():
    df = read_excel_path(current_path)
    st.caption(f"Loaded current year data from `{current_path.name}`")
else:
    up = st.file_uploader("Upload CURRENT YEAR Excel (.xlsx)", type=["xlsx"])
    if not up:
        st.info("Place `data/osbm_staff_survey_2025.xlsx` or upload it to begin.")
        st.stop()
    df = read_excel_bytes(up.getbuffer().tobytes())

# -------------------- Intro paragraph --------------------
N = 110  # invited (adjust if needed)
Z = len(df)
response_rate = (Z / N) * 100 if N else np.nan

st.markdown(
    f"""
OSBM's annual employee survey was administered **July 2nd–15th**, 2025, to all OSBM employees (N = {N}).  
**{Z:,} people** completed the survey (a **{response_rate:.0f}% response rate**).
"""
)


# Optional quick preview
with st.expander("Preview first 10 rows", expanded=False):
    st.dataframe(df.head(10), use_container_width=True)

# ---------- Section helpers (Q32) ----------
def find_section_column(d: pd.DataFrame) -> Optional[str]:
    cols = [str(c).strip() for c in d.columns]
    low  = [c.lower() for c in cols]
    # Prefer explicit wording
    for i, c in enumerate(low):
        if "which best describes the section you work in" in c:
            return cols[i]
    # Fall back to anything with 'section'
    for i, c in enumerate(low):
        if "section" in c and ("work" in c or "describes" in c or "best" in c):
            return cols[i]
    for i, c in enumerate(low):
        if "section" in c:
            return cols[i]
    return None
def normalize_section(val: str) -> Optional[str]:
    """Map raw section labels to requested buckets; return None for missing/invalid."""
    if not isinstance(val, str):
        return None
    v = val.strip().lower()
    if "ncpro" in v:
        return "NCPRO"
    if "budget execution" in v:
        return "Budget Execution"
    if "budget development" in v:
        return "Budget Development"
    if "dea" in v or "internal audit" in v:
        return "DEA or Internal Audit"
    if "business office" in v or "grants" in v or "it" in v or "comms" in v or "communications" in v:
        return "Business Office/Grants/IT/Comms"
    if "intern" in v or "other" in v:
        return "Intern/Other"
    return "Intern/Other"
# Detect once and reuse
sec_col = find_section_column(df)


# ======================================================
# Table 1. Respondents by Section
# ======================================================
st.subheader("Table 1. Respondents by Section")

# Reuse find_section_column and normalize_section defined above
if sec_col:
    sec_series = df[sec_col].astype(str).map(normalize_section)

    # Ordered buckets
    bucket_order = [
        "NCPRO",
        "Budget Execution",
        "Budget Development",
        "DEA or Internal Audit",
        "Business Office/Grants/IT/Comms",
        "Intern/Other",
    ]

    counts = sec_series.value_counts(dropna=False).reindex(bucket_order, fill_value=0)
    total_resp = int(len(df))

    # Build clean table (no index column)
    tbl = pd.DataFrame({
        "Section": counts.index,
        "n": counts.values
    })
    tbl["%"] = (tbl["n"] / total_resp * 100).round(0).astype(int).astype(str) + "%"

    # Add totals row
    total_row = pd.DataFrame({"Section": ["Total"], "n": [tbl["n"].sum()], "%": ["100%"]})
    tbl = pd.concat([tbl, total_row], ignore_index=True)

    st.dataframe(tbl, use_container_width=True)
    st.caption(f"Out of **{total_resp}** respondents.")
else:
    st.info("Couldn't locate the Section (Q32) column in the dataset.")

# ======================================================
# Q1a — Happiness at Work (0–10) — 2025
# ======================================================
st.header("Figure 1. On a scale of 1 to 10, how happy are you at work?")

q1_candidates = [c for c in df.columns if "happy" in c.lower() or "happiness" in c.lower()]
if not q1_candidates:
    st.warning("Couldn’t find a column containing 'happy'/'happiness' for Q1.")
else:
    q1 = q1_candidates[0]
    s = pd.to_numeric(df[q1], errors="coerce").dropna()
    s = s[(s >= 0) & (s <= 10)]

    mean   = s.mean()
    median = s.median()
    top_box = (s >= 8).mean() * 100  # 8–10
    bot_box = (s <= 3).mean() * 100  # 0–3

    # KPI row (color-match the guide lines)
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Responses", f"{len(s):,}")
    k2.markdown(
        f"<div><small>Mean</small><br>"
        f"<span style='color:#f39c12;font-size:1.6rem;'>{mean:.1f}</span></div>",
        unsafe_allow_html=True,
    )
    k3.markdown(
        f"<div><small>Median</small><br>"
        f"<span style='color:#1f77b4;font-size:1.6rem;'>{median:.1f}</span></div>",
        unsafe_allow_html=True,
    )
    k4.metric("Top-box (8–10)", f"{top_box:.0f}%")
    k5.metric("Bottom-box (0–3)", f"{bot_box:.0f}%")

    # Histogram: one bin per integer; whole-number % labels
    fig = px.histogram(
        s,
        x=q1,
        nbins=11,
        range_x=[-0.5, 10.5],
        histnorm="percent",
        title="Distribution of Happiness (0 = Unhappy, 10 = Extremely Happy)",
    )
    fig.update_layout(
        xaxis=dict(tickmode="linear", dtick=1, range=[-0.5, 10.5], title="Happiness"),
        yaxis_title="% of responses",
    )
    fig.update_yaxes(range=[0,50], tick0=0, dtick=10)
    fig.update_traces(texttemplate="%{y:.0f}%", textposition="outside")
    fig.update_traces(hovertemplate="%{y:.0f}%% at %{x}<extra></extra>")

    # Guide lines (values shown via color-coded KPIs)
    fig.add_vline(x=mean,   line_width=2, line_dash="dash", line_color="#f39c12")
    fig.add_vline(x=median, line_width=2, line_dash="dot",  line_color="#1f77b4")

    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        f"""
The large majority of OSBM employees are happy at work: **{top_box:.0f}%** (roughly 2 in 3) rated their
happiness at 8 or higher on a 10-point scale in 2025 (see **Figure 1**). The mean was **{mean:.1f}** and the median **{median:.1f}**,
while only **{bot_box:.0f}%** reported very low scores (0–3).
"""
    )




# ======================================================
# Q1c — Happiness Distribution by Year 
# ======================================================
st.header("Figure 2. On a scale of 1 to 10, how happy are you at work?")

files_sm = {
    "2023": DATA_DIR / "osbm_staff_survey_2023.xlsx",
    "2024": DATA_DIR / "osbm_staff_survey_2024.xlsx",
    "2025": DATA_DIR / "osbm_staff_survey_2025.xlsx",
}
order = ["2023","2024","2025"]

series = {}
for y, path in files_sm.items():
    if not path.exists(): 
        continue
    d = read_excel_path(path)
    d.columns = [str(c).strip() for c in d.columns]
    col = find_q1_column(d)
    if not col: 
        continue
    s_y = pd.to_numeric(d[col], errors="coerce").dropna()
    s_y = s_y[(s_y >= 0) & (s_y <= 10)]
    if not s_y.empty:
        series[y] = s_y

if series:
    fig_sm = make_subplots(
        rows=1, cols=len(series),
        shared_yaxes=True, subplot_titles=[y for y in order if y in series]
    )
    col_idx = 0
    for y in order:
        if y not in series: 
            continue
        col_idx += 1
        fig_sm.add_trace(
            go.Histogram(
                x=series[y],
                histnorm="percent",
                xbins=dict(start=-0.5, end=10.5, size=1),
                marker_color="#1f77b4",
                hovertemplate="%{y:.0f}%% at %{x}<extra>" + y + "</extra>",
                name=y,
                showlegend=False,
            ),
            row=1, col=col_idx
        )
        m, med = series[y].mean(), series[y].median()
        fig_sm.add_vline(x=m,   line_width=2, line_dash="dash", line_color="#f39c12",
                         row=1, col=col_idx)
        fig_sm.add_vline(x=med, line_width=2, line_dash="dot",  line_color="#7f7f7f",
                         row=1, col=col_idx)

    fig_sm.update_layout(
        title="On a scale of 1 to 10, how happy are you at work?",
        bargap=0.0
    )
    fig_sm.update_xaxes(title_text="Happiness (0–10)", tickmode="linear", dtick=1, range=[-0.5,10.5])
    fig_sm.update_yaxes(title_text="% of responses", rangemode="tozero", range=[0,50], tick0=0, dtick=10)
    st.plotly_chart(fig_sm, use_container_width=True)
    if series:
        yr_stats = []
        for y in order:
            if y in series:
                s_y = series[y]
                yr_stats.append({
                    "year": y,
                    "mean": s_y.mean(),
                    "median": s_y.median(),
                    "top": (s_y >= 8).mean() * 100
                })

        means = [r["mean"] for r in yr_stats] if yr_stats else []
        mean_range = (max(means) - min(means)) if means else 0

        st.markdown(
            f"""
Employee happiness levels have been broadly **stable across 2023, 2024, and 2025** (see **Figure 2**).
Top-box scores have remained between {min(r['top'] for r in yr_stats):.0f}% and {max(r['top'] for r in yr_stats):.0f}%,
and means have varied only slightly (range ≈ {mean_range:.1f} points).
This indicates a consistent pattern of generally positive sentiment over time.
"""
        )

# ======================================================
# Table 2. 2025 Happiness by Section (summary table)
# ======================================================
st.subheader("Table 2. On a scale of 1 to 10, how happy are you at work? — by section (2025)")

if sec_col and q1_candidates:
    # Build a tidy frame with normalized sections and numeric happiness
    sec_series = df[sec_col].astype(str).map(normalize_section)
    df_section = pd.DataFrame({"Section": sec_series, "Happiness": df[q1_candidates[0]]})
    df_section["Happiness"] = pd.to_numeric(df_section["Happiness"], errors="coerce")
    df_section = df_section.dropna(subset=["Happiness"])

    # Aggregate: n, mean, median, top-box (>=8), bottom-box (<=3)
    summary = (
        df_section.groupby("Section")["Happiness"]
        .agg(
            n="count",
            mean="mean",
            median="median",
            top=lambda x: (x >= 8).mean() * 100,
            bottom=lambda x: (x <= 3).mean() * 100,
        )
        .reindex([
            "NCPRO",
            "Budget Execution",
            "Budget Development",
            "DEA or Internal Audit",
            "Business Office/Grants/IT/Comms",
            "Intern/Other",
        ], fill_value=0)
    )

    # Format values
    summary["mean"] = summary["mean"].round(1)
    summary["median"] = summary["median"].round(1)
    summary["top"] = summary["top"].round(0).astype(int).astype(str) + "%"
    summary["bottom"] = summary["bottom"].round(0).astype(int).astype(str) + "%"

    st.dataframe(summary, use_container_width=True)
    st.caption("Mean/median are raw scores (0–10). Top-box = % scoring 8–10; Bottom-box = % scoring 0–3.")
else:
    st.info("Happiness-by-section table unavailable (missing Q1 or Section column).")

# ======================================================
# Figure 3. Happiness at Work — Distribution by Section (2025)
# ======================================================
st.header("Figure 3. On a scale of 1 to 10, how happy are you at work? — distribution by section (2025)")

if sec_col and q1_candidates:
    # Build per-section numeric series (using normalized buckets)
    sec_series = df[sec_col].astype(str).map(normalize_section)
    df_sec = pd.DataFrame({"Section": sec_series, "Happiness": df[q1_candidates[0]]})
    df_sec["Happiness"] = pd.to_numeric(df_sec["Happiness"], errors="coerce")
    df_sec = df_sec.dropna(subset=["Happiness"])

    section_order = [
        "NCPRO",
        "Budget Execution",
        "Budget Development",
        "DEA or Internal Audit",
        "Business Office/Grants/IT/Comms",
        "Intern/Other",
    ]
    present_sections = [s for s in section_order if s in df_sec["Section"].unique()]
    if present_sections:
        # Layout: up to 2–3 per row; choose 2 by default for readability on screens
        cols = min(2, len(present_sections))
        rows = (len(present_sections) + cols - 1) // cols
        fig_sec = make_subplots(
            rows=rows, cols=cols, shared_yaxes=True,
            subplot_titles=present_sections,
            horizontal_spacing=0.12,  # give panels breathing room
            vertical_spacing=0.18
        )
        for idx, sect in enumerate(present_sections, start=1):
            s_subset = df_sec.loc[df_sec["Section"] == sect, "Happiness"]
            row_idx = (idx - 1) // cols + 1
            col_idx = (idx - 1) % cols + 1
            fig_sec.add_trace(
                go.Histogram(
                    x=s_subset,
                    histnorm="percent",
                    xbins=dict(start=-0.5, end=10.5, size=1),
                    marker_color="#1f77b4",
                    hovertemplate="%{y:.0f}%% at %{x}<extra>" + sect + "</extra>",
                    showlegend=False,
                    name=sect
                ),
                row=row_idx, col=col_idx
            )
            # Add mean/median lines per panel
            m, med = s_subset.mean(), s_subset.median()
            fig_sec.add_vline(x=m,   line_width=2, line_dash="dash", line_color="#f39c12",
                              row=row_idx, col=col_idx)
            fig_sec.add_vline(x=med, line_width=2, line_dash="dot",  line_color="#7f7f7f",
                              row=row_idx, col=col_idx)

        fig_sec.update_layout(
            title="Happiness Distribution by Section",
            bargap=0.05,
            height=max(360 * rows, 420),  # scale with number of rows
            margin=dict(l=40, r=20, t=60, b=40)
        )
        fig_sec.update_xaxes(title_text="Happiness (0–10)", tickmode="linear", dtick=1, range=[-0.5,10.5])
        fig_sec.update_yaxes(title_text="% of responses", rangemode="tozero", range=[0,50], tick0=0, dtick=10)
        st.plotly_chart(fig_sec, use_container_width=True)
    else:
        st.info("No recognized section categories present for plotting.")
else:
    st.info("Distribution-by-section figure unavailable (missing Q1 or Section column).")


# ======================================================
# Q2 — Would you refer someone to work here?
# ======================================================

def find_q2_column(d: pd.DataFrame) -> Optional[str]:
    """Return the column name for Q2 (referral intent), robust to odd headers."""
    cols = [str(c).strip() for c in d.columns]
    low  = [c.lower() for c in cols]
    # header-based clues
    keys = ["refer", "recommend", "would you refer", "would you recommend"]
    for i, c in enumerate(low):
        if any(k in c for k in keys):
            return cols[i]
    # fallback: any yes/no looking column near the top few columns
    for i, c in enumerate(low[:12]):
        if ("yes" in c) or ("no" in c):
            return cols[i]
    return None

def normalize_yesno(val) -> str:
    if isinstance(val, str):
        v = val.strip().lower()
        if v in ["yes","y","true","t","1"]:
            return "Yes"
        if v in ["no","n","false","f","0"]:
            return "No"
    if isinstance(val, (int, float)):
        if val == 1:
            return "Yes"
        if val == 0:
            return "No"
    return None  # treat everything else as missing

# ---------- Q2 current year (2025)
st.header("Figure 4. Would you refer someone to work here? (2025)")

q2_col = find_q2_column(df)
if not q2_col:
    st.info("Couldn’t locate Q2 (referral) in the dataset.")
else:
    q2_norm = df[q2_col].apply(normalize_yesno)
    valid = q2_norm.dropna()
    n_valid = len(valid)
    yes_pct = (valid == "Yes").mean() * 100 if n_valid else 0
    no_pct  = (valid == "No").mean() * 100 if n_valid else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Valid responses", f"{n_valid:,}")
    c2.metric("Would refer (Yes)", f"{yes_pct:.0f}%")
    c3.metric("Would not (No)", f"{no_pct:.0f}%")

    # Donut chart (Yes vs No), percentages
    pie_df = pd.DataFrame({"Response": ["Yes","No"], "Percent": [yes_pct, no_pct]})
    fig_q2 = px.pie(pie_df, names="Response", values="Percent", hole=0.5,
                    title="Would you refer someone to work here? (2025)", color="Response",
                    color_discrete_map={"Yes":"#1f77b4","No":"#7f7f7f"})
    fig_q2.update_traces(textposition="outside", texttemplate="%{label}: %{value:.0f}%")
    st.plotly_chart(fig_q2, use_container_width=True)

    st.markdown(
        f"""
**{yes_pct:.0f}%** of respondents say they would refer someone to work at OSBM in 2025 (see **Figure 4**).
"""
    )

#
# ---------- Q2 by year (trend line)
st.header("Figure 5. Would you refer someone to work here? (by year, 2023–2025)")

files_q2 = {
    "2023": DATA_DIR / "osbm_staff_survey_2023.xlsx",
    "2024": DATA_DIR / "osbm_staff_survey_2024.xlsx",
    "2025": DATA_DIR / "osbm_staff_survey_2025.xlsx",
}
order_years = ["2023","2024","2025"]
rows_trend = []
for y, path in files_q2.items():
    if not path.exists():
        continue
    d = read_excel_path(path)
    col = find_q2_column(d)
    if not col:
        continue
    v = d[col].apply(normalize_yesno).dropna()
    if v.empty:
        continue
    yes_y = (v == "Yes").mean() * 100
    rows_trend.append({"Year": y, "% Yes": round(yes_y, 0)})

if rows_trend:
    trend_q2 = pd.DataFrame(rows_trend)
    trend_q2["Year"] = pd.Categorical(trend_q2["Year"], categories=order_years, ordered=True)
    trend_q2 = trend_q2.sort_values("Year")
    fig_q2trend = go.Figure(go.Scatter(
        x=trend_q2["Year"], y=trend_q2["% Yes"],
        mode="lines+markers+text",
        line=dict(width=3, color="#1f77b4"),
        marker=dict(size=10),
        text=[f"{v:.0f}%" for v in trend_q2["% Yes"]],
        textposition="top center",
        hovertemplate="%{y:.0f}%% in %{x}<extra></extra>"
    ))
    fig_q2trend.update_layout(
        title="Referral (Yes) — Trend (2023–2025)",
        yaxis=dict(title="% Yes", range=[0,100], tick0=0, dtick=10),
        xaxis=dict(title="Survey Year", type="category"),
        margin=dict(l=40, r=20, t=60, b=40),
        height=420
    )
    st.plotly_chart(fig_q2trend, use_container_width=True)
else:
    st.info("No yearly referral data found to plot the trend.")


# ======================================================
# Table 4. Q2 — Referral Percentages by Year (2023–2025)
# ======================================================
st.subheader("Table 4. Would you refer someone to work here? — percentages by year (2023–2025)")

# Build year-wise series locally for this table
files_q2_tbl = {
    "2023": DATA_DIR / "osbm_staff_survey_2023.xlsx",
    "2024": DATA_DIR / "osbm_staff_survey_2024.xlsx",
    "2025": DATA_DIR / "osbm_staff_survey_2025.xlsx",
}
order_years = ["2023","2024","2025"]
rows = []
for y, path in files_q2_tbl.items():
    if not path.exists():
        continue
    d = read_excel_path(path)
    col = find_q2_column(d)
    if not col:
        continue
    v = d[col].apply(normalize_yesno).dropna()
    if v.empty:
        continue
    n_y = len(v)
    yes_y = (v == "Yes").mean() * 100
    no_y  = (v == "No").mean() * 100
    rows.append({"Year": y, "n": n_y, "% Yes": f"{yes_y:.0f}%", "% No": f"{no_y:.0f}%"})

if rows:
    tbl_year = pd.DataFrame(rows)
    tbl_year["Year"] = pd.Categorical(tbl_year["Year"], categories=order_years, ordered=True)
    tbl_year = tbl_year.sort_values("Year")
    st.dataframe(tbl_year.set_index("Year"), use_container_width=True)
else:
    st.info("No yearly referral data found to populate the percentages table.")

# ---------- Q2 by section (table)
st.subheader("Table 3. Would you refer someone to work here? — by section (2025)")

if sec_col and q2_col:
    sec_series = df[sec_col].astype(str).map(normalize_section)
    q2v = df[q2_col].apply(normalize_yesno)
    tidy = pd.DataFrame({"Section": sec_series, "Refer": q2v}).dropna()

    bucket_order = [
        "NCPRO",
        "Budget Execution",
        "Budget Development",
        "DEA or Internal Audit",
        "Business Office/Grants/IT/Comms",
        "Intern/Other",
    ]
    def pct_yes(g):
        v = g.dropna()
        return (v == "Yes").mean() * 100 if not v.empty else 0

    # counts and % Yes
    counts = tidy["Section"].value_counts().reindex(bucket_order, fill_value=0)
    pctyes = tidy.groupby("Section")["Refer"].apply(pct_yes).reindex(bucket_order, fill_value=0)

    tbl_q2 = pd.DataFrame({
        "Section": bucket_order,
        "n": counts.values,
        "% Yes": pctyes.round(0).astype(int).astype(str) + "%"
    })

    total_row = pd.DataFrame({
        "Section": ["Total"],
        "n": [tbl_q2["n"].sum()],
        "% Yes": [f"{round((tidy['Refer']=='Yes').mean()*100):.0f}%"]
    })
    tbl_q2 = pd.concat([tbl_q2, total_row], ignore_index=True)

    st.dataframe(tbl_q2, use_container_width=True)
else:
    st.info("Q2-by-section table unavailable (missing Q2 or Section column).")




# ======================================================
# ---------- UI helper: checkbox-based section selector ----------
def section_checkbox_selector(options: List[str], key_prefix: str) -> List[str]:
    """Render checkboxes for section filter (using widget keys only); returns selected list.
    Includes Select All / Clear All that directly set the checkbox widget states.
    """
    if not options:
        return []
    # Ensure widget keys exist and default selected = True
    for opt in options:
        k_cb = f"{key_prefix}_{opt}_cb"
        if k_cb not in st.session_state:
            st.session_state[k_cb] = True

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Select all sections", key=f"{key_prefix}_all"):
            for opt in options:
                st.session_state[f"{key_prefix}_{opt}_cb"] = True
            # force immediate rerun so downstream masks update this cycle
            if hasattr(st, "rerun"):
                st.rerun()
            else:
                st.experimental_rerun()
    with c2:
        if st.button("Clear all", key=f"{key_prefix}_none"):
            for opt in options:
                st.session_state[f"{key_prefix}_{opt}_cb"] = False
            if hasattr(st, "rerun"):
                st.rerun()
            else:
                st.experimental_rerun()

    # Render checkboxes in two columns; state fully controlled by widget keys
    cols = st.columns(2)
    for i, opt in enumerate(options):
        col = cols[i % 2]
        with col:
            st.checkbox(opt, key=f"{key_prefix}_{opt}_cb")

    selected = [opt for opt in options if st.session_state.get(f"{key_prefix}_{opt}_cb", False)]
    return selected

# ======================================================
# Figure 6. If you would refer someone to work at OSBM, why? If not, why not?
# ======================================================
st.header("Figure 6. If you would refer someone to work at OSBM, why? If not, why not?")

# Exact expected column names from the instrument
Q2_TARGET = "Would you refer someone to work here?"
Q3_TARGET = "If you would refer someone to work at OSBM, why? If not, why not?"

def find_exact_column(d: pd.DataFrame, target: str) -> Optional[str]:
    t = target.strip().lower()
    for c in d.columns:
        if str(c).strip().lower() == t:
            return c
    return None

# Resolve Q2 and Q3 columns with exact names first; fall back to heuristics if needed.
q2_exact = find_exact_column(df, Q2_TARGET)
q3_exact = find_exact_column(df, Q3_TARGET)

q2_col_final = q2_exact or find_q2_column(df)
q3_col_final = q3_exact  # Do not allow user selection; require this exact column
if (q2_col_final is None) or (q3_col_final is None):
    missing = []
    if q2_col_final is None:
        missing.append(f"`{Q2_TARGET}`")
    if q3_col_final is None:
        missing.append(f"`{Q3_TARGET}`")
    st.info("Referral comments explorer unavailable. Missing column(s): " + ", ".join(missing))
else:
    # Radio to choose cohort; comments column is fixed
    choice = st.radio("Show comments for referral response:", ["Yes", "No"], horizontal=True)

    # Build tidy comments conditioned on Q2, with Section for filtering
    if sec_col:
        sec_series = df[sec_col].astype(str).map(normalize_section)
        tidy_comments = pd.DataFrame({
            "Refer": df[q2_col_final].apply(normalize_yesno),
            "Section": sec_series,
            "Comment": df[q3_col_final].astype(str).str.strip()
        }).dropna(subset=["Refer", "Comment"])
        present_sections = sorted([s for s in tidy_comments["Section"].dropna().unique()])
        st.caption("Filter comments by section:")
        sel_secs = section_checkbox_selector(present_sections, key_prefix="q2q3")
        sec_mask = tidy_comments["Section"].isin(sel_secs) if sel_secs else False
    else:
        tidy_comments = pd.DataFrame({
            "Refer": df[q2_col_final].apply(normalize_yesno),
            "Comment": df[q3_col_final].astype(str).str.strip()
        }).dropna(subset=["Refer", "Comment"])
        sec_mask = True

    show_df = tidy_comments[
        (tidy_comments["Refer"] == choice) &
        (tidy_comments["Comment"].str.len() > 0) &
        (sec_mask)
    ][["Comment"]]
    st.dataframe(show_df, use_container_width=True, height=420)
    st.caption(f"{len(show_df):,} {choice} comment(s)")


# ======================================================
# Q5/Q6 — Re-apply & reasons
# ======================================================

# Exact expected column names from the instrument for Q5/Q6
Q5_TARGET = "Would you re-apply for your job?"
# The PDF shows: "Why or why not? Expanding on question 5."
# We'll try the short form first, then fall back to the longer form if present.
Q6_TARGETS = [
    "Why or why not?",
    "Why or why not? Expanding on question 5."
]

def find_exact_any(d: pd.DataFrame, targets: list[str]) -> Optional[str]:
    lows = [str(c).strip().lower() for c in d.columns]
    for t in targets:
        t_low = t.strip().lower()
        for i, low in enumerate(lows):
            if low == t_low:
                return d.columns[i]
    return None

def find_q5_column(d: pd.DataFrame) -> Optional[str]:
    exact = find_exact_any(d, [Q5_TARGET])
    if exact:
        return exact
    # heuristic fallback
    cols = [str(c).strip() for c in d.columns]
    low  = [c.lower() for c in cols]
    keys = ["re-apply", "reapply", "re apply", "would you re-apply", "would you reapply"]
    for i, c in enumerate(low):
        if any(k in c for k in keys):
            return cols[i]
    return None

def find_q6_column(d: pd.DataFrame) -> Optional[str]:
    exact = find_exact_any(d, Q6_TARGETS)
    if exact:
        return exact
    # heuristic fallback: a long-text column near Q5
    cols = [str(c).strip() for c in d.columns]
    low  = [c.lower() for c in cols]
    q5_idx = None
    q5_col_tmp = find_q5_column(d)
    if q5_col_tmp:
        try:
            q5_idx = low.index(q5_col_tmp.lower())
        except ValueError:
            q5_idx = None
    # pick a texty column after Q5
    for i, c in enumerate(cols):
        if q5_idx is not None and i <= q5_idx:
            continue
        s = d[c].astype(str)
        if s.map(len).mean() > 20 and s.nunique(dropna=True) > 10:
            return c
    return None

# ---------- Q5 current year (2025)
st.header("Figure 7. Would you re-apply for your job? (2025)")

q5_col = find_q5_column(df)
if not q5_col:
    st.info(f"Couldn’t locate Q5 in the dataset (expected: `{Q5_TARGET}`).")
else:
    q5_norm = df[q5_col].apply(normalize_yesno)
    valid5 = q5_norm.dropna()
    n_valid5 = len(valid5)
    yes5 = (valid5 == "Yes").mean() * 100 if n_valid5 else 0
    no5  = (valid5 == "No").mean() * 100 if n_valid5 else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Valid responses", f"{n_valid5:,}")
    c2.metric("Would re-apply (Yes)", f"{yes5:.0f}%")
    c3.metric("Would not (No)", f"{no5:.0f}%")

    # Donut chart
    pie5 = pd.DataFrame({"Response": ["Yes","No"], "Percent": [yes5, no5]})
    fig_q5 = px.pie(pie5, names="Response", values="Percent", hole=0.5,
                    title="Would you re-apply for your job? (2025)",
                    color="Response",
                    color_discrete_map={"Yes":"#1f77b4","No":"#7f7f7f"})
    fig_q5.update_traces(textposition="outside", texttemplate="%{label}: %{value:.0f}%")
    st.plotly_chart(fig_q5, use_container_width=True)

# ---------- Q5 trend (by year)
st.header("Figure 8. Would you re-apply for your job? (by year, 2023–2025)")

files_q5 = {
    "2023": DATA_DIR / "osbm_staff_survey_2023.xlsx",
    "2024": DATA_DIR / "osbm_staff_survey_2024.xlsx",
    "2025": DATA_DIR / "osbm_staff_survey_2025.xlsx",
}
order_years_q5 = ["2023","2024","2025"]
rows_trend5 = []
for y, path in files_q5.items():
    if not path.exists():
        continue
    d = read_excel_path(path)
    col5 = find_q5_column(d)
    if not col5:
        continue
    v = d[col5].apply(normalize_yesno).dropna()
    if v.empty:
        continue
    yes_y = (v == "Yes").mean() * 100
    rows_trend5.append({"Year": y, "% Yes": round(yes_y, 0)})

if rows_trend5:
    trend5 = pd.DataFrame(rows_trend5)
    trend5["Year"] = pd.Categorical(trend5["Year"], categories=order_years_q5, ordered=True)
    trend5 = trend5.sort_values("Year")
    fig_q5trend = go.Figure(go.Scatter(
        x=trend5["Year"], y=trend5["% Yes"],
        mode="lines+markers+text",
        line=dict(width=3, color="#1f77b4"),
        marker=dict(size=10),
        text=[f"{v:.0f}%" for v in trend5["% Yes"]],
        textposition="top center",
        hovertemplate="%{y:.0f}%% in %{x}<extra></extra>"
    ))
    fig_q5trend.update_layout(
        title="Re-apply (Yes) — Trend (2023–2025)",
        yaxis=dict(title="% Yes", range=[0,100], tick0=0, dtick=10),
        xaxis=dict(title="Survey Year", type="category"),
        margin=dict(l=40, r=20, t=60, b=40),
        height=420
    )
    st.plotly_chart(fig_q5trend, use_container_width=True)
else:
    st.info("No yearly re-apply data found to plot the trend.")

# ---------- Q5 by section (table)
st.subheader("Table 5. Would you re-apply for your job? — by section (2025)")

if sec_col and q5_col:
    sec_series = df[sec_col].astype(str).map(normalize_section)
    q5v = df[q5_col].apply(normalize_yesno)
    tidy5 = pd.DataFrame({"Section": sec_series, "Reapply": q5v}).dropna()

    bucket_order = [
        "NCPRO",
        "Budget Execution",
        "Budget Development",
        "DEA or Internal Audit",
        "Business Office/Grants/IT/Comms",
        "Intern/Other",
    ]
    def pct_yes5(g):
        v = g.dropna()
        return (v == "Yes").mean() * 100 if not v.empty else 0

    counts5 = tidy5["Section"].value_counts().reindex(bucket_order, fill_value=0)
    pctyes5 = tidy5.groupby("Section")["Reapply"].apply(pct_yes5).reindex(bucket_order, fill_value=0)

    tbl_q5 = pd.DataFrame({
        "Section": bucket_order,
        "n": counts5.values,
        "% Yes": pctyes5.round(0).astype(int).astype(str) + "%"
    })
    total_row5 = pd.DataFrame({
        "Section": ["Total"],
        "n": [tbl_q5["n"].sum()],
        "% Yes": [f"{round((tidy5['Reapply']=='Yes').mean()*100):.0f}%"]
    })
    tbl_q5 = pd.concat([tbl_q5, total_row5], ignore_index=True)
    st.dataframe(tbl_q5, use_container_width=True)
else:
    st.info("Q5-by-section table unavailable (missing Q5 or Section column).")

# ---------- Q6 comments (conditional on Q5)
st.header("Figure 9. Why or why not? (conditional on re-apply response, 2025)")

q6_col = find_q6_column(df)
if (q5_col is None) or (q6_col is None):
    miss = []
    if q5_col is None:
        miss.append(f"`{Q5_TARGET}`")
    if q6_col is None:
        miss.append("`Why or why not?`")
    st.info("Re-apply comments explorer unavailable. Missing: " + ", ".join(miss))
else:
    choice5 = st.radio("Show comments for re-apply response:", ["Yes", "No"], horizontal=True)
    # Build tidy re-apply comments with Section for filtering
    if sec_col:
        sec_series = df[sec_col].astype(str).map(normalize_section)
        tidy6 = pd.DataFrame({
            "Reapply": df[q5_col].apply(normalize_yesno),
            "Section": sec_series,
            "Comment": df[q6_col].astype(str).str.strip()
        }).dropna(subset=["Reapply","Comment"])
        present_sections6 = sorted([s for s in tidy6["Section"].dropna().unique()])
        st.caption("Filter comments by section:")
        sel_secs6 = section_checkbox_selector(present_sections6, key_prefix="q5q6")
        sec_mask6 = tidy6["Section"].isin(sel_secs6) if sel_secs6 else False
    else:
        tidy6 = pd.DataFrame({
            "Reapply": df[q5_col].apply(normalize_yesno),
            "Comment": df[q6_col].astype(str).str.strip()
        }).dropna(subset=["Reapply","Comment"])
        sec_mask6 = True

    show6 = tidy6[
        (tidy6["Reapply"] == choice5) &
        (tidy6["Comment"].str.len() > 0) &
        (sec_mask6)
    ][["Comment"]]
    st.dataframe(show6, use_container_width=True, height=420)

    st.caption(f"{len(show6):,} {choice5} comment(s)")


# ======================================================
# Figure 10. Please provide your sentiment towards the statements below (stacked percentages, 2025)
# ======================================================
st.header("Figure 10. Please provide your sentiment towards the statements below")

# We will try to detect the five Q4 statements by keywords.
LIKERT_ORDER = [
    "Strongly Disagree",
    "Disagree",
    "Slightly Disagree",
    "Slightly Agree",
    "Agree",
    "Strongly Agree",
]

# (removed duplicate import of Optional)
# Normalize raw responses to the canonical 6-point order above
def normalize_likert(x: str) -> Optional[str]:
    if not isinstance(x, str):
        return None
    v = " ".join(str(x).strip().split())  # collapse internal whitespace
    vl = v.lower()

    # Priority order: most specific first, then generic terms last
    rules = [
        ("strongly disagree", "Strongly Disagree"),
        ("slightly disagree", "Slightly Disagree"),
        ("slightly agree", "Slightly Agree"),
        ("strongly agree", "Strongly Agree"),
        ("disagree", "Disagree"),
        ("agree", "Agree"),
    ]

    # Try exact matches first (case-insensitive)
    for pat, label in rules:
        if vl == pat:
            return label

    # Then substring (handles trailing text or minor variations)
    for pat, label in rules:
        if pat in vl:
            return label

    return None

# Define target items with indicative keywords and readable labels
Q4_ITEMS = [
    (("clear", "career"), "I have a clear understanding of my career or promotion path."),
    (("valued",), "I feel valued at work."),
    (("full", "potential"), "I'll be able to reach my full potential at OSBM."),
    (("reapply", "current"), "If given the chance, I would reapply to my current job."),
    (("life", "work", "harmony"), "I am able to maintain life–work harmony."),
]

# Try to find matching columns for each item
cols_lower = [str(c).strip() for c in df.columns]
cols_lower_lc = [c.lower() for c in cols_lower]

def find_col_by_keywords(keywords: Tuple[str, ...]) -> Optional[str]:
    for i, c in enumerate(cols_lower_lc):
        if all(k in c for k in keywords):
            return cols_lower[i]
    return None

found = []
for keys, label in Q4_ITEMS:
    col = find_col_by_keywords(keys)
    if col is not None:
        found.append((label, col))

if not found:
    st.info("Couldn’t locate the Q4 Likert statements in the dataset.")
else:
    # ---- Q4 filters: Section + cluster selection ----
    # Section filter (checkboxes)
    if sec_col:
        sec_norm = df[sec_col].astype(str).map(normalize_section)
        present_sections4 = sorted([s for s in sec_norm.dropna().unique()])
        st.caption("Filter statements by section:")
        sel_secs4 = section_checkbox_selector(present_sections4, key_prefix="q4_sections")
        mask4 = sec_norm.isin(sel_secs4) if sel_secs4 else pd.Series(False, index=df.index)
    else:
        mask4 = pd.Series(True, index=df.index)
    # Build filtered frame for Q4 tables/figures
    df_q4 = df.loc[mask4].copy()
    n_pool4 = int(len(df_q4))
    st.caption(f"Current section filter (Q4): {', '.join(sel_secs4) if sec_col else 'All sections'} — N = {n_pool4}")

    # Cluster selector (bottom / middle / top) for the figure
    st.caption("Select which rating clusters to display in the chart:")
    # Initialize defaults once
    for k, default in [("q4_bottom", True), ("q4_middle", True), ("q4_top", True)]:
        if k not in st.session_state:
            st.session_state[k] = default
    cA, cB, cC = st.columns(3)
    with cA:
        st.checkbox("Bottom-box (SD + D)", key="q4_bottom")
    with cB:
        st.checkbox("Middle (Slightly)", key="q4_middle")
    with cC:
        st.checkbox("Top-box (A + SA)", key="q4_top")
    q4_show_bottom = st.session_state["q4_bottom"]
    q4_show_middle = st.session_state["q4_middle"]
    q4_show_top = st.session_state["q4_top"]

    # Derive which categories to include on each side, keeping center-adjacent ordering logic
    neg_selected = []
    pos_selected = []
    if q4_show_bottom:
        neg_selected += ["Strongly Disagree", "Disagree"]
    if q4_show_middle:
        neg_selected += ["Slightly Disagree"]
        pos_selected += ["Slightly Agree"]
    if q4_show_top:
        pos_selected += ["Agree", "Strongly Agree"]

    # Order to stack: left side nearest to zero first
    neg_order = [cat for cat in ["Slightly Disagree", "Disagree", "Strongly Disagree"] if cat in neg_selected]
    pos_order = [cat for cat in ["Slightly Agree", "Agree", "Strongly Agree"] if cat in pos_selected]

    # Selected categories and Displayed N (rows with any selected category across Q4 items)
    show_cats_q4 = neg_order + pos_order
    if show_cats_q4:
        mask_display_q4 = pd.Series(False, index=df_q4.index)
        for _, col in found:
            mask_display_q4 |= df_q4[col].apply(normalize_likert).isin(show_cats_q4)
        displayed_n_q4 = int(mask_display_q4.sum())
    else:
        displayed_n_q4 = 0


    # Prepare percentages per item/category for plotting
    item_labels = []
    pct_by_cat = {cat: [] for cat in LIKERT_ORDER}

    for label, col in found:
        s = df_q4[col].apply(normalize_likert).dropna()
        counts = (s.value_counts(normalize=True) * 100)
        item_labels.append(label)
        for cat in LIKERT_ORDER:
            pct_by_cat[cat].append(round(float(counts.get(cat, 0.0)), 0))
    # --- Diverging stacked (centered at 0) ---
    # Left side (Disagree) -> negative; Right side (Agree) -> positive
    NEG_CATS = ["Strongly Disagree", "Disagree", "Slightly Disagree"]
    POS_CATS = ["Slightly Agree", "Agree", "Strongly Agree"]

    # Okabe–Ito colorblind-safe palette (orange vs blue families)
    color_map = {
        "Strongly Disagree": "#D55E00",   # strong orange
        "Disagree": "#E69F00",            # orange
        "Slightly Disagree": "#F0E442",   # yellow-ish (kept on left, still accessible)
        "Slightly Agree": "#56B4E9",      # sky blue
        "Agree": "#0072B2",               # blue
        "Strongly Agree": "#004B87",      # dark blue
    }

    fig_q4 = go.Figure()

    # Left side: ensure Slightly Disagree is adjacent to zero, then Disagree, then Strongly Disagree outward.
    # Because in negative stacking the FIRST trace sits closest to zero, add Slightly first, then Disagree, then Strongly.
    for cat in neg_order:
        fig_q4.add_trace(
            go.Bar(
                x=[-v for v in pct_by_cat[cat]],
                y=item_labels,
                name=cat,
                orientation="h",
                marker_color=color_map[cat],
                hovertemplate=f"{cat}: "+"%{customdata:.0f}%<extra></extra>",
                customdata=pct_by_cat[cat],
                legendrank=10 if cat=="Strongly Disagree" else (20 if cat=="Disagree" else 30)
            )
        )

    # Right side: add traces in order: Slightly Agree (adjacent to zero), Agree, Strongly Agree (farthest)
    for cat in pos_order:
        fig_q4.add_trace(
            go.Bar(
                x=pct_by_cat[cat],
                y=item_labels,
                name=cat,
                orientation="h",
                marker_color=color_map[cat],
                hovertemplate=f"{cat}: "+"%{customdata:.0f}%<extra></extra>",
                customdata=pct_by_cat[cat],
                legendrank=40 if cat=="Slightly Agree" else (50 if cat=="Agree" else 60)
            )
        )

    fig_q4.update_layout(
        barmode="relative",  # so negatives stack to the left and positives to the right
        xaxis=dict(
            title="% of responses",
            range=[-100, 100],
            tickmode="array",
            tickvals=[-100,-90,-80,-70,-60,-50,-40,-30,-20,-10,0,10,20,30,40,50,60,70,80,90,100],
            ticktext=[f"{abs(v)}%" for v in [-100,-90,-80,-70,-60,-50,-40,-30,-20,-10,0,10,20,30,40,50,60,70,80,90,100]],
            zeroline=True, zerolinewidth=2, zerolinecolor="#888"
        ),
        yaxis=dict(
            autorange="reversed",
            tickfont=dict(size=16, color="black"),
            automargin=True
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, traceorder="normal"),
        margin=dict(l=40, r=20, t=60, b=40),
        height=440 + 40 * max(0, len(item_labels) - 4)
    )


    st.plotly_chart(fig_q4, use_container_width=True)
    st.caption(f"Displayed N (based on current selections) = {displayed_n_q4}.")

    # ----- Table 6: counts and percentages by statement & response (always show all categories) -----
    rows_tbl = []
    for label, col in found:
        s = df_q4[col].apply(normalize_likert).dropna()
        # If clusters are selected, restrict counts/denominator to those clusters
        if show_cats_q4:
            s_sel = s[s.isin(show_cats_q4)]
        else:
            s_sel = s.iloc[0:0]  # empty selection when no clusters selected
        total = len(s_sel)
        row = {"Statement": label}
        counts = s_sel.value_counts()
        for cat in LIKERT_ORDER:
            n = int(counts.get(cat, 0))
            pct = (n / total * 100) if total else 0
            row[cat] = f"{n} ({pct:.0f}%)"
        rows_tbl.append(row)
    tbl_q4 = pd.DataFrame(rows_tbl, columns=["Statement"] + LIKERT_ORDER)
    st.dataframe(tbl_q4, use_container_width=True)
    st.caption(
        f"Displayed N (based on current selections) = {displayed_n_q4}. "
        "Per-statement N may vary due to missing answers."
    )

    # If no clusters are selected, display info and zero displayed N
    if not show_cats_q4:
        st.info("Select at least one rating cluster to display the chart.")
        st.caption("Displayed N (based on current selections) = 0.")
# ======================================================
# Figure 11. Thinking about organization culture, please provide your sentiment toward the statements below (diverging stacked, 2025)
# ======================================================
st.header("Figure 11. Organization culture statements — diverging stacked (2025)")

# Q10 items (detected by keywords); aligns with instrument language
Q10_ITEMS = [
    (("coworkers", "respect"), "My OSBM coworkers respect each other."),
    (("feel", "respected"), "I feel respected as an employee at OSBM."),
    (("management", "positive", "culture"), "The OSBM Management Team contributes to a positive work culture."),
    (("mission", "vision"), "I know what OSBM’s mission and vision are."),
]

# Try to find matching columns for each Q10 item
cols_lower = [str(c).strip() for c in df.columns]
cols_lower_lc = [c.lower() for c in cols_lower]

def find_q10_col_by_keywords(keywords: Tuple[str, ...]) -> Optional[str]:
    for i, c in enumerate(cols_lower_lc):
        if all(k in c for k in keywords):
            return cols_lower[i]
    return None

found_q10 = []
for keys, label in Q10_ITEMS:
    col = find_q10_col_by_keywords(keys)
    if col is not None:
        found_q10.append((label, col))

if not found_q10:
    st.info("Couldn’t locate the Q10 organization-culture statements in the dataset.")
else:
    # ---- Q10 filters: Section + cluster selection ----
    if sec_col:
        sec_norm10 = df[sec_col].astype(str).map(normalize_section)
        present_sections10 = sorted([s for s in sec_norm10.dropna().unique()])
        st.caption("Filter statements by section:")
        sel_secs10 = section_checkbox_selector(present_sections10, key_prefix="q10_sections")
        mask10 = sec_norm10.isin(sel_secs10) if sel_secs10 else pd.Series(False, index=df.index)
    else:
        mask10 = pd.Series(True, index=df.index)
    # Build filtered frame for Q10 tables/figures
    df_q10 = df.loc[mask10].copy()
    n_pool10 = int(len(df_q10))
    st.caption(f"Current section filter (Q10): {', '.join(sel_secs10) if sec_col else 'All sections'} — N = {n_pool10}")

    st.caption("Select which rating clusters to display in the chart:")
    for k, default in [("q10_bottom", True), ("q10_middle", True), ("q10_top", True)]:
        if k not in st.session_state:
            st.session_state[k] = default
    dA, dB, dC = st.columns(3)
    with dA:
        st.checkbox("Bottom-box (SD + D)", key="q10_bottom")
    with dB:
        st.checkbox("Middle (Slightly)", key="q10_middle")
    with dC:
        st.checkbox("Top-box (A + SA)", key="q10_top")
    q10_show_bottom = st.session_state["q10_bottom"]
    q10_show_middle = st.session_state["q10_middle"]
    q10_show_top = st.session_state["q10_top"]

    neg_sel10 = []
    pos_sel10 = []
    if q10_show_bottom:
        neg_sel10 += ["Strongly Disagree", "Disagree"]
    if q10_show_middle:
        neg_sel10 += ["Slightly Disagree"]
        pos_sel10 += ["Slightly Agree"]
    if q10_show_top:
        pos_sel10 += ["Agree", "Strongly Agree"]

    neg_order10 = [cat for cat in ["Slightly Disagree", "Disagree", "Strongly Disagree"] if cat in neg_sel10]
    pos_order10 = [cat for cat in ["Slightly Agree", "Agree", "Strongly Agree"] if cat in pos_sel10]

    show_cats_q10 = neg_order10 + pos_order10
    if show_cats_q10:
        mask_display_q10 = pd.Series(False, index=df_q10.index)
        for _, col in found_q10:
            mask_display_q10 |= df_q10[col].apply(normalize_likert).isin(show_cats_q10)
        displayed_n_q10 = int(mask_display_q10.sum())
    else:
        displayed_n_q10 = 0


    # Prepare percentages per item/category for plotting
    item_labels10 = []
    pct_by_cat10 = {cat: [] for cat in LIKERT_ORDER}

    for label, col in found_q10:
        s = df_q10[col].apply(normalize_likert).dropna()
        counts = (s.value_counts(normalize=True) * 100)
        item_labels10.append(label)
        for cat in LIKERT_ORDER:
            pct_by_cat10[cat].append(round(float(counts.get(cat, 0.0)), 0))

    # Diverging stacked (centered at 0), colorblind-safe
    NEG_CATS = ["Strongly Disagree", "Disagree", "Slightly Disagree"]
    POS_CATS = ["Slightly Agree", "Agree", "Strongly Agree"]
    color_map = {
        "Strongly Disagree": "#D55E00",
        "Disagree": "#E69F00",
        "Slightly Disagree": "#F0E442",
        "Slightly Agree": "#56B4E9",
        "Agree": "#0072B2",
        "Strongly Agree": "#004B87",
    }

    fig_q10 = go.Figure()

    # Left side: Slightly Disagree near zero, then Disagree, Strongly Disagree farthest
    for cat in neg_order10:
        fig_q10.add_trace(
            go.Bar(
                x=[-v for v in pct_by_cat10[cat]],
                y=item_labels10,
                name=cat,
                orientation="h",
                marker_color=color_map[cat],
                hovertemplate=f"{cat}: "+"%{customdata:.0f}%<extra></extra>",
                customdata=pct_by_cat10[cat],
                legendrank=10 if cat=="Strongly Disagree" else (20 if cat=="Disagree" else 30)
            )
        )

    # Right side: Slightly Agree near zero, then Agree, then Strongly Agree
    for cat in pos_order10:
        fig_q10.add_trace(
            go.Bar(
                x=pct_by_cat10[cat],
                y=item_labels10,
                name=cat,
                orientation="h",
                marker_color=color_map[cat],
                hovertemplate=f"{cat}: "+"%{customdata:.0f}%<extra></extra>",
                customdata=pct_by_cat10[cat],
                legendrank=40 if cat=="Slightly Agree" else (50 if cat=="Agree" else 60)
            )
        )

    fig_q10.update_layout(
        barmode="relative",
        xaxis=dict(
            title="% of responses",
            range=[-100, 100],
            tickmode="array",
            tickvals=[-100,-90,-80,-70,-60,-50,-40,-30,-20,-10,0,10,20,30,40,50,60,70,80,90,100],
            ticktext=[f"{abs(v)}%" for v in [-100,-90,-80,-70,-60,-50,-40,-30,-20,-10,0,10,20,30,40,50,60,70,80,90,100]],
            zeroline=True, zerolinewidth=2, zerolinecolor="#888"
        ),
        yaxis=dict(
            autorange="reversed",
            tickfont=dict(size=16, color="black"),
            automargin=True
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, traceorder="normal"),
        margin=dict(l=40, r=20, t=60, b=40),
        height=440 + 40 * max(0, len(item_labels10) - 4)
    )
    st.plotly_chart(fig_q10, use_container_width=True)
    st.caption(f"Displayed N (based on current selections) = {displayed_n_q10}.")

    # ----- Table 7: counts and percentages by statement & response (always show all categories) -----
    rows_tbl10 = []
    for label, col in found_q10:
        s = df_q10[col].apply(normalize_likert).dropna()
        if show_cats_q10:
            s_sel = s[s.isin(show_cats_q10)]
        else:
            s_sel = s.iloc[0:0]
        total = len(s_sel)
        row = {"Statement": label}
        counts = s_sel.value_counts()
        for cat in LIKERT_ORDER:
            n = int(counts.get(cat, 0))
            pct = (n / total * 100) if total else 0
            row[cat] = f"{n} ({pct:.0f}%)"
        rows_tbl10.append(row)
    tbl_q10 = pd.DataFrame(rows_tbl10, columns=["Statement"] + LIKERT_ORDER)
    st.dataframe(tbl_q10, use_container_width=True)
    st.caption(
        f"Displayed N (based on current selections) = {displayed_n_q10}. "
        "Per-statement N may vary due to missing answers."
    )

    # If no clusters are selected, display info and zero displayed N
    if not show_cats_q10:
        st.info("Select at least one rating cluster to display the chart.")
        st.caption("Displayed N (based on current selections) = 0.")



# ======================================================
# Figure 12. Sentiment toward selected statements (Q4, Q10, Q21) — diverging stacked (2025)
# ======================================================
st.header("Figure 12. Sentiment toward selected statements (Q4, Q10, Q21)")

# --- Detect Q21 (single Likert statement) ---
# First, try keyword hints; else pick the best generic Likert-looking column not already included in Q4/Q10

def find_q21_column(d: pd.DataFrame, exclude: List[str]) -> Optional[Tuple[str, str]]:
    cols = [str(c).strip() for c in d.columns]
    low  = [c.lower() for c in cols]
    # Heuristic keywords for Q21 (communications / all-hands area)
    kwsets = [
        ("all-hands", "value"),
        ("all hands", "value"),
        ("all-hands", "meeting"),
        ("all hands", "meeting"),
        ("informed", "office"),
        ("across", "office"),
        ("cadence", "all"),
    ]
    for i, c in enumerate(low):
        if cols[i] in exclude:
            continue
        for kws in kwsets:
            if all(k in c for k in kws):
                return (cols[i], cols[i])
    # Generic Likert-like detector: choose the column with most mappable Likert responses
    best = None
    best_count = 0
    for i, c in enumerate(cols):
        if cols[i] in exclude:
            continue
        s = d[cols[i]].apply(normalize_likert)
        ok = s.dropna()
        # Keep if majority of non-null map to our 6 categories
        if len(ok) and (len(ok) / max(1, d[cols[i]].notna().sum())) >= 0.6:
            if len(ok) > best_count:
                best = cols[i]
                best_count = len(ok)
    if best:
        return (best, best)
    return None

# Build master item list from Q4/Q10 and detected Q21
items_all: List[Tuple[str, str]] = []
# Q4
if 'found' in locals() and found:
    items_all += found
# Q10
if 'found_q10' in locals() and found_q10:
    items_all += found_q10
# Q21
exclude_cols = [col for _, col in items_all]
q21 = find_q21_column(df, exclude_cols)
if q21 is not None:
    # Use the column header as the label
    items_all.append((q21[0], q21[1]))

if not items_all:
    st.info("Couldn’t locate Q4/Q10/Q21 Likert statements to combine.")
else:
    # ---- Filters: Section + clusters (shared) ----
    if sec_col:
        sec_normC = df[sec_col].astype(str).map(normalize_section)
        present_sectionsC = sorted([s for s in sec_normC.dropna().unique()])
        st.caption("Filter statements by section:")
        sel_secsC = section_checkbox_selector(present_sectionsC, key_prefix="combo_sections")
        maskC = sec_normC.isin(sel_secsC) if sel_secsC else pd.Series(False, index=df.index)
    else:
        maskC = pd.Series(True, index=df.index)
    # Filtered frame
    df_combo = df.loc[maskC].copy()

    st.caption("Select which rating clusters to display in the chart:")
    for k, default in [("combo_bottom", True), ("combo_middle", True), ("combo_top", True)]:
        if k not in st.session_state:
            st.session_state[k] = default
    uA, uB, uC = st.columns(3)
    with uA:
        st.checkbox("Bottom-box (SD + D)", key="combo_bottom")
    with uB:
        st.checkbox("Middle (Slightly)", key="combo_middle")
    with uC:
        st.checkbox("Top-box (A + SA)", key="combo_top")
    show_bottom = st.session_state["combo_bottom"]
    show_middle = st.session_state["combo_middle"]
    show_top    = st.session_state["combo_top"]

    neg_selC, pos_selC = [], []
    if show_bottom:
        neg_selC += ["Strongly Disagree", "Disagree"]
    if show_middle:
        neg_selC += ["Slightly Disagree"]; pos_selC += ["Slightly Agree"]
    if show_top:
        pos_selC += ["Agree", "Strongly Agree"]

    neg_orderC = [c for c in ["Slightly Disagree", "Disagree", "Strongly Disagree"] if c in neg_selC]
    pos_orderC = [c for c in ["Slightly Agree", "Agree", "Strongly Agree"] if c in pos_selC]
    show_catsC  = neg_orderC + pos_orderC

    # Displayed N across all included items and selected clusters
    if show_catsC:
        mask_display_C = pd.Series(False, index=df_combo.index)
        for _, col in items_all:
            mask_display_C |= df_combo[col].apply(normalize_likert).isin(show_catsC)
        displayed_n_C = int(mask_display_C.sum())
    else:
        displayed_n_C = 0

    # Percentages per item/category
    item_labelsC = []
    pct_by_catC = {cat: [] for cat in LIKERT_ORDER}
    for label, col in items_all:
        s = df_combo[col].apply(normalize_likert).dropna()
        counts = (s.value_counts(normalize=True) * 100)
        item_labelsC.append(label)
        for cat in LIKERT_ORDER:
            pct_by_catC[cat].append(round(float(counts.get(cat, 0.0)), 0))

    # Chart — diverging stacked
    color_map = {
        "Strongly Disagree": "#D55E00",
        "Disagree": "#E69F00",
        "Slightly Disagree": "#F0E442",
        "Slightly Agree": "#56B4E9",
        "Agree": "#0072B2",
        "Strongly Agree": "#004B87",
    }
    fig_combo = go.Figure()
    for cat in neg_orderC:
        fig_combo.add_trace(go.Bar(
            x=[-v for v in pct_by_catC[cat]], y=item_labelsC, name=cat, orientation="h",
            marker_color=color_map[cat], hovertemplate=f"{cat}: "+"%{customdata:.0f}%<extra></extra>",
            customdata=pct_by_catC[cat], legendrank=10 if cat=="Strongly Disagree" else (20 if cat=="Disagree" else 30)
        ))
    for cat in pos_orderC:
        fig_combo.add_trace(go.Bar(
            x=pct_by_catC[cat], y=item_labelsC, name=cat, orientation="h",
            marker_color=color_map[cat], hovertemplate=f"{cat}: "+"%{customdata:.0f}%<extra></extra>",
            customdata=pct_by_catC[cat], legendrank=40 if cat=="Slightly Agree" else (50 if cat=="Agree" else 60)
        ))
    fig_combo.update_layout(
        barmode="relative",
        xaxis=dict(title="% of responses", range=[-100,100], tickmode="array",
                   tickvals=[-100,-90,-80,-70,-60,-50,-40,-30,-20,-10,0,10,20,30,40,50,60,70,80,90,100],
                   ticktext=[f"{abs(v)}%" for v in [-100,-90,-80,-70,-60,-50,-40,-30,-20,-10,0,10,20,30,40,50,60,70,80,90,100]],
                   zeroline=True, zerolinewidth=2, zerolinecolor="#888"),
        yaxis=dict(autorange="reversed", tickfont=dict(size=16, color="black"), automargin=True),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, traceorder="normal"),
        margin=dict(l=40, r=20, t=60, b=40),
        height=440 + 40 * max(0, len(item_labelsC) - 4)
    )
    st.plotly_chart(fig_combo, use_container_width=True)
    st.caption(f"Displayed N (based on current selections) = {displayed_n_C}.")

    # Table under the chart — always show all six categories, but counts based on selected clusters
    rows_tblC = []
    for label, col in items_all:
        s = df_combo[col].apply(normalize_likert).dropna()
        if show_catsC:
            s_sel = s[s.isin(show_catsC)]
        else:
            s_sel = s.iloc[0:0]
        total = len(s_sel)
        row = {"Statement": label}
        counts = s_sel.value_counts()
        for cat in LIKERT_ORDER:
            n = int(counts.get(cat, 0))
            pct = (n / total * 100) if total else 0
            row[cat] = f"{n} ({pct:.0f}%)"
        rows_tblC.append(row)
    tbl_combo = pd.DataFrame(rows_tblC, columns=["Statement"] + LIKERT_ORDER)
    st.dataframe(tbl_combo, use_container_width=True)
    st.caption("Counts/percentages reflect current section & cluster selections. Per-statement N may vary due to missing answers.")