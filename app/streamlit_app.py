# app/streamlit_app.py
# Run from repo root:  streamlit run app/streamlit_app.py

# -------------------- Imports --------------------
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, List, Tuple, Dict

# -------------------- Global styling --------------------
st.set_page_config(page_title="OSBM Survey Report", layout="wide")
st.markdown(
    """
    <style>
      .stMarkdown h2 { font-size: 1.5rem !important; } /* st.header */
      .stMarkdown h3 { font-size: 1.5rem !important; } /* st.subheader */
      .stDataFrame { font-size: 0.95rem; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("OSBM Employee Satisfaction — 2024–2025")

# -------------------- Paths & readers --------------------
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

# -------------------- Likert normalization --------------------
LIKERT_ORDER = [
    "Strongly Disagree",
    "Disagree",
    "Slightly Disagree",
    "Slightly Agree",
    "Agree",
    "Strongly Agree",
]

def normalize_likert(x: str) -> Optional[str]:
    if not isinstance(x, str):
        return None
    v = " ".join(str(x).strip().split())
    vl = v.lower()
    rules = [
        ("strongly disagree", "Strongly Disagree"),
        ("slightly disagree", "Slightly Disagree"),
        ("slightly agree", "Slightly Agree"),
        ("strongly agree", "Strongly Agree"),
        ("disagree", "Disagree"),
        ("agree", "Agree"),
    ]
    for pat, label in rules:
        if vl == pat:
            return label
    for pat, label in rules:
        if pat in vl:
            return label
    return None

# -------------------- Yes/No normalization --------------------
def normalize_yesno(val) -> Optional[str]:
    if isinstance(val, str):
        v = val.strip().lower()
        if v in {"yes", "y", "true", "t", "1"}: return "Yes"
        if v in {"no", "n", "false", "f", "0"}: return "No"
    if isinstance(val, bool):
        return "Yes" if val else "No"
    if isinstance(val, (int, float)):
        if val == 1: return "Yes"
        if val == 0: return "No"
    return None

# -------------------- Section (Q32) helpers --------------------
def find_section_column(d: pd.DataFrame) -> Optional[str]:
    cols = [str(c).strip() for c in d.columns]
    low  = [c.lower() for c in cols]
    for i, c in enumerate(low):
        if "which best describes the section you work in" in c:
            return cols[i]
    for i, c in enumerate(low):
        if "section" in c and ("work" in c or "describes" in c or "best" in c):
            return cols[i]
    for i, c in enumerate(low):
        if "section" in c:
            return cols[i]
    return None

def normalize_section(val: str) -> Optional[str]:
    if not isinstance(val, str):
        return None
    v = val.strip().lower()
    if "ncpro" in v: return "NCPRO"
    if "budget execution" in v: return "Budget Execution"
    if "budget development" in v: return "Budget Development"
    if "dea" in v or "internal audit" in v: return "DEA or Internal Audit"
    if "business office" in v or "grants" in v or "it" in v or "comms" in v or "communications" in v:
        return "Business Office/Grants/IT/Comms"
    if "intern" in v or "other" in v: return "Intern/Other"
    return "Intern/Other"

sec_col = None  # will set after load

# -------------------- UI helpers --------------------
def section_checkbox_selector(options: List[str], key_prefix: str) -> List[str]:
    """Checkbox list with Select all/Clear all and immediate rerun."""
    if not options: return []
    for opt in options:
        k = f"{key_prefix}_{opt}_cb"
        if k not in st.session_state: st.session_state[k] = True
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Select all sections", key=f"{key_prefix}_all"):
            for opt in options: st.session_state[f"{key_prefix}_{opt}_cb"] = True
            st.rerun() if hasattr(st, "rerun") else st.experimental_rerun()
    with c2:
        if st.button("Clear all", key=f"{key_prefix}_none"):
            for opt in options: st.session_state[f"{key_prefix}_{opt}_cb"] = False
            st.rerun() if hasattr(st, "rerun") else st.experimental_rerun()
    cols = st.columns(2)
    for i, opt in enumerate(options):
        with cols[i % 2]:
            st.checkbox(opt, key=f"{key_prefix}_{opt}_cb")
    return [opt for opt in options if st.session_state.get(f"{key_prefix}_{opt}_cb", False)]

def cluster_toggles(key_prefix: str) -> Tuple[bool, bool, bool]:
    defaults = [(f"{key_prefix}_bottom", True), (f"{key_prefix}_middle", True), (f"{key_prefix}_top", True)]
    for k, default in defaults:
        if k not in st.session_state: st.session_state[k] = default
    cA, cB, cC = st.columns(3)
    with cA: st.checkbox("Bottom-box (SD + D)", key=f"{key_prefix}_bottom")
    with cB: st.checkbox("Middle (Slightly)", key=f"{key_prefix}_middle")
    with cC: st.checkbox("Top-box (A + SA)", key=f"{key_prefix}_top")
    return (st.session_state[f"{key_prefix}_bottom"],
            st.session_state[f"{key_prefix}_middle"],
            st.session_state[f"{key_prefix}_top"])

# -------------------- Data load --------------------
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

# Intro — edit N as needed
N = 110
Z = len(df)
response_rate = (Z / N * 100) if N else np.nan
st.markdown(
    f"OSBM's annual employee survey was administered **July 2nd–15th**, 2025, to all OSBM employees (N = {N}).  \n"
    f"**{Z:,} people** completed the survey (a **{response_rate:.0f}% response rate**)."
)

with st.expander("Preview first 10 rows", expanded=False):
    st.dataframe(df.head(10), use_container_width=True)

# Detect section column
sec_col = find_section_column(df)

# ======================================================
# Table 1 — Respondents by Section
# ======================================================
st.subheader("Table 1. Respondents by Section")
if sec_col:
    sec_series = df[sec_col].astype(str).map(normalize_section)
    order_buckets = [
        "NCPRO","Budget Execution","Budget Development",
        "DEA or Internal Audit","Business Office/Grants/IT/Comms","Intern/Other"
    ]
    counts = sec_series.value_counts(dropna=False).reindex(order_buckets, fill_value=0)
    total_resp = int(len(df))
    tbl = pd.DataFrame({"Section": counts.index, "n": counts.values})
    tbl["%"] = (tbl["n"]/total_resp * 100).round(0).astype(int).astype(str) + "%"
    tbl = pd.concat([tbl, pd.DataFrame({"Section":["Total"], "n":[tbl["n"].sum()], "%":["100%"]})], ignore_index=True)
    st.dataframe(tbl, use_container_width=True)
else:
    st.info("Couldn't locate the Section (Q32) column.")

# ======================================================
# Figure 1 — Q1 Happiness (0–10) — Current
# ======================================================
st.header("Figure 1. On a scale of 1 to 10, how happy are you at work?")
def find_q1_column(d: pd.DataFrame) -> Optional[str]:
    cols = [str(c).strip() for c in d.columns]; low=[c.lower() for c in cols]
    for i,c in enumerate(low):
        if ("happy" in c or "happiness" in c) and ("0-10" in c or "0–10" in c or "work" in c): return cols[i]
    for i,c in enumerate(low):
        if ("happy" in c or "happiness" in c): return cols[i]
    candidates=[]
    for name in cols:
        s = pd.to_numeric(d[name], errors="coerce").dropna()
        if len(s) and (s.between(0,10).mean()>0.9) and s.nunique()<=11: candidates.append((name,len(s)))
    if candidates: return sorted(candidates, key=lambda x:x[1], reverse=True)[0][0]
    return None

q1_col = find_q1_column(df)
if not q1_col:
    st.warning("Couldn’t find Q1 happiness column.")
else:
    s = pd.to_numeric(df[q1_col], errors="coerce").dropna(); s = s[(s>=0)&(s<=10)]
    mean, median = s.mean(), s.median()
    top_box, bot_box = (s>=8).mean()*100, (s<=3).mean()*100
    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("Responses", f"{len(s):,}")
    k2.markdown(f"<div><small>Mean</small><br><span style='color:#f39c12;font-size:1.6rem;'>{mean:.1f}</span></div>", unsafe_allow_html=True)
    k3.markdown(f"<div><small>Median</small><br><span style='color:#1f77b4;font-size:1.6rem;'>{median:.1f}</span></div>", unsafe_allow_html=True)
    k4.metric("Top-box (8–10)", f"{top_box:.0f}%"); k5.metric("Bottom-box (0–3)", f"{bot_box:.0f}%")
    fig = px.histogram(s, x=q1_col, nbins=11, range_x=[-0.5,10.5], histnorm="percent",
                       title="Distribution of Happiness (0 = Unhappy, 10 = Extremely Happy)")
    fig.update_layout(xaxis=dict(tickmode="linear", dtick=1, range=[-0.5,10.5], title="Happiness"),
                      yaxis_title="% of responses")
    fig.update_yaxes(range=[0,50], tick0=0, dtick=10)
    fig.update_traces(texttemplate="%{y:.0f}%", textposition="outside",
                      hovertemplate="%{y:.0f}%% at %{x}<extra></extra>")
    fig.add_vline(x=mean, line_width=2, line_dash="dash", line_color="#f39c12")
    fig.add_vline(x=median, line_width=2, line_dash="dot", line_color="#1f77b4")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        f"**{top_box:.0f}%** (≈ two in three) rated **8–10** in 2025. Mean = **{mean:.1f}**, median = **{median:.1f}**; "
        f"**{bot_box:.0f}%** rated 0–3."
    )

# ======================================================
# Figure 2 — Q1 Happiness by Year (small multiples)
# ======================================================
st.header("Figure 2. On a scale of 1 to 10, how happy are you at work? (by year)")
year_files = {"2023": DATA_DIR/"osbm_staff_survey_2023.xlsx",
              "2024": DATA_DIR/"osbm_staff_survey_2024.xlsx",
              "2025": DATA_DIR/"osbm_staff_survey_2025.xlsx"}
series = {}
for y,pth in year_files.items():
    if not pth.exists(): continue
    d = read_excel_path(pth); col=find_q1_column(d); 
    if not col: continue
    s = pd.to_numeric(d[col], errors="coerce").dropna(); s=s[(s>=0)&(s<=10)]
    if len(s): series[y]=s
if series:
    fig_sm = make_subplots(rows=1, cols=len(series), shared_yaxes=True,
                           subplot_titles=[y for y in ["2023","2024","2025"] if y in series])
    col_idx=0
    for y in ["2023","2024","2025"]:
        if y not in series: continue
        col_idx+=1; s=series[y]
        fig_sm.add_trace(go.Histogram(x=s, histnorm="percent",
                         xbins=dict(start=-0.5,end=10.5,size=1),
                         marker_color="#1f77b4",
                         hovertemplate="%{y:.0f}%% at %{x}<extra>"+y+"</extra>",
                         showlegend=False, name=y), row=1, col=col_idx)
        fig_sm.add_vline(x=s.mean(), row=1,col=col_idx, line_width=2,line_dash="dash",line_color="#f39c12")
        fig_sm.add_vline(x=s.median(), row=1,col=col_idx, line_width=2,line_dash="dot",line_color="#7f7f7f")
    fig_sm.update_layout(bargap=0.0)
    fig_sm.update_xaxes(title_text="Happiness (0–10)", tickmode="linear", dtick=1, range=[-0.5,10.5])
    fig_sm.update_yaxes(title_text="% of responses", range=[0,50], tick0=0, dtick=10)
    st.plotly_chart(fig_sm, use_container_width=True)

# ======================================================
# Table 2 — Q1 by Section (2025)
# ======================================================
st.subheader("Table 2. On a scale of 1 to 10, how happy are you at work? — by section (2025)")
if sec_col and q1_col:
    sec_series = df[sec_col].astype(str).map(normalize_section)
    df_section = pd.DataFrame({"Section": sec_series, "Happiness": df[q1_col]})
    df_section["Happiness"] = pd.to_numeric(df_section["Happiness"], errors="coerce")
    df_section = df_section.dropna(subset=["Happiness"])
    order_buckets = ["NCPRO","Budget Execution","Budget Development","DEA or Internal Audit",
                     "Business Office/Grants/IT/Comms","Intern/Other"]
    summary = (df_section.groupby("Section")["Happiness"]
               .agg(n="count", mean="mean", median="median",
                    top=lambda x:(x>=8).mean()*100, bottom=lambda x:(x<=3).mean()*100)
               .reindex(order_buckets, fill_value=0))
    summary["mean"]=summary["mean"].round(1); summary["median"]=summary["median"].round(1)
    summary["top"]=(summary["top"]).round(0).astype(int).astype(str)+"%"
    summary["bottom"]=(summary["bottom"]).round(0).astype(int).astype(str)+"%"
    st.dataframe(summary, use_container_width=True)

# ======================================================
# Figure 3 — Q1 Distribution by Section (2 per row)
# ======================================================
st.header("Figure 3. On a scale of 1 to 10, how happy are you at work? — distribution by section (2025)")
if sec_col and q1_col:
    sec_series = df[sec_col].astype(str).map(normalize_section)
    df_sec = pd.DataFrame({"Section": sec_series, "Happiness": df[q1_col]})
    df_sec["Happiness"] = pd.to_numeric(df_sec["Happiness"], errors="coerce")
    df_sec = df_sec.dropna(subset=["Happiness"])
    order_buckets = ["NCPRO","Budget Execution","Budget Development","DEA or Internal Audit",
                     "Business Office/Grants/IT/Comms","Intern/Other"]
    present = [s for s in order_buckets if s in df_sec["Section"].unique()]
    if present:
        cols = min(2, len(present)); rows = (len(present)+cols-1)//cols
        fig_sec = make_subplots(rows=rows, cols=cols, shared_yaxes=True,
                                subplot_titles=present, horizontal_spacing=0.12, vertical_spacing=0.18)
        for idx, sect in enumerate(present, start=1):
            s = df_sec.loc[df_sec["Section"]==sect,"Happiness"]
            r=(idx-1)//cols+1; c=(idx-1)%cols+1
            fig_sec.add_trace(go.Histogram(x=s, histnorm="percent",
                                           xbins=dict(start=-0.5,end=10.5,size=1),
                                           marker_color="#1f77b4",
                                           hovertemplate="%{y:.0f}%% at %{x}<extra>"+sect+"</extra>",
                                           showlegend=False), row=r,col=c)
            fig_sec.add_vline(x=s.mean(), row=r,col=c, line_width=2,line_dash="dash",line_color="#f39c12")
            fig_sec.add_vline(x=s.median(), row=r,col=c, line_width=2,line_dash="dot",line_color="#7f7f7f")
        fig_sec.update_layout(bargap=0.05, height=max(360*rows,420), margin=dict(l=40,r=20,t=60,b=40))
        fig_sec.update_xaxes(title_text="Happiness (0–10)", tickmode="linear", dtick=1, range=[-0.5,10.5])
        fig_sec.update_yaxes(title_text="% of responses", range=[0,50], tick0=0, dtick=10)
        st.plotly_chart(fig_sec, use_container_width=True)

# ======================================================
# Q2 — Referral: Current, Trend, by Section, Comments
# ======================================================
st.header("Figure 4. Would you refer someone to work here? (2025)")
def find_q2_column(d: pd.DataFrame) -> Optional[str]:
    cols = [str(c).strip() for c in d.columns]; low=[c.lower() for c in cols]
    keys = ["refer","recommend","would you refer","would you recommend"]
    for i,c in enumerate(low):
        if any(k in c for k in keys): return cols[i]
    for i,c in enumerate(low[:12]):
        if "yes" in c or "no" in c: return cols[i]
    return None

q2_col = find_q2_column(df)
if not q2_col:
    st.info("Couldn’t locate Q2 (referral).")
else:
    q2_norm = df[q2_col].apply(normalize_yesno).dropna()
    yes_pct = (q2_norm=="Yes").mean()*100 if len(q2_norm) else 0
    fig_q2 = px.pie(pd.DataFrame({"Response":["Yes","No"],
                                  "Percent":[yes_pct, 100-yes_pct]}),
                    names="Response", values="Percent", hole=0.5,
                    title="Would you refer someone to work here? (2025)",
                    color="Response", color_discrete_map={"Yes":"#1f77b4","No":"#7f7f7f"})
    fig_q2.update_traces(textposition="outside", texttemplate="%{label}: %{value:.0f}%")
    st.plotly_chart(fig_q2, use_container_width=True)

st.header("Figure 5. Would you refer someone to work here? (by year, 2023–2025)")
rows_trend=[]
for y,pth in year_files.items():
    if not pth.exists(): continue
    d=read_excel_path(pth); col=find_q2_column(d); 
    if not col: continue
    v=d[col].apply(normalize_yesno).dropna()
    if len(v): rows_trend.append({"Year": y, "% Yes": round((v=="Yes").mean()*100,0)})
if rows_trend:
    trend_q2=pd.DataFrame(rows_trend); trend_q2["Year"]=pd.Categorical(trend_q2["Year"],["2023","2024","2025"],True)
    trend_q2=trend_q2.sort_values("Year")
    fig_tr=go.Figure(go.Scatter(x=trend_q2["Year"], y=trend_q2["% Yes"], mode="lines+markers+text",
                                line=dict(width=3,color="#1f77b4"), marker=dict(size=10),
                                text=[f"{v:.0f}%" for v in trend_q2["% Yes"]],
                                textposition="top center",
                                hovertemplate="%{y:.0f}%% in %{x}<extra></extra>"))
    fig_tr.update_layout(yaxis=dict(title="% Yes", range=[0,100], dtick=10), xaxis=dict(type="category"))
    st.plotly_chart(fig_tr, use_container_width=True)

st.subheader("Table 3. Would you refer someone to work here? — by section (2025)")
if sec_col and q2_col:
    sec_series = df[sec_col].astype(str).map(normalize_section)
    tidy = pd.DataFrame({"Section":sec_series, "Refer": df[q2_col].apply(normalize_yesno)}).dropna()
    order_buckets = ["NCPRO","Budget Execution","Budget Development","DEA or Internal Audit",
                     "Business Office/Grants/IT/Comms","Intern/Other"]
    counts = tidy["Section"].value_counts().reindex(order_buckets, fill_value=0)
    pctyes = tidy.groupby("Section")["Refer"].apply(lambda v:(v=="Yes").mean()*100).reindex(order_buckets, fill_value=0)
    out = pd.DataFrame({"Section":order_buckets,"n":counts.values,"% Yes":pctyes.round(0).astype(int).astype(str)+"%"})
    total_row=pd.DataFrame({"Section":["Total"],"n":[out["n"].sum()],"% Yes":[f"{round((tidy['Refer']=='Yes').mean()*100):.0f}%"]})
    st.dataframe(pd.concat([out,total_row], ignore_index=True), use_container_width=True)

# Comments explorer for Q2/Q3
st.header("Figure 6. If you would refer someone to work at OSBM, why? If not, why not?")
Q2_TARGET = "Would you refer someone to work here?"
Q3_TARGET = "If you would refer someone to work at OSBM, why? If not, why not?"

def find_exact_column(d: pd.DataFrame, target: str) -> Optional[str]:
    t=target.strip().lower()
    for c in d.columns:
        if str(c).strip().lower()==t: return c
    return None

q2_exact = find_exact_column(df, Q2_TARGET)
q3_exact = find_exact_column(df, Q3_TARGET)
q2_final = q2_exact or find_q2_column(df)
q3_final = q3_exact
if (q2_final is None) or (q3_final is None):
    missing=[]
    if q2_final is None: missing.append(f"`{Q2_TARGET}`")
    if q3_final is None: missing.append(f"`{Q3_TARGET}`")
    st.info("Referral comments explorer unavailable. Missing column(s): " + ", ".join(missing))
else:
    cohort = st.radio("Show comments for referral response:", ["Yes","No"], horizontal=True)
    if sec_col:
        secs = df[sec_col].astype(str).map(normalize_section)
        tidy = pd.DataFrame({"Refer": df[q2_final].apply(normalize_yesno),
                             "Section": secs,
                             "Comment": df[q3_final].astype(str).str.strip()}).dropna(subset=["Refer","Comment"])
        present = sorted([s for s in tidy["Section"].dropna().unique()])
        st.caption("Filter comments by section:"); sel = section_checkbox_selector(present, "q2q3")
        mask_sec = tidy["Section"].isin(sel) if sel else False
    else:
        tidy = pd.DataFrame({"Refer": df[q2_final].apply(normalize_yesno),
                             "Comment": df[q3_final].astype(str).str.strip()}).dropna(subset=["Refer","Comment"])
        mask_sec = True
    show = tidy[(tidy["Refer"]==cohort) & (tidy["Comment"].str.len()>0) & (mask_sec)][["Comment"]]
    st.dataframe(show, use_container_width=True, height=420)
    st.caption(f"{len(show):,} {cohort} comment(s)")

# ======================================================
# Q5/Q6 — Re-apply, Trend, by Section, Comments
# ======================================================
st.header("Figure 7. Would you re-apply for your job? (2025)")
Q5_TARGET = "Would you re-apply for your job?"
Q6_TARGETS = ["Why or why not?","Why or why not? Expanding on question 5."]

def find_exact_any(d: pd.DataFrame, targets: List[str]) -> Optional[str]:
    lows=[str(c).strip().lower() for c in d.columns]
    for t in targets:
        t_low=t.strip().lower()
        for i,low in enumerate(lows):
            if low==t_low: return d.columns[i]
    return None

def find_q5_column(d: pd.DataFrame) -> Optional[str]:
    exact=find_exact_any(d, [Q5_TARGET])
    if exact: return exact
    cols=[str(c).strip() for c in d.columns]; low=[c.lower() for c in cols]
    keys=["re-apply","reapply","re apply","would you re-apply","would you reapply"]
    for i,c in enumerate(low):
        if any(k in c for k in keys): return cols[i]
    return None

def find_q6_column(d: pd.DataFrame) -> Optional[str]:
    exact=find_exact_any(d, Q6_TARGETS)
    if exact: return exact
    cols=[str(c).strip() for c in d.columns]; low=[c.lower() for c in cols]
    q5c=find_q5_column(d)
    q5_idx=low.index(q5c.lower()) if q5c and q5c.lower() in low else None
    for i,c in enumerate(cols):
        if q5_idx is not None and i<=q5_idx: continue
        s=d[c].astype(str)
        if s.map(len).mean()>20 and s.nunique(dropna=True)>10: return c
    return None

q5_col = find_q5_column(df)
if not q5_col:
    st.info(f"Couldn’t locate `{Q5_TARGET}`.")
else:
    v = df[q5_col].apply(normalize_yesno).dropna()
    yes = (v=="Yes").mean()*100 if len(v) else 0
    fig5 = px.pie(pd.DataFrame({"Response":["Yes","No"],"Percent":[yes,100-yes]}),
                  names="Response", values="Percent", hole=0.5,
                  title="Would you re-apply for your job? (2025)",
                  color="Response", color_discrete_map={"Yes":"#1f77b4","No":"#7f7f7f"})
    fig5.update_traces(textposition="outside", texttemplate="%{label}: %{value:.0f}%")
    st.plotly_chart(fig5, use_container_width=True)

st.header("Figure 8. Would you re-apply for your job? (by year, 2023–2025)")
rows_trend5=[]
for y,pth in year_files.items():
    if not pth.exists(): continue
    d=read_excel_path(pth); col=find_q5_column(d)
    if not col: continue
    v=d[col].apply(normalize_yesno).dropna()
    if len(v): rows_trend5.append({"Year": y, "% Yes": round((v=="Yes").mean()*100,0)})
if rows_trend5:
    trend5=pd.DataFrame(rows_trend5); trend5["Year"]=pd.Categorical(trend5["Year"],["2023","2024","2025"],True)
    trend5=trend5.sort_values("Year")
    fig5t=go.Figure(go.Scatter(x=trend5["Year"], y=trend5["% Yes"], mode="lines+markers+text",
                               line=dict(width=3,color="#1f77b4"), marker=dict(size=10),
                               text=[f"{v:.0f}%" for v in trend5["% Yes"]],
                               textposition="top center",
                               hovertemplate="%{y:.0f}%% in %{x}<extra></extra>"))
    fig5t.update_layout(yaxis=dict(title="% Yes", range=[0,100], dtick=10), xaxis=dict(type="category"))
    st.plotly_chart(fig5t, use_container_width=True)

st.subheader("Table 5. Would you re-apply for your job? — by section (2025)")
if sec_col and q5_col:
    sec_series = df[sec_col].astype(str).map(normalize_section)
    tidy5 = pd.DataFrame({"Section": sec_series, "Reapply": df[q5_col].apply(normalize_yesno)}).dropna()
    order_buckets = ["NCPRO","Budget Execution","Budget Development","DEA or Internal Audit",
                     "Business Office/Grants/IT/Comms","Intern/Other"]
    counts = tidy5["Section"].value_counts().reindex(order_buckets, fill_value=0)
    pctyes = tidy5.groupby("Section")["Reapply"].apply(lambda v:(v=="Yes").mean()*100).reindex(order_buckets, fill_value=0)
    out = pd.DataFrame({"Section":order_buckets,"n":counts.values,"% Yes":pctyes.round(0).astype(int).astype(str)+"%"})
    total_row = pd.DataFrame({"Section":["Total"],"n":[out["n"].sum()], "% Yes":[f"{round((tidy5['Reapply']=='Yes').mean()*100):.0f}%"]})
    st.dataframe(pd.concat([out,total_row], ignore_index=True), use_container_width=True)

# Comments explorer for Q5/Q6
st.header("Figure 9. Why or why not? (conditional on re-apply response, 2025)")
q6_col = find_q6_column(df)
if (q5_col is None) or (q6_col is None):
    missing=[]
    if q5_col is None: missing.append(f"`{Q5_TARGET}`")
    if q6_col is None: missing.append("`Why or why not?`")
    st.info("Re-apply comments explorer unavailable. Missing: " + ", ".join(missing))
else:
    choice5 = st.radio("Show comments for re-apply response:", ["Yes","No"], horizontal=True)
    if sec_col:
        sec_series = df[sec_col].astype(str).map(normalize_section)
        tidy6 = pd.DataFrame({"Reapply": df[q5_col].apply(normalize_yesno),
                              "Section": sec_series,
                              "Comment": df[q6_col].astype(str).str.strip()}).dropna(subset=["Reapply","Comment"])
        present_sections6 = sorted([s for s in tidy6["Section"].dropna().unique()])
        st.caption("Filter comments by section:"); sel6=section_checkbox_selector(present_sections6, "q5q6")
        mask_sec6 = tidy6["Section"].isin(sel6) if sel6 else False
    else:
        tidy6 = pd.DataFrame({"Reapply": df[q5_col].apply(normalize_yesno),
                              "Comment": df[q6_col].astype(str).str.strip()}).dropna(subset=["Reapply","Comment"])
        mask_sec6 = True
    show6 = tidy6[(tidy6["Reapply"]==choice5) & (tidy6["Comment"].str.len()>0) & (mask_sec6)][["Comment"]]
    st.dataframe(show6, use_container_width=True, height=420)
    st.caption(f"{len(show6):,} {choice5} comment(s)")

# ======================================================
# Figure 10 — Q4 Likert cluster (diverging) with selectors, sorted by Top-box
# ======================================================
st.header("Figure 10. Please provide your sentiment towards the statements below")
# Detect Q4 statements by keywords
Q4_ITEMS = [
    (("clear","career"), "I have a clear understanding of my career or promotion path."),
    (("valued",), "I feel valued at work."),
    (("full","potential"), "I'll be able to reach my full potential at OSBM."),
    (("reapply","current"), "If given the chance, I would reapply to my current job."),
    (("life","work","harmony"), "I am able to maintain life–work harmony."),
]
cols_lower = [str(c).strip() for c in df.columns]; cols_lower_lc=[c.lower() for c in cols_lower]
def find_col_by_keywords(keys: Tuple[str,...]) -> Optional[str]:
    for i,c in enumerate(cols_lower_lc):
        if all(k in c for k in keys): return cols_lower[i]
    return None
found=[]
for keys,label in Q4_ITEMS:
    col=find_col_by_keywords(keys)
    if col: found.append((label,col))

if not found:
    st.info("Couldn’t locate Q4 statements.")
else:
    # Section filter
    if sec_col:
        sec_norm = df[sec_col].astype(str).map(normalize_section)
        present_sections4 = sorted([s for s in sec_norm.dropna().unique()])
        st.caption("Filter statements by section:"); sel_secs4=section_checkbox_selector(present_sections4,"q4_sections")
        mask4 = sec_norm.isin(sel_secs4) if sel_secs4 else pd.Series(False,index=df.index)
    else:
        mask4 = pd.Series(True, index=df.index)
    df_q4 = df.loc[mask4].copy()
    st.caption(f"Current section filter (Q4): {', '.join(sel_secs4) if sec_col else 'All sections'} — N = {len(df_q4)}")

    # Cluster toggles
    st.caption("Select which rating clusters to display in the chart:"); btm,mid,top = cluster_toggles("q4")
    neg_sel, pos_sel = [], []
    if btm: neg_sel += ["Strongly Disagree","Disagree"]
    if mid: neg_sel += ["Slightly Disagree"]; pos_sel += ["Slightly Agree"]
    if top: pos_sel += ["Agree","Strongly Agree"]
    neg_order=[c for c in ["Slightly Disagree","Disagree","Strongly Disagree"] if c in neg_sel]
    pos_order=[c for c in ["Slightly Agree","Agree","Strongly Agree"] if c in pos_sel]
    show_cats = neg_order + pos_order

    # Displayed N across selected clusters
    if show_cats:
        mask_disp = pd.Series(False,index=df_q4.index)
        for _,col in found: mask_disp |= df_q4[col].apply(normalize_likert).isin(show_cats)
        dispN = int(mask_disp.sum())
    else:
        dispN = 0

    # Order items by Top-box on section-filtered data
    scored=[]
    for label,col in found:
        s_all=df_q4[col].apply(normalize_likert).dropna()
        tb=(s_all.isin(["Agree","Strongly Agree"]).mean()*100) if len(s_all) else 0
        scored.append((tb,label,col))
    scored.sort(reverse=True)

    # Build percentages for chart in sorted order
    item_labels=[]; pct_by_cat={cat:[] for cat in LIKERT_ORDER}
    for tb,label,col in scored:
        s=df_q4[col].apply(normalize_likert).dropna()
        counts=(s.value_counts(normalize=True)*100)
        item_labels.append(label)
        for cat in LIKERT_ORDER: pct_by_cat[cat].append(round(float(counts.get(cat,0.0)),0))

    # Chart
    color_map = {"Strongly Disagree":"%s"% "#D55E00","Disagree":"#E69F00","Slightly Disagree":"#F0E442",
                 "Slightly Agree":"#56B4E9","Agree":"#0072B2","Strongly Agree":"#004B87"}
    fig = go.Figure()
    for cat in neg_order:
        fig.add_trace(go.Bar(x=[-v for v in pct_by_cat[cat]], y=item_labels, name=cat, orientation="h",
                             marker_color=color_map[cat],
                             hovertemplate=f"{cat}: "+"%{customdata:.0f}%<extra></extra>",
                             customdata=pct_by_cat[cat]))
    for cat in pos_order:
        fig.add_trace(go.Bar(x=pct_by_cat[cat], y=item_labels, name=cat, orientation="h",
                             marker_color=color_map[cat],
                             hovertemplate=f"{cat}: "+"%{customdata:.0f}%<extra></extra>",
                             customdata=pct_by_cat[cat]))
    fig.update_layout(barmode="relative",
                      xaxis=dict(title="% of responses", range=[-100,100], tickmode="array",
                                 tickvals=[-100,-90,-80,-70,-60,-50,-40,-30,-20,-10,0,10,20,30,40,50,60,70,80,90,100],
                                 ticktext=[f"{abs(v)}%" for v in [-100,-90,-80,-70,-60,-50,-40,-30,-20,-10,0,10,20,30,40,50,60,70,80,90,100]],
                                 zeroline=True, zerolinewidth=2, zerolinecolor="#888"),
                      yaxis=dict(autorange="reversed", tickfont=dict(size=16,color="black"), automargin=True),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, traceorder="normal"),
                      margin=dict(l=40,r=20,t=60,b=40),
                      height=440 + 40*max(0,len(item_labels)-4))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Displayed N (based on current selections) = {dispN}.")

    # Table (always show all 6 categories; counts based on selected clusters)
    rows=[]; 
    for tb,label,col in scored:
        s=df_q4[col].apply(normalize_likert).dropna()
        s_sel = s[s.isin(show_cats)] if show_cats else s.iloc[0:0]
        total=len(s_sel)
        row={"Statement":label}
        counts=s_sel.value_counts()
        for cat in LIKERT_ORDER:
            n=int(counts.get(cat,0)); pct=(n/total*100) if total else 0
            row[cat]=f"{n} ({pct:.0f}%)"
        rows.append(row)
    st.dataframe(pd.DataFrame(rows, columns=["Statement"]+LIKERT_ORDER), use_container_width=True)
    if not show_cats:
        st.info("Select at least one rating cluster to display the chart.")
        st.caption("Displayed N (based on current selections) = 0.")

# ======================================================
# Figure 11 — Q10 Likert cluster (sorted by Top-box), same selectors
# ======================================================
st.header("Figure 11. Organization culture statements — diverging stacked (2025)")
Q10_ITEMS = [
    (("coworkers","respect"), "My OSBM coworkers respect each other."),
    (("feel","respected"), "I feel respected as an employee at OSBM."),
    (("management","positive","culture"), "The OSBM Management Team contributes to a positive work culture."),
    (("mission","vision"), "I know what OSBM’s mission and vision are."),
]
cols_lower=[str(c).strip() for c in df.columns]; cols_lower_lc=[c.lower() for c in cols_lower]
def find_q10_col_by_keywords(keys: Tuple[str,...]) -> Optional[str]:
    for i,c in enumerate(cols_lower_lc):
        if all(k in c for k in keys): return cols_lower[i]
    return None
found10=[]
for keys,label in Q10_ITEMS:
    col=find_q10_col_by_keywords(keys)
    if col: found10.append((label,col))

if not found10:
    st.info("Couldn’t locate the Q10 organization-culture statements in the dataset.")
else:
    if sec_col:
        sec_norm10=df[sec_col].astype(str).map(normalize_section)
        present10=sorted([s for s in sec_norm10.dropna().unique()])
        st.caption("Filter statements by section:"); sel10=section_checkbox_selector(present10,"q10_sections")
        mask10=sec_norm10.isin(sel10) if sel10 else pd.Series(False,index=df.index)
    else:
        mask10=pd.Series(True,index=df.index)
    df_q10=df.loc[mask10].copy()
    st.caption(f"Current section filter (Q10): {', '.join(sel10) if sec_col else 'All sections'} — N = {len(df_q10)}")

    st.caption("Select which rating clusters to display in the chart:"); btm,mid,top=cluster_toggles("q10")
    neg_sel, pos_sel=[], []
    if btm: neg_sel+=["Strongly Disagree","Disagree"]
    if mid: neg_sel+=["Slightly Disagree"]; pos_sel+=["Slightly Agree"]
    if top: pos_sel+=["Agree","Strongly Agree"]
    neg_order=[c for c in ["Slightly Disagree","Disagree","Strongly Disagree"] if c in neg_sel]
    pos_order=[c for c in ["Slightly Agree","Agree","Strongly Agree"] if c in pos_sel]
    show_cats = neg_order + pos_order

    if show_cats:
        mask_disp=pd.Series(False,index=df_q10.index)
        for _,col in found10: mask_disp |= df_q10[col].apply(normalize_likert).isin(show_cats)
        dispN=int(mask_disp.sum())
    else:
        dispN=0

    # Sort by Top-box on section-filtered data
    scored10=[]
    for label,col in found10:
        s_all=df_q10[col].apply(normalize_likert).dropna()
        tb=(s_all.isin(["Agree","Strongly Agree"]).mean()*100) if len(s_all) else 0
        scored10.append((tb,label,col))
    scored10.sort(reverse=True)

    # Build percentages for chart in sorted order
    item_labels10=[]; pct_by_cat10={cat:[] for cat in LIKERT_ORDER}
    for tb,label,col in scored10:
        s=df_q10[col].apply(normalize_likert).dropna()
        counts=(s.value_counts(normalize=True)*100)
        item_labels10.append(label)
        for cat in LIKERT_ORDER: pct_by_cat10[cat].append(round(float(counts.get(cat,0.0)),0))

    # Chart
    color_map={"Strongly Disagree":"#D55E00","Disagree":"#E69F00","Slightly Disagree":"#F0E442",
               "Slightly Agree":"#56B4E9","Agree":"#0072B2","Strongly Agree":"#004B87"}
    fig10=go.Figure()
    for cat in neg_order:
        fig10.add_trace(go.Bar(x=[-v for v in pct_by_cat10[cat]], y=item_labels10, name=cat, orientation="h",
                               marker_color=color_map[cat],
                               hovertemplate=f"{cat}: "+"%{customdata:.0f}%<extra></extra>",
                               customdata=pct_by_cat10[cat]))
    for cat in pos_order:
        fig10.add_trace(go.Bar(x=pct_by_cat10[cat], y=item_labels10, name=cat, orientation="h",
                               marker_color=color_map[cat],
                               hovertemplate=f"{cat}: "+"%{customdata:.0f}%<extra></extra>",
                               customdata=pct_by_cat10[cat]))
    fig10.update_layout(barmode="relative",
                        xaxis=dict(title="% of responses", range=[-100,100], tickmode="array",
                                   tickvals=[-100,-90,-80,-70,-60,-50,-40,-30,-20,-10,0,10,20,30,40,50,60,70,80,90,100],
                                   ticktext=[f"{abs(v)}%" for v in [-100,-90,-80,-70,-60,-50,-40,-30,-20,-10,0,10,20,30,40,50,60,70,80,90,100]],
                                   zeroline=True, zerolinewidth=2, zerolinecolor="#888"),
                        yaxis=dict(autorange="reversed", tickfont=dict(size=16,color="black"), automargin=True),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, traceorder="normal"),
                        margin=dict(l=40,r=20,t=60,b=40),
                        height=440 + 40*max(0,len(item_labels10)-4))
    st.plotly_chart(fig10, use_container_width=True)
    st.caption(f"Displayed N (based on current selections) = {dispN}.")

    # Table
    rows=[]
    for tb,label,col in scored10:
        s=df_q10[col].apply(normalize_likert).dropna()
        s_sel=s[s.isin(show_cats)] if show_cats else s.iloc[0:0]
        total=len(s_sel); row={"Statement":label}
        counts=s_sel.value_counts()
        for cat in LIKERT_ORDER:
            n=int(counts.get(cat,0)); pct=(n/total*100) if total else 0
            row[cat]=f"{n} ({pct:.0f}%)"
        rows.append(row)
    st.dataframe(pd.DataFrame(rows, columns=["Statement"]+LIKERT_ORDER), use_container_width=True)
    if not show_cats:
        st.info("Select at least one rating cluster to display the chart.")
        st.caption("Displayed N (based on current selections) = 0.")

# ======================================================
# Figure 12 — Combined Likert: Q4 + Q10 + Q21
# ======================================================
st.header("Figure 12. Sentiment toward selected statements (Q4, Q10, Q21)")

def find_q21_column(d: pd.DataFrame, exclude: List[str]) -> Optional[Tuple[str,str]]:
    cols=[str(c).strip() for c in d.columns]; low=[c.lower() for c in cols]
    target="i know what is going on across the office"
    for i,c in enumerate(low):
        if c==target: return (cols[i], cols[i])
    kwsets=[("know","going on","office"), ("informed","office"), ("across","office")]
    for i,c in enumerate(low):
        if cols[i] in exclude: continue
        for kws in kwsets:
            if all(k in c for k in kws): return (cols[i], cols[i])
    best=None; best_count=0
    for i,c in enumerate(cols):
        if cols[i] in exclude: continue
        s=d[cols[i]].apply(normalize_likert).dropna()
        if len(s) and (len(s)/max(1,d[cols[i]].notna().sum()))>=0.6:
            if len(s)>best_count: best, best_count = cols[i], len(s)
    if best: return (best,best)
    return None

# Gather items from Q4/Q10 detections plus Q21
items_all=[]
if 'found' in locals() and found: items_all+=found
if 'found10' in locals() and found10: items_all+=found10
exclude_cols=[col for _,col in items_all]
q21=find_q21_column(df, exclude_cols)
if q21 is not None: items_all.append((q21[0], q21[1]))

if not items_all:
    st.info("Couldn’t locate Q4/Q10/Q21 Likert statements to combine.")
else:
    # Section filter
    if sec_col:
        secsC=df[sec_col].astype(str).map(normalize_section)
        presentC=sorted([s for s in secsC.dropna().unique()])
        st.caption("Filter statements by section:"); selC=section_checkbox_selector(presentC,"combo_sections")
        maskC=secsC.isin(selC) if selC else pd.Series(False,index=df.index)
    else:
        maskC=pd.Series(True,index=df.index)
    df_combo=df.loc[maskC].copy()

    # Cluster toggles
    st.caption("Select which rating clusters to display in the chart:"); btm,mid,top=cluster_toggles("combo")
    neg_sel,pos_sel=[],[]
    if btm: neg_sel+=["Strongly Disagree","Disagree"]
    if mid: neg_sel+=["Slightly Disagree"]; pos_sel+=["Slightly Agree"]
    if top: pos_sel+=["Agree","Strongly Agree"]
    neg_order=[c for c in ["Slightly Disagree","Disagree","Strongly Disagree"] if c in neg_sel]
    pos_order=[c for c in ["Slightly Agree","Agree","Strongly Agree"] if c in pos_sel]
    show_cats=neg_order+pos_order

    # Displayed N
    if show_cats:
        mask_disp=pd.Series(False,index=df_combo.index)
        for _,col in items_all: mask_disp|=df_combo[col].apply(normalize_likert).isin(show_cats)
        dispN=int(mask_disp.sum())
    else:
        dispN=0

    # Order by Top-box (A+SA) on section-filtered data
    scoredC=[]
    for label,col in items_all:
        s_all=df_combo[col].apply(normalize_likert).dropna()
        tb=(s_all.isin(["Agree","Strongly Agree"]).mean()*100) if len(s_all) else 0
        scoredC.append((tb,label,col))
    scoredC.sort(reverse=True)

    # Build chart arrays
    item_labelsC=[]; pct_by_catC={cat:[] for cat in LIKERT_ORDER}
    for tb,label,col in scoredC:
        s=df_combo[col].apply(normalize_likert).dropna()
        counts=(s.value_counts(normalize=True)*100)
        item_labelsC.append(label)
        for cat in LIKERT_ORDER: pct_by_catC[cat].append(round(float(counts.get(cat,0.0)),0))

    color_map={"Strongly Disagree":"#D55E00","Disagree":"#E69F00","Slightly Disagree":"#F0E442",
               "Slightly Agree":"#56B4E9","Agree":"#0072B2","Strongly Agree":"#004B87"}
    figC=go.Figure()
    for cat in neg_order:
        figC.add_trace(go.Bar(x=[-v for v in pct_by_catC[cat]], y=item_labelsC, name=cat, orientation="h",
                              marker_color=color_map[cat],
                              hovertemplate=f"{cat}: "+"%{customdata:.0f}%<extra></extra>",
                              customdata=pct_by_catC[cat]))
    for cat in pos_order:
        figC.add_trace(go.Bar(x=pct_by_catC[cat], y=item_labelsC, name=cat, orientation="h",
                              marker_color=color_map[cat],
                              hovertemplate=f"{cat}: "+"%{customdata:.0f}%<extra></extra>",
                              customdata=pct_by_catC[cat]))
    figC.update_layout(barmode="relative",
                       xaxis=dict(title="% of responses", range=[-100,100], tickmode="array",
                                  tickvals=[-100,-90,-80,-70,-60,-50,-40,-30,-20,-10,0,10,20,30,40,50,60,70,80,90,100],
                                  ticktext=[f"{abs(v)}%" for v in [-100,-90,-80,-70,-60,-50,-40,-30,-20,-10,0,10,20,30,40,50,60,70,80,90,100]],
                                  zeroline=True, zerolinewidth=2, zerolinecolor="#888"),
                       yaxis=dict(autorange="reversed", tickfont=dict(size=16,color="black"), automargin=True),
                       legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, traceorder="normal"),
                       margin=dict(l=40,r=20,t=60,b=40),
                       height=440 + 40*max(0,len(item_labelsC)-4))
    st.plotly_chart(figC, use_container_width=True)
    st.caption(f"Displayed N (based on current selections) = {dispN}.")

    # Combined table beneath chart
    rows=[]
    for tb,label,col in scoredC:
        s=df_combo[col].apply(normalize_likert).dropna()
        s_sel=s[s.isin(show_cats)] if show_cats else s.iloc[0:0]
        total=len(s_sel); row={"Statement":label}
        counts=s_sel.value_counts()
        for cat in LIKERT_ORDER:
            n=int(counts.get(cat,0)); pct=(n/total*100) if total else 0
            row[cat]=f"{n} ({pct:.0f}%)"
        rows.append(row)
    st.dataframe(pd.DataFrame(rows, columns=["Statement"]+LIKERT_ORDER), use_container_width=True)
    st.caption("Counts/percentages reflect current section & cluster selections. Per-statement N may vary due to missing answers.")

    # ======================================================
# Figure 13 — Q12 Recognition frequency (with trend)
# ======================================================
st.header("Figure 13. How often do you receive recognition for your work?")

def find_q12_column(d: pd.DataFrame) -> Optional[str]:
    cols = [str(c).strip() for c in d.columns]
    low  = [c.lower() for c in cols]
    # Prefer exact phrasing per instrument
    target = "how frequently do you receive recognition from your manager?"
    for i, c in enumerate(low):
        if c == target:
            return cols[i]
    # Heuristics
    for i, c in enumerate(low):
        if "recognition" in c and ("manager" in c or "how frequently" in c or "how often" in c):
            return cols[i]
    return None

q12_col = find_q12_column(df)
if not q12_col:
    st.info("Couldn’t locate Q12 (recognition frequency).")
else:
    tidy12 = df[q12_col].dropna().astype(str).str.strip()
    counts12 = tidy12.value_counts(normalize=True) * 100
    # Reorder: Never, Daily, Weekly, Monthly
    order_rec = ["Never", "Daily", "Weekly", "Monthly"]
    counts12 = counts12.reindex(order_rec, fill_value=0)
    fig12 = px.bar(x=counts12.index, y=counts12.values,
                   labels={'x':"Response",'y':'% of respondents'},
                   title="Recognition frequency (2025)")
    fig12.update_traces(texttemplate="%{y:.0f}%", textposition="outside")
    fig12.update_yaxes(range=[0,75], tick0=0, dtick=10)
    st.plotly_chart(fig12, use_container_width=True)

    # Trend across years
    rows_trend12 = []
    for y, pth in year_files.items():
        if not pth.exists(): continue
        d = read_excel_path(pth); col = find_q12_column(d)
        if not col: continue
        v = d[col].dropna().astype(str)
        if len(v):
            v_low = v.str.lower()
            pct_mo = v_low.str.contains("daily|weekly|month", regex=True).mean() * 100
            rows_trend12.append({"Year": y, "% Monthly or More": round(pct_mo, 0)})
    if rows_trend12:
        trend12 = pd.DataFrame(rows_trend12)
        trend12["Year"] = pd.Categorical(trend12["Year"], ["2023","2024","2025"], ordered=True)
        trend12 = trend12.sort_values("Year")
        fig12t = go.Figure(go.Scatter(x=trend12["Year"], y=trend12["% Monthly or More"], mode="lines+markers+text",
                                      line=dict(width=3,color="#1f77b4"), marker=dict(size=10),
                                      text=[f"{v:.0f}%" for v in trend12["% Monthly or More"]],
                                      textposition="top center"))
        fig12t.update_layout(yaxis=dict(title="% recognizing at least monthly", range=[0,100], dtick=10),
                             xaxis=dict(type="category"))
        st.plotly_chart(fig12t, use_container_width=True)

# ======================================================
# Figure 14 — Q15 Work–life harmony (Yes/No with trend)
# ======================================================
st.header("Figure 14. Are you able to maintain work–life harmony?")

def find_q15_column(d: pd.DataFrame) -> Optional[str]:
    cols = [str(c).strip() for c in d.columns]
    low = [c.lower() for c in cols]
    # Prefer exact phrasing variants from the instrument
    targets = [
        "are you able to maintain life - work harmony?",
        "are you able to maintain life–work harmony?",
        "are you able to maintain life-work harmony?"
    ]
    for i, c in enumerate(low):
        if c in targets:
            return cols[i]
    # Heuristic fallback
    for i, c in enumerate(low):
        if "life" in c and "work" in c and "harmony" in c:
            return cols[i]
    return None

q15_col = find_q15_column(df)
if not q15_col:
    st.info("Couldn’t locate Q15 (work–life harmony).")
else:
    # --- 2025 single donut ---
    v15_2025 = df[q15_col].apply(normalize_yesno).dropna()
    yes15_2025 = (v15_2025 == "Yes").mean() * 100 if len(v15_2025) else 0
    fig15 = px.pie(
        pd.DataFrame({"Response": ["Yes", "No"], "Percent": [yes15_2025, 100 - yes15_2025]}),
        names="Response",
        values="Percent",
        hole=0.5,
        title="Work–life harmony (2025)",
        color="Response",
        color_discrete_map={"Yes": "#1f77b4", "No": "#7f7f7f"},
    )
    fig15.update_traces(textposition="outside", texttemplate="%{label}: %{value:.0f}%")
    st.plotly_chart(fig15, use_container_width=True)

    # --- Small-multiples donuts by year (2023–2025), left-to-right ---
    years_panel = [y for y in ["2023", "2024", "2025"] if year_files.get(y) and year_files[y].exists()]
    if years_panel:
        cols = len(years_panel)
        fig15sm = make_subplots(
            rows=1,
            cols=cols,
            specs=[[{"type": "domain"}] * cols],
            subplot_titles=years_panel,
        )
        for idx, y in enumerate(years_panel, start=1):
            d_y = read_excel_path(year_files[y])
            col_y = find_q15_column(d_y)
            if not col_y:
                continue
            v_y = d_y[col_y].apply(normalize_yesno).dropna()
            yes_y = (v_y == "Yes").mean() * 100 if len(v_y) else 0
            no_y = 100 - yes_y
            fig15sm.add_trace(
                go.Pie(
                    labels=["Yes", "No"],
                    values=[yes_y, no_y],
                    hole=0.5,
                    marker_colors=["#1f77b4", "#7f7f7f"],
                    showlegend=False,
                    textinfo="label+percent",
                    textposition="outside",
                ),
                row=1,
                col=idx,
            )
        fig15sm.update_layout(
            title_text="Work–life harmony (Yes/No) by year",
            margin=dict(l=40, r=20, t=60, b=40),
            height=420,
        )
        st.plotly_chart(fig15sm, use_container_width=True)

    # Trend
    rows_trend15 = []
    for y, pth in year_files.items():
        if not pth.exists():
            continue
        d = read_excel_path(pth)
        col = find_q15_column(d)
        if not col:
            continue
        v = d[col].apply(normalize_yesno).dropna()
        if len(v):
            rows_trend15.append({"Year": y, "% Yes": round((v == "Yes").mean() * 100, 0)})
    if rows_trend15:
        trend15 = pd.DataFrame(rows_trend15)
        trend15["Year"] = pd.Categorical(trend15["Year"], ["2023", "2024", "2025"], ordered=True)
        trend15 = trend15.sort_values("Year")
        fig15t = go.Figure(
            go.Scatter(
                x=trend15["Year"],
                y=trend15["% Yes"],
                mode="lines+markers+text",
                line=dict(width=3, color="#1f77b4"),
                marker=dict(size=10),
                text=[f"{v:.0f}%" for v in trend15["% Yes"]],
                textposition="top center",
            )
        )
        fig15t.update_layout(yaxis=dict(title="% Yes", range=[0, 100], dtick=10), xaxis=dict(type="category"))
        st.plotly_chart(fig15t, use_container_width=True)

# ======================================================
# Figure 15 — Q17 Staff meetings (attendance Yes/No, trend)
# ======================================================
st.header("Figure 15. Do you attend all-staff meetings?")

def find_q17_column(d: pd.DataFrame) -> Optional[str]:
    cols = [str(c).strip() for c in d.columns]
    low  = [c.lower() for c in cols]
    target = "do you think the all-hands staff meetings are valuable?"
    for i, c in enumerate(low):
        if c == target:
            return cols[i]
    for i, c in enumerate(low):
        if ("all-hands" in c or "staff meeting" in c) and ("valuable" in c or "value" in c):
            return cols[i]
    return None

q17_col = find_q17_column(df)
if not q17_col:
    st.info("Couldn’t locate Q17 (attendance).")
else:
    v17 = df[q17_col].apply(normalize_yesno).dropna()
    yes17 = (v17=="Yes").mean()*100 if len(v17) else 0
    fig17 = px.pie(pd.DataFrame({"Response":["Yes","No"],"Percent":[yes17,100-yes17]}),
                   names="Response", values="Percent", hole=0.5,
                   title="Staff meeting attendance (2025)",
                   color="Response", color_discrete_map={"Yes":"#1f77b4","No":"#7f7f7f"})
    fig17.update_traces(textposition="outside", texttemplate="%{label}: %{value:.0f}%")
    st.plotly_chart(fig17, use_container_width=True)

    # Trend
    rows_trend17 = []
    for y, pth in year_files.items():
        if not pth.exists(): continue
        d = read_excel_path(pth); col = find_q17_column(d)
        if not col: continue
        v = d[col].apply(normalize_yesno).dropna()
        if len(v):
            rows_trend17.append({"Year": y, "% Yes": round((v=="Yes").mean()*100, 0)})
    if rows_trend17:
        trend17 = pd.DataFrame(rows_trend17)
        trend17["Year"] = pd.Categorical(trend17["Year"], ["2023","2024","2025"], ordered=True)
        trend17 = trend17.sort_values("Year")
        fig17t = go.Figure(go.Scatter(x=trend17["Year"], y=trend17["% Yes"], mode="lines+markers+text",
                                      line=dict(width=3,color="#1f77b4"), marker=dict(size=10),
                                      text=[f"{v:.0f}%" for v in trend17["% Yes"]],
                                      textposition="top center"))
        fig17t.update_layout(yaxis=dict(title="% Yes", range=[0,100], dtick=10), xaxis=dict(type="category"))
        st.plotly_chart(fig17t, use_container_width=True)

# ======================================================
# Figure 16 — Q18 Staff meetings valuable? (Likert with trend)
# ======================================================
st.header("Figure 16. How valuable are staff meetings?")

def find_q18_column(d: pd.DataFrame) -> Optional[str]:
    cols = [str(c).strip() for c in d.columns]
    low  = [c.lower() for c in cols]
    target = "how often should we have all-hands staff meetings?"
    for i, c in enumerate(low):
        if c == target:
            return cols[i]
    for i, c in enumerate(low):
        if ("how often" in c or "frequency" in c) and ("all-hands" in c or "staff meeting" in c):
            return cols[i]
    return None

q18_col = find_q18_column(df)
if not q18_col:
    st.info("Couldn’t locate Q18 (value of staff meetings).")
else:
    v18 = df[q18_col].dropna().astype(str).str.strip()
    # Show 2025 distribution across options
    order_opts = ["We should not have them","Bi-weekly","Monthly","Quarterly"]
    counts18 = (v18.value_counts(normalize=True) * 100).reindex(order_opts, fill_value=0)
    fig18 = px.bar(x=counts18.index, y=counts18.values,
                   labels={"x":"Preference","y":"% of respondents"},
                   title="Preferred frequency of all-hands meetings (2025)")
    fig18.update_traces(texttemplate="%{y:.0f}%", textposition="outside")
    fig18.update_yaxes(range=[0,75], tick0=0, dtick=10)
    st.plotly_chart(fig18, use_container_width=True)

    # Second figure: 2023 vs 2024 distributions side-by-side
    comp_years = ["2023","2024"]
    data_comp = []
    for y in comp_years:
        pth = year_files.get(y)
        if not pth or not pth.exists(): continue
        d = read_excel_path(pth); col = find_q18_column(d)
        if not col: continue
        v = d[col].dropna().astype(str).str.strip()
        counts = (v.value_counts(normalize=True)*100).reindex(order_opts, fill_value=0)
        data_comp.append((y, counts))
    if data_comp:
        fig18c = make_subplots(rows=1, cols=len(data_comp), shared_yaxes=True,
                               subplot_titles=[y for y,_ in data_comp])
        for idx,(y,counts) in enumerate(data_comp, start=1):
            fig18c.add_trace(go.Bar(x=counts.index, y=counts.values, name=y,
                                    hovertemplate="%{y:.0f}%% — %{x}<extra>"+y+"</extra>"),
                             row=1, col=idx)
        fig18c.update_yaxes(title_text="% of respondents", range=[0,75], tick0=0, dtick=10)
        fig18c.update_layout(title="Preferred frequency of all-hands meetings (2023 vs 2024)",
                             barmode="group", margin=dict(l=40,r=20,t=60,b=40))
        st.plotly_chart(fig18c, use_container_width=True)


    # Trend: percent choosing 'Monthly'
    rows_trend18 = []
    for y, pth in year_files.items():
        if not pth.exists(): continue
        d = read_excel_path(pth); col = find_q18_column(d)
        if not col: continue
        v = d[col].dropna().astype(str)
        if len(v):
            pct_monthly = (v.str.contains("Monthly", case=False)).mean() * 100
            rows_trend18.append({"Year": y, "% Monthly": round(pct_monthly, 0)})
    if rows_trend18:
        trend18 = pd.DataFrame(rows_trend18)
        trend18["Year"] = pd.Categorical(trend18["Year"], ["2023","2024","2025"], ordered=True)
        trend18 = trend18.sort_values("Year")
        fig18t = go.Figure(go.Scatter(x=trend18["Year"], y=trend18["% Monthly"], mode="lines+markers+text",
                                      line=dict(width=3,color="#1f77b4"), marker=dict(size=10),
                                      text=[f"{v:.0f}%" for v in trend18["% Monthly"]],
                                      textposition="top center"))
        fig18t.update_layout(yaxis=dict(title="% Monthly", range=[0,100], dtick=10), xaxis=dict(type="category"))
        st.plotly_chart(fig18t, use_container_width=True)

    # --- Panel: 2023, 2024, 2025 distributions side-by-side ---
    panel_years = [y for y in ["2023","2024","2025"] if year_files.get(y) and year_files[y].exists()]
    data_panel = []
    for y in panel_years:
        dpy = read_excel_path(year_files[y])
        coly = find_q18_column(dpy)
        if not coly:
            continue
        vy = dpy[coly].dropna().astype(str).str.strip()
        cnts = (vy.value_counts(normalize=True)*100).reindex(order_opts, fill_value=0)
        data_panel.append((y, cnts))
    if data_panel:
        fig18panel = make_subplots(rows=1, cols=len(data_panel), shared_yaxes=True,
                                   subplot_titles=[y for y,_ in data_panel])
        for idx,(y,cnts) in enumerate(data_panel, start=1):
            fig18panel.add_trace(go.Bar(x=cnts.index, y=cnts.values, name=y,
                                        hovertemplate="%{y:.0f}%% — %{x}<extra>"+y+"</extra>"),
                                 row=1, col=idx)
        fig18panel.update_yaxes(title_text="% of respondents", range=[0,75], tick0=0, dtick=10)
        fig18panel.update_layout(title="Preferred frequency of all-hands meetings (2023–2025)",
                                 margin=dict(l=40,r=20,t=60,b=40))
        st.plotly_chart(fig18panel, use_container_width=True)

# ======================================================
# Figure 17 — Open-text Comments Explorer
# ======================================================
st.header("Figure 17. Open-text Comments Explorer")

# Map from friendly name -> detector for column
COMMENTS_MAP: Dict[str, Tuple[str, List[str]]] = {
    "Q8 — What would make you more likely to stay at OSBM long-term?":
        ("what would make you more likely to stay at osbm long-term", ["stay","long-term","likely"]),
    "Q9 — One thing OSBM could do to improve your work experience":
        ("what is one thing osbm could do to improve your work experience", ["improve","work experience"]),
    "Q11 — Three words to describe our work culture":
        ("what three words would you use to describe our work culture", ["three words","culture"]),
    "Q19 — Suggestions to improve all-hands staff meetings":
        ("do you have any suggestions to improve all-hands staff meetings", ["suggestions","all-hands"]),
    "Q20 — Improve design/management of meetings you attend":
        ("do you have any recommendations for how to improve the design and management of meetings", ["design","management","meetings"]),
    "Q23 — Suggestions to improve communications":
        ("do you have suggestions on how to improve communications", ["improve","communications"]),
    "Q28 — Change anything about events you have participated in":
        ("for the events that you have participated in, would you change anything about that activity", ["events","change anything"]),
    "Q29 — What appreciation activities would you be interested in":
        ("what appreciation activities would you be interested in", ["appreciation","activities"]),
    "Q31 — What could OSBM do to improve your sense of belonging":
        ("what could osbm do to improve your sense belonging in the workplace", ["belonging","workplace"]),
    "Q33 — Comments to increase satisfaction with tools":
        ("do you have any comments or recommendations about what would increase your satisfaction with the tools you use for work", ["satisfaction","tools"]),
    "Q34 — Comments about calendar and telework flexibilities":
        ("any comments about calender and telework flexibilities", ["calendar","telework"]),
}

def find_open_text_column(d: pd.DataFrame, target: str, keywords: List[str]) -> Optional[str]:
    cols = [str(c).strip() for c in d.columns]
    low  = [c.lower() for c in cols]
    t = target.strip().lower()
    for i,c in enumerate(low):
        if c == t:
            return cols[i]
    # fallback keyword search
    for i,c in enumerate(low):
        if all(k in c for k in keywords):
            return cols[i]
    return None

# Section selector
if sec_col:
    sec_normC = df[sec_col].astype(str).map(normalize_section)
    presentC = sorted([s for s in sec_normC.dropna().unique()])
    st.caption("Filter by section:")
    sel_secsC = section_checkbox_selector(presentC, key_prefix="comments_sections")
    mask_secC = sec_normC.isin(sel_secsC) if sel_secsC else pd.Series(False, index=df.index)
else:
    mask_secC = pd.Series(True, index=df.index)

# Happiness cluster selector (based on Q1)
st.caption("Filter by happiness cluster (based on Q1):")
hb_btm, hb_mid, hb_top = cluster_toggles("comments_hb")
q1_for_cluster = find_q1_column(df)
if q1_for_cluster:
    h = pd.to_numeric(df[q1_for_cluster], errors="coerce")
    mask_hb = pd.Series(False, index=df.index)
    if hb_btm:
        mask_hb |= h.between(0,3, inclusive="both")
    if hb_mid:
        mask_hb |= h.between(4,7, inclusive="both")
    if hb_top:
        mask_hb |= h.between(8,10, inclusive="both")
else:
    mask_hb = pd.Series(True, index=df.index)

# Question chooser
q_label = st.selectbox("Select an open-text question to explore:", list(COMMENTS_MAP.keys()))
tgt, kw = COMMENTS_MAP[q_label]
col_open = find_open_text_column(df, tgt, kw)

if not col_open:
    st.info("Couldn’t locate that open-text question in the dataset.")
else:
    comments = df[col_open].astype(str).str.strip()
    mask = mask_secC & mask_hb & comments.str.len().gt(0)
    showC = pd.DataFrame({"Comment": comments[mask]})
    st.dataframe(showC, use_container_width=True, height=500)

    st.caption(f"Displayed N (based on current selections) = {len(showC)}.")

# ======================================================
# Figure 18 — Q11 Three words to describe our work culture (word cloud)
# ======================================================
st.header("Figure 18. Three words to describe our work culture")

def find_q11_column(d: pd.DataFrame) -> Optional[str]:
    cols = [str(c).strip() for c in d.columns]
    low  = [c.lower() for c in cols]
    target = "what three words would you use to describe our work culture"
    for i,c in enumerate(low):
        if c == target:
            return cols[i]
    for i,c in enumerate(low):
        if ("three words" in c or "3 words" in c) and ("culture" in c):
            return cols[i]
    return None

q11_col = find_q11_column(df)
if not q11_col:
    st.info("Couldn’t locate Q11 (three words).")
else:
    # Section filter (reuse Q4 style)
    if sec_col:
        sec_norm11 = df[sec_col].astype(str).map(normalize_section)
        present11 = sorted([s for s in sec_norm11.dropna().unique()])
        st.caption("Filter by section:")
        sel11 = section_checkbox_selector(present11, key_prefix="q11_sections")
        mask11 = sec_norm11.isin(sel11) if sel11 else pd.Series(False, index=df.index)
    else:
        mask11 = pd.Series(True, index=df.index)

    # Happiness cluster (based on Q1)
    st.caption("Filter by happiness cluster (based on Q1):")
    hb_btm, hb_mid, hb_top = cluster_toggles("q11_hb")
    q1_for_cluster = find_q1_column(df)
    if q1_for_cluster:
        h = pd.to_numeric(df[q1_for_cluster], errors="coerce")
        mask_hb = pd.Series(False, index=df.index)
        if hb_btm: mask_hb |= h.between(0,3, inclusive="both")
        if hb_mid: mask_hb |= h.between(4,7, inclusive="both")
        if hb_top: mask_hb |= h.between(8,10, inclusive="both")
    else:
        mask_hb = pd.Series(True, index=df.index)

    # Prepare tokens
    import re
    comments = df.loc[mask11 & mask_hb, q11_col].astype(str).str.lower().str.strip()
    tokens = []
    for c in comments.dropna():
        # split on commas/slashes/spaces, keep alpha/apostrophes
        parts = re.findall(r"[a-z']+", c.replace("/"," ").replace(","," "))
        tokens.extend(parts)
    STOP = set("""
        the a an and or of to in for with on at by from about as is are was were be been being i me my we our you your
        they their them it its this that these those not yes if but so just more very really also etc etc osbm work
        working workplace team teams office culture
    """.split())
    words = [w for w in tokens if len(w) >= 3 and w not in STOP]

    # Frequency
    from collections import Counter
    freq = Counter(words)
    total_words = sum(freq.values())

    # Word cloud (if library available)
    try:
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt
        if total_words > 0:
            wc = WordCloud(width=900, height=450, background_color="white").generate_from_frequencies(freq)
            fig_wc, ax = plt.subplots(figsize=(9,4.5))
            ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
            st.pyplot(fig_wc, use_container_width=True)
        else:
            st.caption("No words to display for current selections.")
    except Exception:
        # Fallback: show top terms bar chart
        if total_words > 0:
            top_terms = pd.DataFrame(freq.most_common(20), columns=["term","count"])
            fig_wc = px.bar(top_terms, x="term", y="count", title="Top words (fallback)")
            st.plotly_chart(fig_wc, use_container_width=True)
        else:
            st.caption("No words to display for current selections.")

    st.caption(f"Displayed N (based on current selections) = {comments.dropna().str.len().gt(0).sum()}.")

# ======================================================
# Figure 19 — Importance of information sources (diverging stacked, 2025)
# ======================================================
st.header("Figure 19. Importance of information sources")

# Q22 item list per instrument
Q22_ITEMS = [
    (("all hands","meetings"),  "All Hands Staff Meetings"),
    (("section","team","meetings"), "Section/Team Meetings"),
    (("informal","conversations"), "Informal Conversations with Co-Workers"),
    (("notes","office"),        "\"Notes from the Office\" (Monthly Newsletter)"),
    (("website",),               "OSBM Website"),
    (("intranet",),              "OSBM Intranet Site"),
    (("individual","supervisor"), "Individual Meetings with your Supervisor"),
    (("linkedin",),              "OSBM LinkedIn Account"),
    (("hr","highlights"),       "OSBM HR Highlights Email"),
    (("news","clips"),          "OSBM News Clips (Email)"),
]

cols_lc = [str(c).strip() for c in df.columns]
cols_lc_lower = [c.lower() for c in cols_lc]

def find_q22_col_by_keywords(keys: Tuple[str,...]) -> Optional[str]:
    for i,c in enumerate(cols_lc_lower):
        if all(k in c for k in keys):
            return cols_lc[i]
    return None

found22: List[Tuple[str,str]] = []
for keys,label in Q22_ITEMS:
    col = find_q22_col_by_keywords(keys)
    if col: found22.append((label,col))

if not found22:
    st.info("Couldn’t locate Q22 (sources) columns in the dataset.")
else:
    # Section filter
    if sec_col:
        sec_norm22 = df[sec_col].astype(str).map(normalize_section)
        present22 = sorted([s for s in sec_norm22.dropna().unique()])
        st.caption("Filter statements by section:")
        sel22 = section_checkbox_selector(present22, key_prefix="q22_sections")
        mask22 = sec_norm22.isin(sel22) if sel22 else pd.Series(False, index=df.index)
    else:
        mask22 = pd.Series(True, index=df.index)
    df_q22 = df.loc[mask22].copy()

    # Cluster toggles
    st.caption("Select which rating clusters to display in the chart:")
    btm, mid, top = cluster_toggles("q22")
    neg_sel, pos_sel = [], []
    if btm: neg_sel += ["Strongly Disagree","Disagree"]
    if mid: neg_sel += ["Slightly Disagree"]; pos_sel += ["Slightly Agree"]
    if top: pos_sel += ["Agree","Strongly Agree"]
    neg_order = [c for c in ["Slightly Disagree","Disagree","Strongly Disagree"] if c in neg_sel]
    pos_order = [c for c in ["Slightly Agree","Agree","Strongly Agree"] if c in pos_sel]
    show_cats = neg_order + pos_order

    # Displayed N across selected clusters
    if show_cats:
        mask_disp22 = pd.Series(False, index=df_q22.index)
        for _, col in found22:
            mask_disp22 |= df_q22[col].apply(normalize_likert).isin(show_cats)
        dispN22 = int(mask_disp22.sum())
    else:
        dispN22 = 0

    # Order items by Top-box (A+SA) on section-filtered data
    scored22 = []  # (topbox_pct, label, col)
    for label,col in found22:
        s_all = df_q22[col].apply(normalize_likert).dropna()
        tb = (s_all.isin(["Agree","Strongly Agree"]).mean() * 100) if len(s_all) else 0
        scored22.append((tb, label, col))
    scored22.sort(reverse=True)

    # Build percentages arrays in sorted order
    item_labels22 = []
    pct_by_cat22 = {cat: [] for cat in LIKERT_ORDER}
    for tb,label,col in scored22:
        s = df_q22[col].apply(normalize_likert).dropna()
        counts = (s.value_counts(normalize=True) * 100)
        item_labels22.append(label)
        for cat in LIKERT_ORDER:
            pct_by_cat22[cat].append(round(float(counts.get(cat, 0.0)), 0))

    # Chart — diverging stacked
    color_map = {"Strongly Disagree":"#D55E00","Disagree":"#E69F00","Slightly Disagree":"#F0E442",
                 "Slightly Agree":"#56B4E9","Agree":"#0072B2","Strongly Agree":"#004B87"}
    fig22 = go.Figure()
    for cat in neg_order:
        fig22.add_trace(go.Bar(x=[-v for v in pct_by_cat22[cat]], y=item_labels22, name=cat, orientation="h",
                               marker_color=color_map[cat],
                               hovertemplate=f"{cat}: "+"%{customdata:.0f}%<extra></extra>",
                               customdata=pct_by_cat22[cat]))
    for cat in pos_order:
        fig22.add_trace(go.Bar(x=pct_by_cat22[cat], y=item_labels22, name=cat, orientation="h",
                               marker_color=color_map[cat],
                               hovertemplate=f"{cat}: "+"%{customdata:.0f}%<extra></extra>",
                               customdata=pct_by_cat22[cat]))
    fig22.update_layout(barmode="relative",
                        xaxis=dict(title="% of responses", range=[-100,100], tickmode="array",
                                   tickvals=[-100,-90,-80,-70,-60,-50,-40,-30,-20,-10,0,10,20,30,40,50,60,70,80,90,100],
                                   ticktext=[f"{abs(v)}%" for v in [-100,-90,-80,-70,-60,-50,-40,-30,-20,-10,0,10,20,30,40,50,60,70,80,90,100]],
                                   zeroline=True, zerolinewidth=2, zerolinecolor="#888"),
                        yaxis=dict(autorange="reversed", tickfont=dict(size=16, color="black"), automargin=True),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                        margin=dict(l=40, r=20, t=60, b=40),
                        height=440 + 40*max(0, len(item_labels22) - 4))
    st.plotly_chart(fig22, use_container_width=True)
    st.caption(f"Displayed N (based on current selections) = {dispN22}.")

    # Table beneath the chart
    rows22 = []
    for tb,label,col in scored22:
        s = df_q22[col].apply(normalize_likert).dropna()
        s_sel = s[s.isin(show_cats)] if show_cats else s.iloc[0:0]
        total = len(s_sel)
        row = {"Statement": label}
        counts = s_sel.value_counts()
        for cat in LIKERT_ORDER:
            n = int(counts.get(cat, 0)); pct = (n/total*100) if total else 0
            row[cat] = f"{n} ({pct:.0f}%)"
        rows22.append(row)
    st.dataframe(pd.DataFrame(rows22, columns=["Statement"] + LIKERT_ORDER), use_container_width=True)
    if not show_cats:
        st.info("Select at least one rating cluster to display the chart.")
        st.caption("Displayed N (based on current selections) = 0.")