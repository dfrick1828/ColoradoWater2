
import os
from datetime import date, datetime, timedelta
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(
    page_title="Northern Colorado Water Right Outlook",
    page_icon="💧",
    layout="wide",
)

# -----------------------------
# Branding / styling
# -----------------------------
REGIME_ORDER = [
    "Free River",
    "Mild Administration",
    "Normal Administration",
    "Senior Administration",
    "Exceptional Administration",
]
REGIME_SCORE = {
    "Free River": 0,
    "Mild Administration": 25,
    "Normal Administration": 50,
    "Senior Administration": 75,
    "Exceptional Administration": 100,
}
REGIME_COLORS = {
    "Free River": "#79c7d6",
    "Mild Administration": "#f1d36b",
    "Normal Administration": "#f2a34a",
    "Senior Administration": "#e45f56",
    "Exceptional Administration": "#7c4cc2",
}

def safe_to_datetime(series_or_value):
    """
    Robust datetime parser for CDSS fields that may contain mixed formats,
    timezone-aware strings, plain dates, blanks, or unexpected objects.
    """
    if isinstance(series_or_value, pd.Series):
        s = series_or_value.astype("string").str.strip()
        s = s.replace({"": pd.NA, "NaT": pd.NA, "None": pd.NA, "nan": pd.NA})
        return pd.to_datetime(s, errors="coerce", format="mixed", utc=False)
    if pd.isna(series_or_value):
        return pd.NaT
    return pd.to_datetime(str(series_or_value).strip(), errors="coerce", format="mixed", utc=False)

def ensure_column(df, column_name, default=pd.NA):
    """Guarantee a column exists before downstream parsing."""
    if column_name not in df.columns:
        df[column_name] = default
    return df
INCLUDED_WDS = {1, 3, 69}
EXCLUDED_WDS = {2, 4, 5, 6, 7, 8, 9}

st.markdown(
    """
<style>
[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(circle at 12% 0%, rgba(121,199,214,.15), transparent 28%),
    radial-gradient(circle at 86% 8%, rgba(124,76,194,.12), transparent 30%),
    linear-gradient(180deg, #071018, #0b1117 70%);
  color: #edf4f7;
}
[data-testid="stHeader"] { background: rgba(0,0,0,0); }
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; }
.hero {
  border: 1px solid rgba(255,255,255,.12);
  border-radius: 28px;
  padding: 34px 38px;
  background:
    linear-gradient(90deg, rgba(7,16,24,.97), rgba(7,16,24,.78), rgba(7,16,24,.38)),
    url('https://images.unsplash.com/photo-1500530855697-b586d89ba3ee?auto=format&fit=crop&w=1800&q=80');
  background-size: cover;
  background-position: center;
  box-shadow: 0 30px 90px rgba(0,0,0,.35);
  margin-bottom: 24px;
}
.eyebrow {
  color: #79c7d6;
  text-transform: uppercase;
  letter-spacing: .18em;
  font-weight: 800;
  font-size: 12px;
}
.hero h1 {
  font-size: 58px;
  line-height: .95;
  letter-spacing: -.055em;
  margin: 18px 0 18px;
  color: #f3f7f9;
  max-width: 900px;
}
.hero p {
  max-width: 760px;
  font-size: 21px;
  line-height: 1.46;
  color: #d9e6eb;
}
.card {
  border: 1px solid rgba(255,255,255,.10);
  border-radius: 22px;
  background: linear-gradient(180deg, rgba(255,255,255,.062), rgba(255,255,255,.026));
  padding: 22px;
  box-shadow: 0 16px 44px rgba(0,0,0,.20);
}
.kicker {
  color: #9eb0bc;
  text-transform: uppercase;
  letter-spacing: .13em;
  font-weight: 800;
  font-size: 12px;
}
.big {
  font-size: 34px;
  line-height: 1.05;
  letter-spacing: -.045em;
  margin: 8px 0 6px;
  font-weight: 850;
  color: #f3f7f9;
}
.copy { color: #cbd8de; font-size: 15.5px; line-height: 1.55; }
.small-muted { color: #9eb0bc; font-size: 13px; }
.status-pill {
  display: inline-block;
  padding: 8px 13px;
  border-radius: 999px;
  border: 1px solid rgba(255,255,255,.14);
  background: rgba(255,255,255,.08);
  font-weight: 800;
}
.disclaimer {
  border-top: 1px solid rgba(255,255,255,.10);
  margin-top: 18px;
  padding-top: 12px;
  color: #9eb0bc;
  font-size: 12.5px;
}
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# CDSS API helpers
# -----------------------------
BASE_URL = "https://dwr.state.co.us/Rest/GET/api/v2"

def get_api_key():
    # Streamlit Cloud: add this in app secrets as CDSS_API_KEY = "..."
    try:
        return st.secrets.get("CDSS_API_KEY", "")
    except Exception:
        return os.environ.get("CDSS_API_KEY", "")

@st.cache_data(ttl=60 * 20, show_spinner=False)
def cdss_get(endpoint: str, params: dict):
    """Generic CDSS REST GET with Streamlit caching."""
    api_key = get_api_key()
    params = dict(params)
    params.setdefault("format", "json")
    params.setdefault("pageSize", "500000")
    headers = {}
    if api_key:
        headers["ApiKey"] = api_key
    url = f"{BASE_URL}/{endpoint.lstrip('/')}"
    r = requests.get(url, params=params, headers=headers, timeout=35)
    r.raise_for_status()
    obj = r.json()
    if isinstance(obj, dict) and "ResultList" in obj:
        return pd.DataFrame(obj["ResultList"]), url + "?" + urlencode(params), obj
    if isinstance(obj, list):
        return pd.DataFrame(obj), url + "?" + urlencode(params), obj
    return pd.DataFrame(), url + "?" + urlencode(params), obj

@st.cache_data(ttl=60 * 20, show_spinner=False)
def fetch_live_active_calls():
    """
    Pull live active calls from DWR/CDSS and filter to WD1, WD3, WD69,
    excluding districts 2,4,5,6,7,8,9.
    """
    # CDSS supports active administrative calls. Division 1 keeps the request smaller.
    df, url, raw = cdss_get("administrativecalls/active", {
        "division": "1",
        "format": "json",
        "pageSize": "500000",
    })
    if df.empty:
        return df, url, "empty"

    # Normalize expected field names from CDSS JSON.
    rename = {
        "dateTimeSet": "Date Time Set",
        "dateTimeReleased": "Date Time Released",
        "waterSourceName": "Water Source",
        "locationWdid": "Location WDID",
        "locationStructureName": "Location Structure Name",
        "priorityWdid": "Call Priority WDID",
        "priorityStructureName": "Priority Structure Name",
        "priorityAdminNumber": "Priority Admin No",
        "priorityDate": "Priority Date",
        "modified": "Modified",
        "moreInformation": "More_Information",
        "division": "Division",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    df["Location WDID"] = pd.to_numeric(df.get("Location WDID"), errors="coerce")
    df["wd"] = (df["Location WDID"] // 100000).astype("Int64")
    df = df[df["wd"].isin(INCLUDED_WDS)].copy()
    df = df[~df["wd"].isin(EXCLUDED_WDS)].copy()
    return df, url, "live"

@st.cache_data(ttl=60 * 60, show_spinner=False)
def fetch_recent_flow_live(abbrev="CLAFTCCO", days=120):
    """
    Try DWR telemetry daily data first. If it fails, return empty and use fallback.
    The exact field names vary by service output, so we normalize several possibilities.
    """
    start = (date.today() - timedelta(days=days)).strftime("%m/%d/%Y")
    # Try telemetry daily discharge at canyon-mouth station abbreviation.
    attempts = [
        ("telemetrystations/telemetrytimeseriesday", {
            "abbrev": abbrev,
            "parameter": "DISCHRG",
            "min-measDate": start,
            "format": "json",
            "pageSize": "500000",
        }),
        ("telemetrystations/telemetrytimeseriesday", {
            "abbrev": abbrev,
            "parameter": "Q",
            "min-measDate": start,
            "format": "json",
            "pageSize": "500000",
        }),
    ]
    errors = []
    for endpoint, params in attempts:
        try:
            df, url, raw = cdss_get(endpoint, params)
            if df.empty:
                continue
            # Try to find date/value fields.
            date_candidates = [c for c in df.columns if c.lower() in {"measdate", "measurementdate", "date", "dataMeasDate".lower()} or "date" in c.lower()]
            value_candidates = [c for c in df.columns if c.lower() in {"measvalue", "value", "amount", "dataValue".lower()} or "value" in c.lower()]
            if not date_candidates or not value_candidates:
                continue
            out = pd.DataFrame({
                "date": safe_to_datetime(df[date_candidates[0]]),
                "flow_cfs": pd.to_numeric(df[value_candidates[0]], errors="coerce"),
            }).dropna()
            if not out.empty:
                return out.sort_values("date"), url, "live"
        except Exception as e:
            errors.append(str(e))
            continue
    return pd.DataFrame(), "; ".join(errors), "fallback"

# -----------------------------
# Core model logic
# -----------------------------
def classify_regime(set_dt, priority_dt):
    if pd.isna(set_dt) or pd.isna(priority_dt):
        return "Free River", 0, 0
    py = int(pd.Timestamp(priority_dt).year)
    set_dt = pd.Timestamp(set_dt)
    july15 = pd.Timestamp(year=set_dt.year, month=7, day=15)

    if py >= 1871:
        return "Mild Administration", 1, 25
    if 1867 <= py <= 1870:
        return "Normal Administration", 2, 50
    if 1863 <= py <= 1866:
        return "Senior Administration", 3, 75
    # 1862 and earlier
    if set_dt.normalize() < july15:
        return "Exceptional Administration", 4, 100
    return "Senior Administration", 3, 75

@st.cache_data(show_spinner=False)
def load_local_data():
    daily = pd.read_csv("data/SouthPlatte_Basin_Daily_Regime_Index.csv", parse_dates=["date"])
    annual = pd.read_csv("data/SouthPlatte_Basin_Annual_Regime_Days.csv")
    flow = pd.read_csv("data/Selected_Station_Analysis_Xtab_202605281804.csv")
    flow["date"] = pd.to_datetime(flow["meas_date"], errors="coerce")
    flow = flow.rename(columns={"Streamflow Value": "flow_cfs"})[["date", "flow_cfs"]]
    flow["flow_cfs"] = pd.to_numeric(flow["flow_cfs"], errors="coerce")
    return daily, annual, flow

def build_current_regime_from_live(active_calls):
    if active_calls.empty:
        return {
            "regime": "Free River",
            "severity": 0,
            "score": 0,
            "controlling_wd": None,
            "controlling_priority_date": None,
            "controlling_priority_structure": None,
            "source": "live active calls",
        }
    df = active_calls.copy()
    df = ensure_column(df, "Date Time Set")
    df = ensure_column(df, "Priority Date")
    df["set_dt"] = safe_to_datetime(df["Date Time Set"])
    df["priority_dt"] = safe_to_datetime(df["Priority Date"])
    classified = df.apply(lambda r: classify_regime(r["set_dt"], r["priority_dt"]), axis=1, result_type="expand")
    classified.columns = ["regime", "severity", "score"]
    df = pd.concat([df, classified], axis=1)
    df = df.sort_values(["severity", "Priority Admin No"], ascending=[False, True])
    top = df.iloc[0]
    return {
        "regime": top["regime"],
        "severity": int(top["severity"]),
        "score": int(top["score"]),
        "controlling_wd": int(top["wd"]) if pd.notna(top["wd"]) else None,
        "controlling_priority_date": top.get("Priority Date"),
        "controlling_priority_structure": top.get("Priority Structure Name"),
        "source": "live active calls",
    }

def prepare_model_dataset(daily, flow):
    df = daily.merge(flow, on="date", how="left").sort_values("date").reset_index(drop=True)
    df["flow_cfs"] = pd.to_numeric(df["flow_cfs"], errors="coerce").interpolate(limit_direction="both")
    df["flow_7d"] = df["flow_cfs"].rolling(7, min_periods=1).mean()
    df["flow_14d"] = df["flow_cfs"].rolling(14, min_periods=1).mean()
    df["flow_30d"] = df["flow_cfs"].rolling(30, min_periods=1).mean()
    df["flow_change_7d"] = df["flow_cfs"] - df["flow_cfs"].shift(7)
    df["flow_change_14d"] = df["flow_cfs"] - df["flow_cfs"].shift(14)

    df["flow_doy_percentile"] = np.nan
    for doy in range(1, 367):
        window = list(range(max(1, doy-7), min(366, doy+7)+1))
        hist = df[df["doy"].isin(window)]["flow_cfs"].dropna()
        mask = df["doy"] == doy
        if len(hist) > 10:
            df.loc[mask, "flow_doy_percentile"] = df.loc[mask, "flow_cfs"].apply(lambda x: (hist <= x).mean() * 100)

    df["severity_change_7d"] = df["severity"] - df["severity"].shift(7)
    df["severity_change_14d"] = df["severity"] - df["severity"].shift(14)
    regime_change = (df["regime"] != df["regime"].shift(1)).cumsum()
    df["days_in_current_regime"] = df.groupby(regime_change).cumcount() + 1
    df["sin_doy"] = np.sin(2 * np.pi * df["doy"] / 366)
    df["cos_doy"] = np.cos(2 * np.pi * df["doy"] / 366)

    horizon = 30
    df["target_regime_30d"] = df["regime"].shift(-horizon)
    return df

FEATURES = [
    "severity", "score", "historical_percentile", "month", "doy", "sin_doy", "cos_doy",
    "flow_cfs", "flow_7d", "flow_14d", "flow_30d", "flow_change_7d", "flow_change_14d",
    "flow_doy_percentile", "severity_change_7d", "severity_change_14d", "days_in_current_regime",
]

@st.cache_resource(show_spinner=False)
def train_model(model_df):
    train_df = model_df.dropna(subset=FEATURES + ["target_regime_30d"]).copy()
    le = LabelEncoder()
    le.fit(REGIME_ORDER)
    # Drop target classes not present? Force labels by fitting REGIME_ORDER but train only rows with present targets.
    y = le.transform(train_df["target_regime_30d"])
    clf = RandomForestClassifier(
        n_estimators=500,
        max_depth=8,
        min_samples_leaf=10,
        random_state=42,
        class_weight="balanced_subsample",
    )
    clf.fit(train_df[FEATURES], y)
    return clf, le

def make_forecast_row(daily_model, current_state, live_flow=None):
    # Start from latest historical row for persistence/seasonal/flow features, then override current regime.
    row = daily_model.iloc[-1:].copy()
    today = pd.Timestamp(date.today())
    row.loc[row.index[0], "date"] = today
    row.loc[row.index[0], "year"] = today.year
    row.loc[row.index[0], "month"] = today.month
    row.loc[row.index[0], "doy"] = today.dayofyear
    row.loc[row.index[0], "regime"] = current_state["regime"]
    row.loc[row.index[0], "severity"] = current_state["severity"]
    row.loc[row.index[0], "score"] = current_state["score"]

    # Percentile based on historical daily scores.
    hist_scores = daily_model["score"].dropna()
    row.loc[row.index[0], "historical_percentile"] = (hist_scores <= current_state["score"]).mean() * 100

    # Use most recent live flow if available.
    if live_flow is not None and not live_flow.empty:
        lf = live_flow.sort_values("date")
        recent = lf.tail(30).copy()
        last_flow = float(recent["flow_cfs"].iloc[-1])
        row.loc[row.index[0], "flow_cfs"] = last_flow
        row.loc[row.index[0], "flow_7d"] = recent["flow_cfs"].tail(7).mean()
        row.loc[row.index[0], "flow_14d"] = recent["flow_cfs"].tail(14).mean()
        row.loc[row.index[0], "flow_30d"] = recent["flow_cfs"].tail(30).mean()
        row.loc[row.index[0], "flow_change_7d"] = last_flow - float(recent["flow_cfs"].iloc[-8]) if len(recent) >= 8 else 0
        row.loc[row.index[0], "flow_change_14d"] = last_flow - float(recent["flow_cfs"].iloc[-15]) if len(recent) >= 15 else 0

    row.loc[row.index[0], "sin_doy"] = np.sin(2 * np.pi * today.dayofyear / 366)
    row.loc[row.index[0], "cos_doy"] = np.cos(2 * np.pi * today.dayofyear / 366)

    # Keep last historical persistence features unless regime changed sharply.
    return row

# -----------------------------
# Load data and live refresh
# -----------------------------
daily_hist, annual, flow_hist = load_local_data()
model_df = prepare_model_dataset(daily_hist, flow_hist)
clf, le = train_model(model_df)

with st.spinner("Checking CDSS live data..."):
    try:
        live_calls, live_calls_url, live_status = fetch_live_active_calls()
    except Exception as e:
        live_calls = pd.DataFrame()
        live_calls_url = str(e)
        live_status = "fallback"

    try:
        live_flow, live_flow_url, live_flow_status = fetch_recent_flow_live("CLAFTCCO", 120)
    except Exception as e:
        live_flow = pd.DataFrame()
        live_flow_url = str(e)
        live_flow_status = "fallback"

current = build_current_regime_from_live(live_calls) if live_status == "live" else {
    "regime": daily_hist.iloc[-1]["regime"],
    "severity": int(daily_hist.iloc[-1]["severity"]),
    "score": int(daily_hist.iloc[-1]["score"]),
    "controlling_wd": daily_hist.iloc[-1].get("controlling_wd"),
    "controlling_priority_date": daily_hist.iloc[-1].get("controlling_priority_date"),
    "controlling_priority_structure": daily_hist.iloc[-1].get("controlling_priority_structure"),
    "source": "snapshot fallback",
}

forecast_row = make_forecast_row(model_df, current, live_flow if live_flow_status == "live" else None)
X_current = forecast_row[FEATURES].fillna(model_df[FEATURES].median(numeric_only=True))
prob = clf.predict_proba(X_current)[0]
prob_series = pd.Series(prob, index=le.inverse_transform(np.arange(len(prob)))).reindex(REGIME_ORDER, fill_value=0)
prob_series = prob_series.sort_values(ascending=False)

# -----------------------------
# UI
# -----------------------------
st.markdown(
    """
<div class="hero">
  <div class="eyebrow">Live prototype</div>
  <h1>Northern Colorado Water Right Outlook</h1>
  <p>A plain-English view of how the South Platte and Poudre are being administered — and what regime may be coming next.</p>
</div>
""",
    unsafe_allow_html=True,
)

top1, top2, top3 = st.columns(3)
with top1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="kicker">Current regime</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="big" style="color:{REGIME_COLORS[current["regime"]]};">{current["regime"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="copy">Based on {current["source"]}. Controlling WD: {current.get("controlling_wd") or "—"}.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with top2:
    hist_pct = (model_df["score"].dropna() <= current["score"]).mean() * 100
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="kicker">Historical severity</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="big">{hist_pct:.0f}th percentile</div>', unsafe_allow_html=True)
    st.markdown('<div class="copy">How today compares to daily administrative conditions since 2005.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with top3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="kicker">Most likely in 30 days</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="big" style="color:{REGIME_COLORS[prob_series.index[0]]};">{prob_series.index[0]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="copy">Model probability: {prob_series.iloc[0]:.0%}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

left, right = st.columns([1.05, 0.95], gap="large")

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("30-day water-right outlook")
    fig = go.Figure()
    for regime in REGIME_ORDER:
        fig.add_trace(go.Bar(
            x=[prob_series.get(regime, 0)],
            y=[regime],
            orientation="h",
            marker_color=REGIME_COLORS[regime],
            text=[f"{prob_series.get(regime, 0):.0%}"],
            textposition="outside",
            name=regime,
            showlegend=False,
        ))
    fig.update_layout(
        height=350,
        margin=dict(l=10, r=30, t=10, b=10),
        xaxis=dict(range=[0, 1], tickformat=".0%", gridcolor="rgba(255,255,255,.10)"),
        yaxis=dict(categoryorder="array", categoryarray=list(reversed(REGIME_ORDER))),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#edf4f7"),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('<div class="disclaimer">Prototype model. Not an official forecast, legal opinion, or substitute for CDSS/DWR records.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("What this means")
    priority = current.get("controlling_priority_date") or "—"
    structure = current.get("controlling_priority_structure") or "—"
    st.markdown(
        f"""
<div class="copy">
<strong>Current controlling signal:</strong><br>
Priority: {priority}<br>
Structure: {structure}<br><br>
<strong>Plain English:</strong><br>
This translates live call administration into a public-facing water-right condition. The forecast asks whether the basin is likely to ease, hold, or tighten over the next 30 days.
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("### Recent administrative stress")
recent = model_df.tail(180).copy()
if live_status == "live":
    # append a current live point
    new_row = recent.iloc[-1:].copy()
    new_row["date"] = pd.Timestamp(date.today())
    new_row["score"] = current["score"]
    new_row["regime"] = current["regime"]
    recent = pd.concat([recent, new_row], ignore_index=True)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=recent["date"],
    y=recent["score"],
    mode="lines",
    line=dict(color="#79c7d6", width=3),
    fill="tozeroy",
    fillcolor="rgba(121,199,214,.10)",
))
fig.update_layout(
    height=320,
    margin=dict(l=10, r=10, t=20, b=10),
    yaxis=dict(title="Administrative severity", range=[0, 105], gridcolor="rgba(255,255,255,.10)"),
    xaxis=dict(gridcolor="rgba(255,255,255,.06)"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#edf4f7"),
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("### Annual regime history")
annual = annual.rename(columns={annual.columns[0]: "year"}) if annual.columns[0] != "year" else annual
plot_annual = annual[annual["year"] >= 2010].copy()
fig = go.Figure()
for regime in REGIME_ORDER:
    if regime in plot_annual.columns:
        fig.add_trace(go.Bar(
            x=plot_annual["year"].astype(str),
            y=plot_annual[regime],
            name=regime,
            marker_color=REGIME_COLORS[regime],
        ))
fig.update_layout(
    barmode="stack",
    height=430,
    margin=dict(l=10, r=10, t=20, b=10),
    yaxis=dict(title="Days", gridcolor="rgba(255,255,255,.10)"),
    xaxis=dict(title="Year"),
    legend=dict(orientation="h", y=1.12),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#edf4f7"),
)
st.plotly_chart(fig, use_container_width=True)

with st.expander("Live data diagnostics"):
    st.write("Active calls status:", live_status)
    st.write("Active calls URL/error:", live_calls_url)
    st.write("Live active call rows after WD filter:", len(live_calls))
    st.dataframe(live_calls.head(50), use_container_width=True)

    st.write("Flow status:", live_flow_status)
    st.write("Flow URL/error:", live_flow_url)
    if live_flow_status == "live":
        st.dataframe(live_flow.tail(20), use_container_width=True)

st.markdown(
    """
<div class="disclaimer">
Data sources: Colorado DWR/CDSS active administrative calls and telemetry/surface-water REST services when available; historical prototype data is included as a local model snapshot. This platform is for public education and model development only.
</div>
""",
    unsafe_allow_html=True,
)
