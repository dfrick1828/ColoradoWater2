
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(
    page_title="Northern Colorado Water Right Outlook",
    page_icon="💧",
    layout="wide",
)

REGIME_ORDER = [
    "Free River",
    "Mild Administration",
    "Normal Administration",
    "Senior Administration",
    "Exceptional Administration",
]
REGIME_COLORS = {
    "Free River": "#79c7d6",
    "Mild Administration": "#f1d36b",
    "Normal Administration": "#f2a34a",
    "Senior Administration": "#e45f56",
    "Exceptional Administration": "#7c4cc2",
}

st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(circle at 12% 0%, rgba(121,199,214,.15), transparent 28%),
    radial-gradient(circle at 86% 8%, rgba(124,76,194,.12), transparent 30%),
    linear-gradient(180deg, #071018, #0b1117 70%);
  color: #edf4f7;
}
[data-testid="stHeader"] { background: rgba(0,0,0,0); }
.block-container { padding-top: 2rem; padding-bottom: 3rem; }
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
.eyebrow { color:#79c7d6; text-transform:uppercase; letter-spacing:.18em; font-weight:800; font-size:12px; }
.hero h1 { font-size:56px; line-height:.95; letter-spacing:-.055em; margin:18px 0; color:#f3f7f9; }
.hero p { max-width:780px; font-size:21px; line-height:1.46; color:#d9e6eb; }
.card {
  border: 1px solid rgba(255,255,255,.10);
  border-radius: 22px;
  background: linear-gradient(180deg, rgba(255,255,255,.062), rgba(255,255,255,.026));
  padding: 22px;
  box-shadow: 0 16px 44px rgba(0,0,0,.20);
}
.kicker { color:#9eb0bc; text-transform:uppercase; letter-spacing:.13em; font-weight:800; font-size:12px; }
.big { font-size:34px; line-height:1.05; letter-spacing:-.045em; margin:8px 0 6px; font-weight:850; color:#f3f7f9; }
.copy { color:#cbd8de; font-size:15.5px; line-height:1.55; }
.note { color:#9eb0bc; font-size:12.5px; line-height:1.45; border-top:1px solid rgba(255,255,255,.10); margin-top:14px; padding-top:12px; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    daily = pd.read_csv("data/SouthPlatte_Basin_Daily_Regime_Index.csv", parse_dates=["date"])
    annual = pd.read_csv("data/SouthPlatte_Basin_Annual_Regime_Days.csv")
    flow = pd.read_csv("data/Selected_Station_Analysis_Xtab_202605281804.csv")
    flow["date"] = pd.to_datetime(flow["meas_date"], errors="coerce")
    flow = flow.rename(columns={"Streamflow Value": "flow_cfs"})[["date", "flow_cfs"]]
    flow["flow_cfs"] = pd.to_numeric(flow["flow_cfs"], errors="coerce")
    return daily, annual, flow

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
    df["target_regime_30d"] = df["regime"].shift(-30)
    return df

FEATURES = [
    "severity", "score", "historical_percentile", "month", "doy", "sin_doy", "cos_doy",
    "flow_cfs", "flow_7d", "flow_14d", "flow_30d", "flow_change_7d", "flow_change_14d",
    "flow_doy_percentile", "severity_change_7d", "severity_change_14d", "days_in_current_regime",
]

@st.cache_resource
def train_model(model_df):
    train_df = model_df.dropna(subset=FEATURES + ["target_regime_30d"]).copy()
    le = LabelEncoder()
    le.fit(REGIME_ORDER)
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

daily, annual, flow = load_data()
model_df = prepare_model_dataset(daily, flow)
clf, le = train_model(model_df)

latest_available = daily["date"].max()
default_date = min(latest_available, pd.Timestamp("2026-04-28"))

st.markdown("""
<div class="hero">
  <div class="eyebrow">Historical-data prototype</div>
  <h1>Northern Colorado Water Right Outlook</h1>
  <p>A public-facing dashboard that translates historical South Platte and Poudre water-right administration into plain-English regimes and a 30-day outlook.</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.title("Scenario date")
date_choice = st.sidebar.date_input(
    "Choose a historical date",
    value=default_date.date(),
    min_value=daily["date"].min().date(),
    max_value=latest_available.date(),
)
selected_date = pd.Timestamp(date_choice)

row = model_df[model_df["date"] == selected_date]
if row.empty:
    st.error("No model data for that date. Try another date.")
    st.stop()
row = row.iloc[-1:]

X = row[FEATURES].fillna(model_df[FEATURES].median(numeric_only=True))
prob = clf.predict_proba(X)[0]
prob_series = pd.Series(prob, index=le.inverse_transform(np.arange(len(prob)))).reindex(REGIME_ORDER, fill_value=0).sort_values(ascending=False)

current_regime = row.iloc[0]["regime"]
hist_pct = row.iloc[0]["historical_percentile"]
score = row.iloc[0]["score"]
priority = row.iloc[0].get("controlling_priority_date", "—")
structure = row.iloc[0].get("controlling_priority_structure", "—")

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="kicker">Current regime</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="big" style="color:{REGIME_COLORS[current_regime]};">{current_regime}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="copy">Selected date: {selected_date.strftime("%B %d, %Y")}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="kicker">Historical severity</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="big">{hist_pct:.0f}th percentile</div>', unsafe_allow_html=True)
    st.markdown('<div class="copy">Compared with daily administrative conditions since 2005.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="kicker">Most likely in 30 days</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="big" style="color:{REGIME_COLORS[prob_series.index[0]]};">{prob_series.index[0]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="copy">Model probability: {prob_series.iloc[0]:.0%}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

left, right = st.columns([1.05, .95], gap="large")
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
    st.markdown('<div class="note">Historical-only prototype. No live CDSS calls. Designed to be stable on Streamlit Cloud.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Current controlling signal")
    st.markdown(f"""
<div class="copy">
<strong>Priority date:</strong> {priority}<br>
<strong>Structure:</strong> {structure}<br>
<strong>Severity score:</strong> {score}/100<br><br>
This translates water-right administration into an understandable public condition.
</div>
""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("### Administrative stress around selected date")
window = model_df[(model_df["date"] >= selected_date - pd.Timedelta(days=90)) & (model_df["date"] <= selected_date + pd.Timedelta(days=90))]
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=window["date"], y=window["score"], mode="lines",
    line=dict(color="#79c7d6", width=3),
    fill="tozeroy", fillcolor="rgba(121,199,214,.10)"
))
fig.add_vline(x=selected_date, line_width=2, line_dash="dash", line_color="#ffffff")
fig.update_layout(
    height=330,
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

with st.expander("About this prototype"):
    st.write("""
This version intentionally uses only the historical datasets already developed in this project.
It avoids live CDSS API calls so the Streamlit deployment is stable. The date selector lets you
simulate what the dashboard would have shown on a chosen historical date.
""")
    st.dataframe(row, use_container_width=True)

st.markdown('<div class="note">Not an official forecast, legal opinion, or substitute for CDSS/DWR records.</div>', unsafe_allow_html=True)
