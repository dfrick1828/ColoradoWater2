
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from pathlib import Path
import base64

def image_to_base64(path):
    p = Path(path)
    if not p.exists():
        return ""
    return base64.b64encode(p.read_bytes()).decode("utf-8")

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
    "Free River": "#5ec9df",
    "Mild Administration": "#f1d36b",
    "Normal Administration": "#f2a34a",
    "Senior Administration": "#e45f56",
    "Exceptional Administration": "#7c4cc2",
}

PUBLIC_LABELS = {
    "Free River": "Water Readily Available",
    "Mild Administration": "Minor Water-Right Pressure",
    "Normal Administration": "Typical Summer Administration",
    "Senior Administration": "Significant Water Stress",
    "Exceptional Administration": "Exceptional Water Stress",
}

PUBLIC_SHORT = {
    "Free River": "Available",
    "Mild Administration": "Minor Pressure",
    "Normal Administration": "Typical Administration",
    "Senior Administration": "Significant Stress",
    "Exceptional Administration": "Exceptional Stress",
}

PUBLIC_EXPLAIN = {
    "Free River": "Water-right calls are not currently controlling the basin in a meaningful way.",
    "Mild Administration": "Water rights are being administered, but the current condition is relatively mild compared with historic stress periods.",
    "Normal Administration": "The basin is operating under a normal seasonal water-right administration pattern.",
    "Senior Administration": "Senior priorities are controlling the river system, indicating meaningful pressure on junior water rights.",
    "Exceptional Administration": "Administration is unusually restrictive for this time of year and historically rare.",
}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&display=swap');

[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(circle at 15% 0%, rgba(94,201,223,.15), transparent 28%),
    radial-gradient(circle at 83% 2%, rgba(124,76,194,.14), transparent 32%),
    linear-gradient(180deg, #061017, #0c1117 68%, #070b10);
  color: #edf4f7;
  font-family: Inter, sans-serif;
}
[data-testid="stHeader"] { background: rgba(0,0,0,0); }
.block-container { padding-top: 1.7rem; padding-bottom: 3rem; max-width: 1220px; }

.hero {
  min-height: 440px;
  border: 1px solid rgba(255,255,255,.13);
  border-radius: 34px;
  padding: 42px;
  background:
    linear-gradient(90deg, rgba(4,12,18,.97), rgba(4,12,18,.74), rgba(4,12,18,.28)),
    url('https://images.unsplash.com/photo-1500530855697-b586d89ba3ee?auto=format&fit=crop&w=1800&q=80');
  background-size: cover;
  background-position: center;
  box-shadow: 0 35px 110px rgba(0,0,0,.42);
  margin-bottom: 26px;
  position: relative;
}
.eyebrow {
  color: #5ec9df;
  text-transform: uppercase;
  letter-spacing: .20em;
  font-weight: 900;
  font-size: 12px;
}
.hero h1 {
  font-size: 66px;
  line-height: .92;
  letter-spacing: -.065em;
  margin: 18px 0 18px;
  color: #f5fafc;
  max-width: 920px;
}
.hero p {
  max-width: 760px;
  font-size: 22px;
  line-height: 1.45;
  color: #dbe8ee;
}
.hero-footer {
  position:absolute;
  left:42px;
  right:42px;
  bottom:34px;
  display:flex;
  justify-content:space-between;
  gap:20px;
  align-items:end;
}
.badge {
  background:rgba(255,255,255,.10);
  border:1px solid rgba(255,255,255,.16);
  backdrop-filter: blur(10px);
  border-radius:999px;
  padding:10px 15px;
  font-weight:850;
  color:white;
}
.card {
  border: 1px solid rgba(255,255,255,.11);
  border-radius: 26px;
  background: linear-gradient(180deg, rgba(255,255,255,.070), rgba(255,255,255,.027));
  padding: 24px;
  box-shadow: 0 18px 55px rgba(0,0,0,.24);
}
.kicker {
  color:#9eb0bc;
  text-transform:uppercase;
  letter-spacing:.13em;
  font-weight:900;
  font-size:12px;
}
.big {
  font-size:38px;
  line-height:1.02;
  letter-spacing:-.050em;
  margin:10px 0 8px;
  font-weight:900;
  color:#f3f7f9;
}
.copy { color:#cbd8de; font-size:15.8px; line-height:1.58; }
.note {
  color:#9eb0bc;
  font-size:12.5px;
  line-height:1.45;
  border-top:1px solid rgba(255,255,255,.10);
  margin-top:14px;
  padding-top:12px;
}
.metric-row {
  display:flex;
  justify-content:space-between;
  gap:14px;
  border-top:1px solid rgba(255,255,255,.08);
  padding-top:12px;
  margin-top:12px;
}
.metric-label { color:#9eb0bc; font-size:13px; }
.metric-value { color:#f3f7f9; font-weight:850; }

.gauge-wrap {
  margin-top: 16px;
}
.gauge-track {
  width: 100%;
  height: 22px;
  border-radius: 999px;
  background: linear-gradient(90deg, #5ec9df, #f1d36b, #f2a34a, #e45f56, #7c4cc2);
  position: relative;
  box-shadow: inset 0 0 0 1px rgba(255,255,255,.15);
}
.gauge-marker {
  position:absolute;
  top:-9px;
  width: 4px;
  height: 40px;
  border-radius: 999px;
  background:#ffffff;
  box-shadow:0 0 0 5px rgba(255,255,255,.16), 0 8px 18px rgba(0,0,0,.35);
}
.gauge-labels {
  display:flex;
  justify-content:space-between;
  color:#9eb0bc;
  font-size:12px;
  margin-top:9px;
}
.pill {
  display:inline-block;
  padding:8px 12px;
  border-radius:999px;
  border:1px solid rgba(255,255,255,.14);
  background:rgba(255,255,255,.07);
  color:#edf4f7;
  font-weight:800;
  margin:4px 6px 4px 0;
}
.meaning-grid {
  display:grid;
  grid-template-columns: repeat(3, 1fr);
  gap:14px;
}
.meaning-card {
  border:1px solid rgba(255,255,255,.09);
  border-radius:18px;
  padding:16px;
  background:rgba(255,255,255,.035);
}
.meaning-card strong { color:#f3f7f9; }
.section-title {
  font-size:27px;
  letter-spacing:-.035em;
  margin: 0 0 10px;
  font-weight:900;
}
.small-muted { color:#9eb0bc; font-size:13px; }
@media (max-width: 900px) {
  .hero h1 { font-size:44px; }
  .hero-footer { position:static; margin-top:30px; flex-direction:column; align-items:flex-start; }
  .meaning-grid { grid-template-columns: 1fr; }
}

.landing-hero {
  min-height: 720px;
  border: 1px solid rgba(255,255,255,.13);
  border-radius: 38px;
  padding: 52px;
  background:
    linear-gradient(90deg, rgba(3,10,15,.96), rgba(3,10,15,.72), rgba(3,10,15,.24)),
    var(--poudre-bg);
  background-size: cover;
  background-position: center;
  box-shadow: 0 38px 120px rgba(0,0,0,.46);
  position: relative;
  overflow: hidden;
}
.landing-hero h1 {
  font-size: 78px;
  line-height: .90;
  letter-spacing: -.075em;
  margin: 20px 0 18px;
  max-width: 980px;
  color: #f6fbfd;
}
.landing-hero p {
  max-width: 780px;
  font-size: 24px;
  line-height: 1.42;
  color: #dfeaf0;
}
.landing-thesis {
  position: absolute;
  left: 52px;
  bottom: 48px;
  max-width: 820px;
  font-size: 31px;
  line-height: 1.18;
  letter-spacing: -.035em;
  font-weight: 850;
  color: #ffffff;
}
.landing-subline {
  color: #9eb0bc;
  font-size: 14px;
  margin-top: 16px;
  letter-spacing: .02em;
}
.landing-grid {
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 18px;
  margin-top: 22px;
}
.landing-card {
  border: 1px solid rgba(255,255,255,.10);
  border-radius: 26px;
  background: linear-gradient(180deg, rgba(255,255,255,.070), rgba(255,255,255,.027));
  padding: 24px;
  min-height: 185px;
  box-shadow: 0 18px 55px rgba(0,0,0,.24);
}
.landing-number {
  font-size: 13px;
  color: #5ec9df;
  font-weight: 900;
  letter-spacing: .16em;
  text-transform: uppercase;
}
.landing-card h3 {
  font-size: 26px;
  line-height: 1.05;
  letter-spacing: -.04em;
  margin: 12px 0 10px;
  color: #f3f7f9;
}
.landing-card p {
  color: #cbd8de;
  line-height: 1.55;
  font-size: 15.5px;
}
.callout {
  margin-top: 22px;
  border: 1px solid rgba(94,201,223,.25);
  border-radius: 28px;
  padding: 30px;
  background: linear-gradient(135deg, rgba(94,201,223,.11), rgba(124,76,194,.08));
}
.callout h2 {
  font-size: 36px;
  letter-spacing: -.05em;
  margin: 0 0 10px;
}
.callout p {
  color: #d6e4ea;
  font-size: 18px;
  line-height: 1.55;
  max-width: 980px;
}
@media (max-width: 900px) {
  .landing-hero { min-height: 620px; padding: 32px; }
  .landing-hero h1 { font-size: 48px; }
  .landing-thesis { position: static; margin-top: 80px; font-size: 24px; }
  .landing-grid { grid-template-columns: 1fr; }
}

</style>
""", unsafe_allow_html=True)

def apply_expert_probability_overlay(prob_series, current_regime, selected_date):
    adjusted = prob_series.copy().astype(float)
    month = int(pd.Timestamp(selected_date).month)
    irrigation_admin_season = 4 <= month <= 10
    active_admin = current_regime != "Free River"
    if irrigation_admin_season and active_admin:
        adjusted.loc["Free River"] = 0.0
        total = adjusted.sum()
        if total > 0:
            adjusted = adjusted / total
    return adjusted

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

def comparable_years(model_df, selected_date, score):
    doy = int(selected_date.dayofyear)
    y0 = int(selected_date.year)
    comps = []
    for y, g in model_df.groupby(model_df["date"].dt.year):
        if y == y0:
            continue
        w = g[g["doy"].between(max(1, doy-30), doy)]
        if len(w) > 10:
            comps.append((y, abs(w["score"].mean() - score), w["score"].mean()))
    comps = sorted(comps, key=lambda x: x[1])[:3]
    return [str(x[0]) for x in comps]

daily, annual, flow = load_data()
model_df = prepare_model_dataset(daily, flow)
clf, le = train_model(model_df)

page = st.sidebar.radio(
    "View",
    ["Landing Page", "Outlook Dashboard"],
    index=0,
)

latest_available = daily["date"].max()
default_date = min(latest_available, pd.Timestamp("2026-04-28"))


if page == "Landing Page":
    img64 = image_to_base64("assets/poudre_river_hero.png")
    if img64:
        bg = f"url('data:image/png;base64,{img64}')"
    else:
        bg = "url('https://images.unsplash.com/photo-1500530855697-b586d89ba3ee?auto=format&fit=crop&w=1800&q=80')"

    st.markdown(f"""
    <div class="landing-hero" style="--poudre-bg: {bg};">
      <div class="eyebrow">Northern Colorado Water Right Outlook</div>
      <h1>Making Colorado's most important natural resource understandable.</h1>
      <p>Colorado has become exceptionally good at measuring water. Snowpack. Streamflow. Reservoir storage. Drought indices. Yet the question most people actually care about remains unanswered: <strong>What does it mean for water rights?</strong></p>
      <div class="landing-thesis">
        We have weather forecasts.<br>
        We have wildfire forecasts.<br>
        We have drought forecasts.<br><br>
        <strong>Why don't we have water-right forecasts?</strong>
        <div class="landing-subline">South Platte / Cache la Poudre historical prototype · Built from administrative call records, flow data, and expert regime rules.</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="landing-grid">
      <div class="landing-card">
        <div class="landing-number">01</div>
        <h3>Translate the system</h3>
        <p>Water-right administration is usually hidden in call records, priority dates, and professional judgment. The Outlook turns that into plain-English conditions.</p>
      </div>
      <div class="landing-card">
        <div class="landing-number">02</div>
        <h3>Compare today to history</h3>
        <p>Every day is placed in historical context so the public can see whether conditions are routine, unusual, or severe compared with the record since 2005.</p>
      </div>
      <div class="landing-card">
        <div class="landing-number">03</div>
        <h3>Look ahead</h3>
        <p>The prototype estimates the likely administrative condition 30 days ahead, with expert rules layered over the statistical model.</p>
      </div>
    </div>

    <div class="callout">
      <h2>Most water platforms answer: "How much water is there?"</h2>
      <p>
      This platform answers: <strong>"What does that mean?"</strong>
      <br><br>
      Colorado has thousands of water-right owners and millions of water users. Yet most people have no way to understand what water-right administration means for the rivers, farms, cities, golf courses, reservoirs, and communities they depend on.
      <br><br>
      The next generation of water management is not better measurement. It is translating complexity into understanding.
      </p>
    </div>
    """, unsafe_allow_html=True)

    
    st.markdown("""
    <div class="callout">
      <h2>A Weather Forecast for Water Rights</h2>
      <p>
      Not a legal opinion.<br>
      Not a replacement for the State Engineer.<br><br>
      A new way to understand how Colorado's rivers are likely to be administered in the weeks ahead.
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.info("Use the sidebar to switch to the Outlook Dashboard.")
    st.stop()


st.sidebar.title("Scenario")
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
raw_prob_series = pd.Series(prob, index=le.inverse_transform(np.arange(len(prob)))).reindex(REGIME_ORDER, fill_value=0)

current_regime = row.iloc[0]["regime"]
prob_series = apply_expert_probability_overlay(raw_prob_series, current_regime, selected_date).sort_values(ascending=False)

hist_pct = float(row.iloc[0]["historical_percentile"])
score = float(row.iloc[0]["score"])
priority = row.iloc[0].get("controlling_priority_date", "—")
structure = row.iloc[0].get("controlling_priority_structure", "—")
flow_cfs = float(row.iloc[0].get("flow_cfs", np.nan))
flow_pct = float(row.iloc[0].get("flow_doy_percentile", np.nan))
comps = comparable_years(model_df, selected_date, score)

public_label = PUBLIC_LABELS[current_regime]
public_explain = PUBLIC_EXPLAIN[current_regime]
most_likely = prob_series.index[0]
most_likely_public = PUBLIC_LABELS[most_likely]

st.markdown(f"""
<div class="hero">
  <div class="eyebrow">Historical-data prototype</div>
  <h1>Northern Colorado Water Right Outlook</h1>
  <p>Making Colorado water rights understandable — by translating river administration into a plain-English outlook.</p>
  <div class="hero-footer">
    <div class="small-muted">Selected date: {selected_date.strftime("%B %d, %Y")} · South Platte / Poudre framework</div>
    <div class="badge">Current condition: {public_label}</div>
  </div>
</div>
""", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="kicker">Today’s condition</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="big" style="color:{REGIME_COLORS[current_regime]};">{public_label}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="copy">{public_explain}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with c2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="kicker">How unusual is today?</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="big">{hist_pct:.0f}th percentile</div>', unsafe_allow_html=True)
    marker = max(0, min(100, hist_pct))
    st.markdown(f"""
    <div class="gauge-wrap">
      <div class="gauge-track"><div class="gauge-marker" style="left:calc({marker:.1f}% - 2px);"></div></div>
      <div class="gauge-labels"><span>Wet / available</span><span>Typical</span><span>Severe</span></div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with c3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="kicker">30-day outlook</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="big" style="color:{REGIME_COLORS[most_likely]};">{most_likely_public}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="copy">Most likely outcome: {prob_series.iloc[0]:.0%} probability.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("")

left, right = st.columns([1.08, .92], gap="large")
with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">What could the basin look like in 30 days?</div>', unsafe_allow_html=True)
    fig = go.Figure()
    for regime in REGIME_ORDER:
        public = PUBLIC_SHORT[regime]
        fig.add_trace(go.Bar(
            x=[prob_series.get(regime, 0)],
            y=[public],
            orientation="h",
            marker_color=REGIME_COLORS[regime],
            text=[f"{prob_series.get(regime, 0):.0%}"],
            textposition="outside",
            hovertemplate=f"{PUBLIC_LABELS[regime]}<br>%{{x:.0%}}<extra></extra>",
            showlegend=False,
        ))
    fig.update_layout(
        height=380,
        margin=dict(l=10, r=35, t=10, b=10),
        xaxis=dict(range=[0, 1], tickformat=".0%", gridcolor="rgba(255,255,255,.10)"),
        yaxis=dict(categoryorder="array", categoryarray=[PUBLIC_SHORT[r] for r in reversed(REGIME_ORDER)]),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#edf4f7", size=14),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('<div class="note">The displayed forecast includes an expert-rule overlay: if active administration is underway during irrigation season, Free River is removed as a 30-day outcome.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">What does this mean?</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="meaning-grid" style="grid-template-columns:1fr;">
      <div class="meaning-card"><strong>For the public</strong><br><span class="copy">This is like a weather forecast for water rights: not just how much water is in the river, but how the river is being administered.</span></div>
      <div class="meaning-card"><strong>For water-right owners</strong><br><span class="copy">The outlook gives a plain-English signal of whether administrative pressure is likely to ease, hold, or tighten.</span></div>
      <div class="meaning-card"><strong>For Northern Colorado</strong><br><span class="copy">The key issue is not only drought. It is whether priority administration becomes more restrictive.</span></div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("")
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="kicker">Current call signal</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="big" style="font-size:26px;">{priority}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="copy">{structure}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
with m2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="kicker">Canyon-mouth flow</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="big" style="font-size:30px;">{flow_cfs:,.0f} cfs</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="copy">Seasonal flow percentile: {flow_pct:.0f}%</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
with m3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="kicker">Comparable years</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="big" style="font-size:30px;">{", ".join(comps) if comps else "—"}</div>', unsafe_allow_html=True)
    st.markdown('<div class="copy">Based on recent administrative severity near this point in the season.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
with m4:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="kicker">Severity score</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="big" style="font-size:30px;">{score:.0f}/100</div>', unsafe_allow_html=True)
    st.markdown('<div class="copy">Higher means more restrictive water-right administration.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("### Administrative stress around selected date")
window = model_df[(model_df["date"] >= selected_date - pd.Timedelta(days=90)) & (model_df["date"] <= selected_date + pd.Timedelta(days=90))]
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=window["date"], y=window["score"], mode="lines",
    line=dict(color="#5ec9df", width=3),
    fill="tozeroy", fillcolor="rgba(94,201,223,.12)",
    hovertemplate="%{x|%b %d, %Y}<br>Severity: %{y}<extra></extra>",
))
fig.add_vline(x=selected_date, line_width=2, line_dash="dash", line_color="#ffffff")
fig.update_layout(
    height=340,
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
            name=PUBLIC_SHORT[regime],
            marker_color=REGIME_COLORS[regime],
            hovertemplate=f"{PUBLIC_LABELS[regime]}<br>%{{y}} days<extra></extra>",
        ))
fig.update_layout(
    barmode="stack",
    height=450,
    margin=dict(l=10, r=10, t=20, b=10),
    yaxis=dict(title="Days", gridcolor="rgba(255,255,255,.10)"),
    xaxis=dict(title="Year"),
    legend=dict(orientation="h", y=1.13),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#edf4f7"),
)
st.plotly_chart(fig, use_container_width=True)

with st.expander("About this prototype"):
    st.write("""
This historical-data prototype avoids live CDSS API calls so it deploys reliably.
The date selector simulates what the site would have shown on a selected historical date.

The public labels are simplified translations of the technical administrative regimes:
Free River, Mild Administration, Normal Administration, Senior Administration, and Exceptional Administration.
""")
    compare = pd.DataFrame({
        "Raw model probability": raw_prob_series.reindex(REGIME_ORDER),
        "Expert-adjusted probability": prob_series.reindex(REGIME_ORDER),
    })
    compare.index = [PUBLIC_LABELS[i] for i in compare.index]
    st.dataframe(compare.style.format("{:.1%}"), use_container_width=True)
    st.dataframe(row, use_container_width=True)

st.markdown('<div class="note">Not an official forecast, legal opinion, or substitute for CDSS/DWR records. Built as a public-education prototype.</div>', unsafe_allow_html=True)
