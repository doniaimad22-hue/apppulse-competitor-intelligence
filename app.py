"""
AppPulse - Early Warning System & Dynamic Competitor Intelligence for Mobile Apps
Live search -> scrape -> feature engineering -> model benchmark -> dashboard -> triage -> Gemini report
"""
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from google_play_scraper import Sort, reviews, search
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------------------
# Page + theme
# ----------------------------------------------------------------------------
st.set_page_config(page_title="AppPulse | Competitor Intelligence", page_icon="📡", layout="wide")

ACCENT = "#00E5A8"
COMP_COLORS = ["#7C6CFF", "#FF7A59", "#4DA3FF", "#FFC857"]

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: radial-gradient(1200px 600px at 10% -10%, #1b2450 0%, #0B1020 55%) fixed; }
section[data-testid="stSidebar"] { background: #0E1429; border-right: 1px solid #1f2a4d; }
.hero { padding: 28px 32px; border-radius: 20px; margin-bottom: 18px;
        background: linear-gradient(135deg, #131A2E 0%, #1a2350 100%); border: 1px solid #263262; }
.hero h1 { margin: 0; font-size: 2.1rem; font-weight: 800; letter-spacing: -0.5px; color: #fff; }
.hero h1 span { color: #00E5A8; }
.hero p { margin: 6px 0 0; color: #9fb0d9; font-size: 1.02rem; }
.step { background: #131A2E; border: 1px solid #263262; border-radius: 16px; padding: 18px 20px; height: 100%; }
.step b { color: #00E5A8; font-size: 1.4rem; }
.step h4 { margin: 4px 0; color: #fff; } .step p { color: #9fb0d9; margin: 0; font-size: .92rem; }
div[data-testid="stMetric"] { background: #131A2E; border: 1px solid #263262; padding: 16px 18px;
        border-radius: 16px; }
div[data-testid="stMetricValue"] { font-weight: 800; }
.banner { padding: 14px 18px; border-radius: 14px; margin: 10px 0 18px; font-weight: 500; }
.banner.good { background: rgba(0,229,168,.10); border: 1px solid #00E5A8; color: #b8ffe9; }
.banner.bad  { background: rgba(255,90,90,.10); border: 1px solid #ff5a5a; color: #ffd0d0; }
.risk { padding: 18px 22px; border-radius: 16px; font-size: 1.05rem; }
.risk.low { background: rgba(0,229,168,.10); border: 1px solid #00E5A8; }
.risk.med { background: rgba(255,200,87,.10); border: 1px solid #FFC857; }
.risk.high{ background: rgba(255,90,90,.12); border: 1px solid #ff5a5a; }
button[kind="primary"] { background: linear-gradient(90deg,#00E5A8,#00b4d8) !important; color:#06101f !important;
        font-weight: 700 !important; border: 0 !important; }
.stTabs [data-baseweb="tab"] { font-weight: 600; }
footer, #MainMenu { visibility: hidden; }
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
DEMO_APPS = {
    "Uber Eats": "com.ubercab.eats",
    "DoorDash": "com.dd.doordash",
    "Grubhub": "com.grubhub.android",
    "Deliveroo": "com.deliveroo.orderapp",
    "Just Eat": "com.justeat.app.uk",
}
COUNTRIES = {"United States": "us", "United Kingdom": "gb", "Canada": "ca", "Australia": "au", "Germany": "de", "France": "fr"}

CRASH_RX = r"crash|freez|froze|bug|glitch|error|broken|not working|doesn'?t work|won'?t (open|load|work)|stuck|fail|force clos"
ISSUES = {
    "Crashes & bugs": CRASH_RX,
    "Login & account": r"log ?in|sign ?in|password|account|verif|banned|locked|suspend",
    "Payments & refunds": r"refund|charge|payment|credit card|money|price|fee|overcharg|promo|coupon",
    "Delivery & orders": r"deliver|driver|late|cold|missing|wrong order|cancel|never arrived|courier",
    "Slow performance": r"slow|lag|loading|spinning|takes forever|hang",
}
FEATURES = ["text_len", "word_count", "exclam_count", "thumbs_up", "crash_kw_count", "caps_ratio", "hour"]
FEATURE_LABELS = {
    "text_len": "Review length", "word_count": "Word count", "exclam_count": "Exclamation marks",
    "thumbs_up": "Likes (thumbs up)", "crash_kw_count": "Critical-failure keywords",
    "caps_ratio": "ALL-CAPS ratio", "hour": "Hour posted",
}


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def get_api_key():
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY", "") or st.session_state.get("manual_key", "")


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_reviews(app_id, n, lang, country):
    data, _ = reviews(app_id, lang=lang, country=country, sort=Sort.NEWEST, count=n)
    df = pd.DataFrame(data)
    if df.empty:
        return df
    return df[["reviewId", "content", "score", "thumbsUpCount", "at"]]


def clean_text(t):
    t = re.sub(r"http\S+", " ", str(t))
    return re.sub(r"\s+", " ", t).strip()


def text_features(text, thumbs=0, hour=12):
    t = clean_text(text)
    letters = [c for c in t if c.isalpha()]
    return {
        "text_len": len(t),
        "word_count": len(t.split()),
        "exclam_count": t.count("!"),
        "thumbs_up": thumbs,
        "crash_kw_count": len(re.findall(CRASH_RX, t.lower())),
        "caps_ratio": (sum(c.isupper() for c in letters) / len(letters)) if letters else 0.0,
        "hour": hour,
    }


def build_dataset(frames):
    df = pd.concat(frames, ignore_index=True).drop_duplicates("reviewId")
    df = df.dropna(subset=["content", "score"])
    df["content"] = df["content"].map(clean_text)
    df = df[df["content"].str.len() > 0].copy()
    df["clean"] = df["content"].str.lower()
    df["at"] = pd.to_datetime(df["at"])
    feats = pd.DataFrame(
        [text_features(t, th, h) for t, th, h in zip(df["content"], df["thumbsUpCount"], df["at"].dt.hour)],
        index=df.index,
    )
    df = pd.concat([df, feats], axis=1)
    df["critical"] = (df["score"] <= 2).astype(int)
    return df.rename(columns={"thumbsUpCount": "likes"}).reset_index(drop=True)


def train_models(df):
    y = df["critical"]
    if y.nunique() < 2 or y.value_counts().min() < 5:
        return None
    X = df[FEATURES]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1
        ),
    }
    rows, cms = [], {}
    for name, m in models.items():
        m.fit(Xtr, ytr)
        p = m.predict(Xte)
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(yte, p),
            "Precision": precision_score(yte, p, zero_division=0),
            "Recall": recall_score(yte, p, zero_division=0),
        })
        cms[name] = confusion_matrix(yte, p, labels=[0, 1])
    table = pd.DataFrame(rows)
    best = table.sort_values(["Recall", "Precision"], ascending=False).iloc[0]["Model"]
    return {"models": models, "table": table, "cms": cms, "best": best, "n_test": len(yte)}


def feature_importance(res):
    m = res["models"][res["best"]]
    if hasattr(m, "feature_importances_"):
        imp = m.feature_importances_
    else:
        imp = np.abs(m[-1].coef_[0])
        imp = imp / imp.sum()
    return pd.DataFrame({"Feature": [FEATURE_LABELS[f] for f in FEATURES], "Importance": imp}).sort_values("Importance")


def app_summary(df):
    return (
        df.groupby("app_name")
        .agg(reviews=("score", "size"), avg_rating=("score", "mean"),
             critical_pct=("critical", lambda x: 100 * x.mean()), avg_likes=("likes", "mean"))
        .reset_index()
    )


def issue_table(df):
    crit = df[df["critical"] == 1]
    rows = []
    for app, g in crit.groupby("app_name"):
        for issue, rx in ISSUES.items():
            rows.append({"app_name": app, "Issue": issue, "Share of critical reviews (%)": 100 * g["clean"].str.contains(rx, regex=True).mean()})
    return pd.DataFrame(rows)


def style(fig, h=380):
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=h, margin=dict(l=10, r=10, t=50, b=10), font=dict(family="Inter", color="#dbe4ff"),
        legend_title_text="",
    )
    fig.update_xaxes(gridcolor="#1f2a4d")
    fig.update_yaxes(gridcolor="#1f2a4d")
    return fig


def color_map(primary, apps):
    cm, i = {primary: ACCENT}, 0
    for a in apps:
        if a != primary:
            cm[a] = COMP_COLORS[i % len(COMP_COLORS)]
            i += 1
    return cm


def risk_level(p):
    if p >= 0.65:
        return "high", "🔴 HIGH RISK", "Escalate to the engineering lead and reply publicly within 24h."
    if p >= 0.35:
        return "med", "🟠 MEDIUM RISK", "Route to support for follow-up and tag the related feature owner."
    return "low", "🟢 LOW RISK", "No escalation needed. Monitor for similar patterns."

def recommendations(text):
    t = text.lower()
    recs = [f"• **{k}** detected: assign to the owning squad and check the last release." for k, rx in ISSUES.items() if re.search(rx, t)]
    return recs or ["• No specific failure category detected: review manually and respond with an apology + support link."]
   

def offline_report(df, summ, issues, primary):
    comps = summ[summ.app_name != primary]
    me = summ[summ.app_name == primary].iloc[0]
    lines = [f"EXECUTIVE REPORT (offline mode) - {primary}", f"Generated: {datetime.now():%Y-%m-%d %H:%M}", ""]
    lines.append(f"{primary}: avg rating {me.avg_rating:.2f}, critical share {me.critical_pct:.1f}% over {int(me.reviews)} reviews.")
    lines.append(f"Competitor average critical share: {comps.critical_pct.mean():.1f}%.\n")
    for app, g in issues.groupby("app_name"):
        top = g.sort_values("Share of critical reviews (%)", ascending=False).iloc[0]
        lines.append(f"- {app}: top complaint = {top['Issue']} ({top['Share of critical reviews (%)']:.0f}% of critical reviews)")
    lines.append("\nAdd a Gemini API key to get the full AI-written strategic report.")
    return "\n".join(lines)


def gemini_report(df, summ, issues, primary, key):
    from google import genai

    ctx = [summ.round(2).to_string(index=False), "\nIssue shares among critical reviews (%):", issues.round(1).to_string(index=False)]
    for app, g in df[df.critical == 1].groupby("app_name"):
        top = g.sort_values("likes", ascending=False).head(3)
        ctx.append(f"\nMost-liked critical reviews for {app}:")
        ctx += [f"- ({r.score}★, {r.likes} likes) {r.content[:220]}" for r in top.itertuples()]
    prompt = f"""You are a senior product-intelligence analyst. Our app is "{primary}".
Using ONLY the data below (live Google Play reviews), write an executive report in English with these sections:
1. Executive Summary (3 bullets)
2. Competitor Technical Weaknesses (per competitor, evidence-based)
3. Our Weaknesses vs. Competitors
4. Top 5 Development Priorities (ordered, each with expected impact and effort: S/M/L)
5. 30-day Action Plan
Be concrete and concise. No fluff.

DATA:
{chr(10).join(ctx)}"""
    from google.genai import types

    client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=60_000))
    last_err = None
    for model in ("gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash", "gemini-flash-latest", "gemini-3.1-flash-lite"):
        try:
            return client.models.generate_content(model=model, contents=prompt).text
        except Exception as e:  # busy / unavailable -> try the next model
            last_err = e
    raise last_err


# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------
st.session_state.setdefault("catalog", dict(DEMO_APPS))
st.session_state.setdefault("hits", [])
st.session_state.setdefault("result", None)
st.session_state.setdefault("report", None)

# ----------------------------------------------------------------------------
# Sidebar - search & selection (Requirement 1)
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 📡 AppPulse")
    st.caption("Early-warning & competitor intelligence")
    st.markdown("#### 1 · Market")
    country_name = st.selectbox("Store country", list(COUNTRIES), index=0)
    country, lang = COUNTRIES[country_name], "en"

    st.markdown("#### 2 · Search Google Play")
    q = st.text_input("App name", placeholder="e.g. Zomato, Glovo, Wolt, Careem")
    if st.button("🔎 Search", use_container_width=True) and q.strip():
        try:
            hits = search(q.strip(), lang=lang, country=country, n_hits=8)
            st.session_state.hits = hits
            for h in hits:
                st.session_state.catalog.setdefault(h["title"], h["appId"])
        except Exception as e:
            st.error(f"Search failed: {e}")
    if st.session_state.hits:
        st.caption("Matches added to the list below: " + " · ".join(h["title"] for h in st.session_state.hits[:6]))

    st.markdown("#### 3 · Choose apps")
    names = list(st.session_state.catalog)
    primary = st.selectbox("Our app", names, index=names.index("Uber Eats") if "Uber Eats" in names else 0)
    opts = [n for n in names if n != primary]
    default = [d for d in ["DoorDash", "Grubhub", "Deliveroo"] if d in opts]
    comps = st.multiselect("Competitors (2-4)", opts, default=default, max_selections=4)
    n_rev = st.slider("Reviews per app", 200, 3000, 600, 100)
    go_btn = st.button("🚀 Run analysis", type="primary", use_container_width=True, disabled=len(comps) < 2)
    if len(comps) < 2:
        st.warning("Pick at least 2 competitors.")

    st.markdown("---")
    if not get_api_key():
        st.text_input("Gemini API key (optional)", type="password", key="manual_key")

# ----------------------------------------------------------------------------
# Pipeline (Requirements 2 & 3)
# ----------------------------------------------------------------------------
if go_btn:
    frames, warns = [], []
    bar = st.progress(0.0, text="Starting…")
    all_apps = [primary] + comps
    for i, name in enumerate(all_apps):
        bar.progress(i / len(all_apps), text=f"Scraping live reviews: {name}")
        try:
            d = fetch_reviews(st.session_state.catalog[name], n_rev, lang, country)
        except Exception as e:
            warns.append(f"{name}: scraping failed ({e})")
            continue
        if d.empty:
            warns.append(f"{name}: no reviews found in this store/country - skipped.")
            continue
        d = d.copy()
        d["app_name"] = name
        frames.append(d)
    bar.progress(0.9, text="Engineering features & training models…")
    if primary not in [f["app_name"].iloc[0] for f in frames] or len(frames) < 2:
        bar.empty()
        st.session_state.result = None
        st.error("Not enough data: we need reviews for our app and at least one competitor. Try other apps or another country.")
        for w in warns:
            st.warning(w)
    else:
        df = build_dataset(frames)
        res = train_models(df)
        st.session_state.result = {"df": df, "res": res, "primary": primary, "warns": warns, "country": country_name}
        st.session_state.report = None
        bar.empty()

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
st.markdown(
    """<div class="hero"><h1>App<span>Pulse</span> · Early-Warning & Competitor Intelligence</h1>
<p>Search any Google Play app, benchmark it against rivals, detect critical complaints with ML, and get an AI executive report.</p></div>""",
    unsafe_allow_html=True,
)

R = st.session_state.result
if not R:
    c = st.columns(4)
    steps = [("1", "Search", "Find any app live on Google Play."), ("2", "Scrape", "Pull the newest reviews for every app."),
             ("3", "Benchmark", "Logistic Regression vs Random Forest."), ("4", "Report", "GeminiFlash executive brief.")]
    for col, (n, t, d) in zip(c, steps):
        col.markdown(f'<div class="step"><b>{n}</b><h4>{t}</h4><p>{d}</p></div>', unsafe_allow_html=True)
    st.info("👈 Use the sidebar to pick your app and competitors, then press **Run analysis**.")
    st.stop()

df, res, primary = R["df"], R["res"], R["primary"]
for w in R["warns"]:
    st.warning(w)
apps = df["app_name"].unique().tolist()
cmap = color_map(primary, apps)
summ = app_summary(df)
issues = issue_table(df)

t1, t2, t3, t4, t5 = st.tabs(["📊 Dashboard", "🤖 Model Benchmark", "🚨 Live Triage", "🧠 Executive Report", "🗂 Data"])

# ---- Dashboard (Requirement 4) --------------------------------------------
with t1:
    cols = st.columns(len(summ))
    me = summ[summ.app_name == primary].iloc[0]
    comp_avg = summ[summ.app_name != primary].critical_pct.mean()
    for col, r in zip(cols, summ.itertuples()):
        label = f"{'⭐ ' if r.app_name == primary else ''}{r.app_name}"
        col.metric(label, f"{r.critical_pct:.1f}% critical", f"★ {r.avg_rating:.2f} · {int(r.reviews)} reviews", delta_color="off")
    diff = me.critical_pct - comp_avg
    cls = "good" if diff <= 0 else "bad"
    word = "below" if diff <= 0 else "above"
    st.markdown(
        f'<div class="banner {cls}"><b>{primary}</b> has a critical-complaint rate <b>{abs(diff):.1f} pts {word}</b> the competitor average ({comp_avg:.1f}%).</div>',
        unsafe_allow_html=True,
    )
    a, b = st.columns(2)
    f1 = px.bar(summ, x="app_name", y="avg_rating", color="app_name", color_discrete_map=cmap, text=summ.avg_rating.round(2), title="Average rating")
    f1.update_yaxes(range=[0, 5]); f1.update_layout(showlegend=False, xaxis_title="", yaxis_title="")
    a.plotly_chart(style(f1), use_container_width=True)
    f2 = px.bar(summ, x="app_name", y="critical_pct", color="app_name", color_discrete_map=cmap, text=summ.critical_pct.round(1), title="Critical complaints (≤ 2★, %)")
    f2.update_layout(showlegend=False, xaxis_title="", yaxis_title="")
    b.plotly_chart(style(f2), use_container_width=True)
    c, d = st.columns(2)
    f3 = px.bar(issues, x="Issue", y="Share of critical reviews (%)", color="app_name", barmode="group", color_discrete_map=cmap, title="What are users angry about?")
    f3.update_layout(xaxis_title="")
    c.plotly_chart(style(f3), use_container_width=True)
    dist = df.groupby(["app_name", "score"]).size().reset_index(name="n")
    dist["pct"] = 100 * dist["n"] / dist.groupby("app_name")["n"].transform("sum")
    f4 = px.bar(dist, x="app_name", y="pct", color=dist["score"].astype(str), title="Rating distribution (%)",
                color_discrete_sequence=["#ff5a5a", "#ff9f5a", "#ffd25a", "#9be564", "#00E5A8"],
                category_orders={"color": ["1", "2", "3", "4", "5"]})
    f4.update_layout(xaxis_title="", yaxis_title="")
    d.plotly_chart(style(f4), use_container_width=True)

# ---- Benchmark (Requirement 3) --------------------------------------------
with t2:
    if res is None:
        st.error("Not enough critical vs. non-critical reviews to train models. Increase reviews per app or choose other apps.")
    else:
        st.markdown(f"**Train/test split:** 80/20 stratified · `random_state=42` · test size = {res['n_test']} reviews · class_weight = balanced")
        tbl = res["table"].copy()
        st.dataframe(
            tbl.style.format({"Accuracy": "{:.3f}", "Precision": "{:.3f}", "Recall": "{:.3f}"}).highlight_max(subset=["Accuracy", "Precision", "Recall"], color="#0b5d49"),
            use_container_width=True, hide_index=True,
        )
        st.markdown(f'<div class="banner good">🏆 <b>{res["best"]}</b> selected automatically: highest <b>Recall</b>. '
                    'Missing a real critical complaint costs more than a false alarm, so we optimise for Recall over Accuracy.</div>', unsafe_allow_html=True)
        cc = st.columns(2)
        for col, (name, cm) in zip(cc, res["cms"].items()):
            fig = px.imshow(cm, text_auto=True, x=["Not critical", "Critical"], y=["Not critical", "Critical"],
                            color_continuous_scale=["#131A2E", "#00E5A8"], title=f"Confusion matrix · {name}")
            fig.update_layout(xaxis_title="Predicted", yaxis_title="Actual", coloraxis_showscale=False)
            col.plotly_chart(style(fig, 340), use_container_width=True)
        imp = feature_importance(res)
        fi = px.bar(imp, x="Importance", y="Feature", orientation="h", title=f"What drives negative reviews? ({res['best']})")
        fi.update_traces(marker_color=ACCENT)
        st.plotly_chart(style(fi, 360), use_container_width=True)

# ---- Live triage (Requirement 4) ------------------------------------------
with t3:
    if res is None:
        st.error("Model unavailable for this selection.")
    else:
        st.markdown(f"Test any new review against the **{res['best']}** model.")
        with st.form("triage"):
            txt = st.text_area("Review text", height=130, placeholder="The app keeps crashing every time I try to pay!!! Support never answers.")
            c1, c2 = st.columns(2)
            likes = c1.number_input("Likes", 0, 10000, 0)
            hour = c2.slider("Hour posted", 0, 23, datetime.now().hour)
            sub = st.form_submit_button("Analyze risk", type="primary")
        if sub:
            if not txt.strip():
                st.warning("Please type a review first.")
            else:
                X = pd.DataFrame([text_features(txt, likes, hour)])[FEATURES]
                p = float(res["models"][res["best"]].predict_proba(X)[0, 1])
                cls, label, action = risk_level(p)
                g, m = st.columns([1, 1.3])
                gauge = go.Figure(go.Indicator(
                    mode="gauge+number", value=p * 100, number={"suffix": "%"},
                    gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#fff"},
                           "steps": [{"range": [0, 35], "color": "#0b5d49"}, {"range": [35, 65], "color": "#8a6d1c"}, {"range": [65, 100], "color": "#8a2b2b"}]},
                    title={"text": "Probability of critical complaint"}))
                g.plotly_chart(style(gauge, 300), use_container_width=True)
                m.markdown(f'<div class="risk {cls}"><b>{label}</b><br>{action}</div>', unsafe_allow_html=True)
                m.markdown("**Suggested actions**")
                for r in recommendations(txt):
                    m.markdown(r)
                if p >= 0.65:
                    m.markdown("• Open a P1 ticket and notify the product manager.")
                other = [n for n in res["models"] if n != res["best"]][0]
                po = float(res["models"][other].predict_proba(X)[0, 1])
                m.caption(f"Cross-check · {other}: {po * 100:.1f}%")

# ---- Executive report (Requirement 5) --------------------------------------
with t4:
    key = get_api_key()
    st.markdown("Generate a competitor-comparison brief with **Gemini Flash**.")
    if st.button("✨ Generate executive report", type="primary"):
        with st.spinner("Gemini is analysing the data…"):
            try:
                if not key:
                    raise ValueError("no key")
                st.session_state.report = gemini_report(df, summ, issues, primary, key)
            except Exception as e:
                st.session_state.report = offline_report(df, summ, issues, primary)
                st.warning("Gemini unavailable (check your API key); showing an offline summary instead." if key else "No Gemini key found; showing an offline summary instead.")
    if st.session_state.report:
        st.markdown(st.session_state.report)
        st.download_button("⬇️ Download report (TXT)", st.session_state.report, "executive_report.txt", "text/plain")

# ---- Data ------------------------------------------------------------------
with t5:
    show = df[["app_name", "score", "critical", "likes", "text_len", "word_count", "exclam_count", "crash_kw_count", "caps_ratio", "hour", "at", "content"]]
    st.dataframe(show, use_container_width=True, height=420)
    st.download_button("⬇️ Download cleaned data (CSV)", show.to_csv(index=False).encode("utf-8"), "cleaned_reviews.csv", "text/csv")
