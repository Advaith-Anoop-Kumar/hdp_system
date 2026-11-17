from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE = Path(__file__).parent

st.set_page_config(page_title="HeartWise | Heart Disease Risk", page_icon="❤️", layout="wide")

# ------------------------------------------------------------------ styling
st.markdown("""
<style>
.hero{background:linear-gradient(120deg,#FF4B6E 0%,#FF8A5B 55%,#FFC857 100%);
      padding:28px 34px;border-radius:22px;color:white;margin-bottom:18px;
      box-shadow:0 8px 24px rgba(255,75,110,.28)}
.hero h1{margin:0;font-size:2.4rem;color:white;padding:0}
.hero p{margin:6px 0 0;font-size:1.05rem;opacity:.96}
.tip{padding:12px 16px;border-radius:14px;margin-bottom:10px;color:#2B2D42}
.card{background:white;border-radius:18px;padding:16px 20px;
      box-shadow:0 4px 16px rgba(43,45,66,.08);margin-bottom:14px}
.pill{display:block;text-align:center;padding:12px;border-radius:14px;
      font-weight:800;font-size:1.25rem;color:white;margin-bottom:6px}
div[data-testid="stMetric"]{background:white;padding:14px 18px;border-radius:16px;
      box-shadow:0 4px 14px rgba(43,45,66,.08)}
.stButton>button{border-radius:12px;font-weight:600}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------ constants
FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"]

LABELS = {
    "sex": {1: "Male", 0: "Female"},
    "cp": {1: "Typical angina", 2: "Atypical angina", 3: "Non-anginal pain", 4: "Asymptomatic"},
    "fbs": {0: "No", 1: "Yes"},
    "restecg": {0: "Normal", 1: "ST-T abnormality", 2: "LV hypertrophy"},
    "exang": {0: "No", 1: "Yes"},
    "slope": {1: "Upsloping", 2: "Flat", 3: "Downsloping"},
    "thal": {3: "Normal", 6: "Fixed defect", 7: "Reversible defect"},
}

DEFAULTS = dict(age=55, sex=1, cp=4, trestbps=130, chol=240, fbs=0, restecg=0,
                thalach=150, exang=0, oldpeak=1.0, slope=2, ca=0, thal=3)

PRESETS = {
    "🟢 Healthy 30-year-old": dict(age=30, sex=0, cp=3, trestbps=115, chol=180, fbs=0, restecg=0,
                                   thalach=180, exang=0, oldpeak=0.0, slope=1, ca=0, thal=3),
    "🟡 Borderline 52-year-old": dict(age=52, sex=1, cp=2, trestbps=135, chol=250, fbs=0, restecg=1,
                                      thalach=150, exang=0, oldpeak=1.2, slope=2, ca=1, thal=3),
    "🔴 High-risk 62-year-old": dict(age=62, sex=1, cp=4, trestbps=150, chol=290, fbs=1, restecg=2,
                                     thalach=120, exang=1, oldpeak=2.8, slope=2, ca=3, thal=7),
}

GREEN, YELLOW, RED, BLUE = "#2ECC71", "#F5B301", "#FF4B6E", "#4361EE"

for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def apply_preset(values):
    st.session_state.update(values)


# ------------------------------------------------------------------ loaders
@st.cache_resource
def load_model():
    return joblib.load(BASE / "models" / "model.joblib")


@st.cache_data
def load_data():
    path = BASE / "data" / "raw" / "heart.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["Status"] = df["target"].map({0: "Healthy", 1: "Disease"})
    return df


model = load_model()
data = load_data()

# ------------------------------------------------------------------ helpers
def risk_level(p):
    if p < 0.30:
        return "Low Risk", GREEN
    if p < 0.60:
        return "Moderate Risk", YELLOW
    return "High Risk", RED


def gauge(p):
    _, color = risk_level(p)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=p * 100,
        number={"suffix": "%", "font": {"size": 46, "color": "#2B2D42"}},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": color, "thickness": 0.35},
               "bgcolor": "white",
               "steps": [{"range": [0, 30], "color": "#D8F5E3"},
                         {"range": [30, 60], "color": "#FFF1C2"},
                         {"range": [60, 100], "color": "#FFD6DE"}]}))
    fig.update_layout(height=250, margin=dict(l=25, r=25, t=15, b=0), paper_bgcolor="rgba(0,0,0,0)")
    return fig


RADAR = {"age": "Age", "trestbps": "Blood pressure", "chol": "Cholesterol",
         "thalach": "Max heart rate", "oldpeak": "ST depression"}


def radar(patient, df):
    cols = list(RADAR)
    lo, hi = df[cols].min(), df[cols].max()

    def norm(s):
        return ((s - lo) / (hi - lo)).clip(0, 1)

    series = [
        ("Healthy average", norm(df[df.target == 0][cols].mean()), GREEN),
        ("Disease average", norm(df[df.target == 1][cols].mean()), RED),
        ("You", norm(pd.Series({c: patient[c] for c in cols})), BLUE),
    ]
    theta = list(RADAR.values())
    fig = go.Figure()
    for name, s, color in series:
        r = list(s.values)
        fig.add_trace(go.Scatterpolar(
            r=r + [r[0]], theta=theta + [theta[0]], name=name,
            line=dict(color=color, width=4 if name == "You" else 2),
            fill="toself" if name == "You" else None,
            fillcolor="rgba(67,97,238,0.18)" if name == "You" else None))
    fig.update_layout(height=340, margin=dict(l=40, r=40, t=20, b=20),
                      polar=dict(radialaxis=dict(visible=False, range=[0, 1])),
                      legend=dict(orientation="h", y=-0.1), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def tips_for(p):
    t = []
    if p["chol"] >= 240:
        t.append(("high", "🧈", "High cholesterol", "Above 240 mg/dl. A heart-healthy diet and regular exercise could help."))
    elif p["chol"] >= 200:
        t.append(("mid", "🧈", "Borderline cholesterol", "Between 200 and 239 mg/dl. Worth keeping an eye on."))
    if p["trestbps"] >= 140:
        t.append(("high", "🩸", "Elevated blood pressure", "Resting BP is 140 or above. Consider monitoring it and talking to a doctor."))
    elif p["trestbps"] >= 130:
        t.append(("mid", "🩸", "Slightly raised blood pressure", "130 to 139 is considered stage 1 hypertension range."))
    if p["fbs"] == 1:
        t.append(("mid", "🍬", "High fasting blood sugar", "Consider a diabetes screening."))
    if p["exang"] == 1:
        t.append(("high", "🏃", "Exercise-induced chest pain", "Chest pain during exertion deserves a medical evaluation."))
    if p["oldpeak"] >= 2:
        t.append(("high", "📉", "Large ST depression", "A big ST drop during exercise can indicate reduced blood flow."))
    max_hr = 220 - p["age"]
    if p["thalach"] < 0.7 * max_hr:
        t.append(("mid", "💓", "Low peak heart rate", f"Peak HR is below 70% of the age-predicted max (about {max_hr} bpm)."))
    if p["ca"] >= 2:
        t.append(("high", "🫀", "Multiple vessels flagged", "Two or more major vessels showed on fluoroscopy."))
    if p["thal"] in (6, 7):
        t.append(("high", "🧪", "Abnormal thalassemia result", "A fixed or reversible defect appears on the stress test."))
    if not t:
        t.append(("good", "✅", "No major flags", "Nothing here stands out. Keep up regular checkups and activity."))
    return t


def tip_card(level, icon, title, text):
    bg, border = {"high": ("#FFE3E8", RED), "mid": ("#FFF4D6", YELLOW),
                  "good": ("#DDF7E8", GREEN)}[level]
    st.markdown(f'<div class="tip" style="background:{bg};border-left:6px solid {border}">'
                f'<b>{icon} {title}</b><br>{text}</div>', unsafe_allow_html=True)


def importance_df(m):
    try:
        prep, est = m.named_steps["prep"], m.named_steps["model"]
        names = [n.split("__", 1)[1].replace("_", " = ", 1) for n in prep.get_feature_names_out()]
        if hasattr(est, "coef_"):
            vals, signed = est.coef_[0], True
        elif hasattr(est, "feature_importances_"):
            vals, signed = est.feature_importances_, False
        else:
            return None, False
        out = pd.DataFrame({"feature": names, "value": vals})
        out["abs"] = out["value"].abs()
        return out.sort_values("abs", ascending=False).head(12).iloc[::-1], signed
    except Exception:
        return None, False


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("## ❤️ HeartWise")
    st.caption("Try a ready-made patient, then tweak the values.")
    for name, values in PRESETS.items():
        st.button(name, on_click=apply_preset, args=(values,), use_container_width=True)
    st.button("↺ Reset to defaults", on_click=apply_preset, args=(DEFAULTS,), use_container_width=True)
    st.divider()
    st.caption("⚠️ Educational demo only. Not medical advice.")

# ------------------------------------------------------------------ hero + stats
st.markdown("""
<div class="hero">
  <h1>❤️ HeartWise</h1>
  <p>Heart disease risk estimator. Move any control and the prediction updates instantly.</p>
</div>
""", unsafe_allow_html=True)

if data is not None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Patients in dataset", f"{len(data)}")
    c2.metric("With heart disease", f"{data.target.mean() * 100:.0f}%")
    c3.metric("Average age", f"{data.age.mean():.0f} yrs")
    c4.metric("Average cholesterol", f"{data.chol.mean():.0f} mg/dl")

tab_check, tab_explore, tab_whatif, tab_model, tab_about = st.tabs(
    ["🩺 Risk Check", "📊 Explore Data", "🔮 What-If", "🧠 Model Insights", "ℹ️ About"])

# ------------------------------------------------------------------ tab 1: risk check
with tab_check:
    left, right = st.columns([1.1, 1], gap="large")

    with left:
        with st.expander("👤 Patient basics", expanded=True):
            st.slider("Age", 20, 90, key="age")
            st.radio("Sex", [1, 0], format_func=LABELS["sex"].get, horizontal=True, key="sex")
        with st.expander("🫀 Heart and vitals", expanded=True):
            st.selectbox("Chest pain type", [1, 2, 3, 4], format_func=LABELS["cp"].get, key="cp")
            st.slider("Resting blood pressure (mm Hg)", 80, 220, key="trestbps")
            st.slider("Cholesterol (mg/dl)", 100, 600, key="chol")
            st.slider("Max heart rate achieved (bpm)", 60, 220, key="thalach")
            st.radio("Fasting blood sugar above 120 mg/dl", [0, 1],
                     format_func=LABELS["fbs"].get, horizontal=True, key="fbs")
        with st.expander("🧪 Tests and scans", expanded=True):
            st.selectbox("Resting ECG", [0, 1, 2], format_func=LABELS["restecg"].get, key="restecg")
            st.radio("Exercise-induced angina", [0, 1],
                     format_func=LABELS["exang"].get, horizontal=True, key="exang")
            st.slider("ST depression (oldpeak)", 0.0, 7.0, step=0.1, key="oldpeak")
            st.selectbox("Slope of peak exercise ST", [1, 2, 3], format_func=LABELS["slope"].get, key="slope")
            st.select_slider("Major vessels colored (0 to 3)", options=[0, 1, 2, 3], key="ca")
            st.selectbox("Thalassemia", [3, 6, 7], format_func=LABELS["thal"].get, key="thal")

    patient = {f: st.session_state[f] for f in FEATURES}
    prob = float(model.predict_proba(pd.DataFrame([patient])[FEATURES])[0, 1])
    label, color = risk_level(prob)

    with right:
        st.markdown(f'<div class="pill" style="background:{color}">{label} of Heart Disease</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(gauge(prob), key="gauge")
        st.progress(min(prob, 1.0), text=f"Estimated risk probability: {prob:.3f}")

        st.subheader("Personalized suggestions")
        tips = tips_for(patient)
        for level, icon, title, text in tips:
            tip_card(level, icon, title, text)

        if data is not None:
            st.subheader("You vs. the dataset")
            st.plotly_chart(radar(patient, data), key="radar")

        report = (f"HeartWise report\nRisk: {label} ({prob:.1%})\n\nInputs:\n"
                  + "\n".join(f"  {k}: {v}" for k, v in patient.items())
                  + "\n\nNotes:\n" + "\n".join(f"  - {t[2]}: {t[3]}" for t in tips)
                  + "\n\nEducational use only. Not medical advice.\n")
        st.download_button("⬇️ Download report", report, file_name="heartwise_report.txt")

# ------------------------------------------------------------------ tab 2: explore
with tab_explore:
    if data is None:
        st.warning("data/raw/heart.csv not found. Run download_data.py to enable this tab.")
    else:
        cmap = {"Healthy": GREEN, "Disease": RED}
        st.subheader("Compare healthy vs. disease patients")
        numeric = {"Age": "age", "Resting BP": "trestbps", "Cholesterol": "chol",
                   "Max heart rate": "thalach", "ST depression": "oldpeak"}
        pick = st.selectbox("Pick a measurement", list(numeric))
        col = numeric[pick]
        fig = px.histogram(data, x=col, color="Status", barmode="overlay", nbins=25,
                           opacity=0.65, color_discrete_map=cmap)
        fig.add_vline(x=patient[col], line_dash="dash", line_color=BLUE,
                      annotation_text="You", annotation_position="top")
        fig.update_layout(height=340, margin=dict(t=30, b=10), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, key="hist")

        a, b = st.columns(2)
        with a:
            st.subheader("Age vs. max heart rate")
            sc = px.scatter(data, x="age", y="thalach", color="Status",
                            color_discrete_map=cmap, opacity=0.75,
                            labels={"age": "Age", "thalach": "Max heart rate"})
            sc.add_trace(go.Scatter(x=[patient["age"]], y=[patient["thalach"]], mode="markers",
                                    name="You", marker=dict(size=18, color=BLUE, symbol="star",
                                                            line=dict(width=2, color="white"))))
            sc.update_layout(height=360, margin=dict(t=10), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(sc, key="scatter")
        with b:
            st.subheader("Disease rate by chest pain type")
            rate = data.groupby("cp")["target"].mean().reset_index()
            rate["Chest pain"] = rate["cp"].map(LABELS["cp"])
            bar = px.bar(rate, x="Chest pain", y="target", color="target",
                         color_continuous_scale=["#FFD6DE", RED],
                         labels={"target": "Share with disease"})
            bar.update_layout(height=360, margin=dict(t=10), coloraxis_showscale=False,
                              yaxis_tickformat=".0%", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(bar, key="cpbar")

        st.subheader("What moves together?")
        corr = data[FEATURES + ["target"]].corr()
        heat = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
        heat.update_layout(height=480, margin=dict(t=10), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(heat, key="heat")

        with st.expander("See the raw data"):
            st.dataframe(data.drop(columns="Status"), height=300)

# ------------------------------------------------------------------ tab 3: what-if
with tab_whatif:
    st.subheader("How would risk change if one thing changed?")
    st.caption("Everything else stays as set on the Risk Check tab.")
    options = {"Age": ("age", 20, 90), "Resting BP": ("trestbps", 80, 220),
               "Cholesterol": ("chol", 100, 600), "Max heart rate": ("thalach", 60, 220),
               "ST depression": ("oldpeak", 0.0, 7.0)}
    choice = st.radio("Vary:", list(options), horizontal=True)
    var, lo, hi = options[choice]
    grid = np.linspace(lo, hi, 50)
    rows = pd.DataFrame([patient] * len(grid))
    rows[var] = grid
    probs = model.predict_proba(rows[FEATURES])[:, 1]

    fig = go.Figure()
    fig.add_hrect(y0=0, y1=0.30, fillcolor="#D8F5E3", opacity=0.6, line_width=0)
    fig.add_hrect(y0=0.30, y1=0.60, fillcolor="#FFF1C2", opacity=0.6, line_width=0)
    fig.add_hrect(y0=0.60, y1=1.0, fillcolor="#FFD6DE", opacity=0.6, line_width=0)
    fig.add_trace(go.Scatter(x=grid, y=probs, mode="lines", line=dict(color=BLUE, width=4), name="Risk"))
    fig.add_trace(go.Scatter(x=[patient[var]], y=[prob], mode="markers", name="Current",
                             marker=dict(size=16, color=RED, line=dict(width=2, color="white"))))
    fig.update_layout(height=400, yaxis=dict(range=[0, 1], tickformat=".0%", title="Estimated risk"),
                      xaxis_title=choice, margin=dict(t=20), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, key="whatif")

    target = st.slider("Compare with a different value", float(lo), float(hi), float(patient[var]))
    alt = dict(patient)
    alt[var] = target
    alt_p = float(model.predict_proba(pd.DataFrame([alt])[FEATURES])[0, 1])
    m1, m2, m3 = st.columns(3)
    m1.metric("Current risk", f"{prob:.1%}")
    m2.metric(f"With {choice} = {target:g}", f"{alt_p:.1%}", delta=f"{(alt_p - prob) * 100:+.1f} pts",
              delta_color="inverse")
    m3.metric("Risk level", risk_level(alt_p)[0])

# ------------------------------------------------------------------ tab 4: model insights
with tab_model:
    est_name = type(model.named_steps["model"]).__name__
    st.subheader(f"What the model pays attention to ({est_name})")
    imp, signed = importance_df(model)
    if imp is None:
        st.info("Feature importance isn't available for this model type.")
    else:
        if signed:
            colors = [RED if v > 0 else GREEN for v in imp["value"]]
            st.caption("Red bars push risk up, green bars push it down. Longer means stronger.")
        else:
            colors = [BLUE] * len(imp)
            st.caption("Longer bars mean the feature matters more to the model.")
        fig = go.Figure(go.Bar(x=imp["value"], y=imp["feature"], orientation="h", marker_color=colors))
        fig.update_layout(height=460, margin=dict(t=10), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, key="importance")
    st.info("Features written as name = value (for example cp = 4.0) are categories. "
            "The model learned a separate weight for each one.")

# ------------------------------------------------------------------ tab 5: about
with tab_about:
    st.subheader("About this project")
    st.write("HeartWise estimates heart disease risk from 13 clinical measurements using a model "
             "trained on the UCI Heart Disease dataset (Cleveland, 303 patients). It was built with "
             "Python, scikit-learn, XGBoost, Pandas and Streamlit.")
    st.subheader("What do the inputs mean?")
    glossary = {
        "Chest pain type": "Typical angina is classic exertional chest pain. Asymptomatic means no chest pain, "
                           "which is surprisingly common among patients who do have disease in this dataset.",
        "Resting blood pressure": "Blood pressure at rest, in mm Hg.",
        "Cholesterol": "Serum cholesterol in mg/dl.",
        "Fasting blood sugar": "Whether fasting blood sugar is above 120 mg/dl.",
        "Resting ECG": "Electrocardiogram result at rest.",
        "Max heart rate": "Highest heart rate reached during the stress test.",
        "Exercise-induced angina": "Chest pain triggered by exercise.",
        "ST depression (oldpeak)": "How much the ST segment dropped during exercise compared with rest.",
        "Slope": "Direction of the ST segment at peak exercise.",
        "Major vessels (ca)": "Number of major vessels (0 to 3) showing up on fluoroscopy.",
        "Thalassemia": "Blood-flow result from a thallium stress test.",
    }
    for k, v in glossary.items():
        with st.expander(k):
            st.write(v)
    st.warning("This tool is for educational and demonstration purposes only. Do not rely on it for "
               "medical decisions or diagnosis. For real clinical concerns, consult a licensed "
               "healthcare professional.")