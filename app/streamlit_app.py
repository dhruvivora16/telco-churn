import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

from src.config import CATEGORICAL, MODEL_PATH, NUMERIC
from src.features import add_features, clean_name

st.set_page_config(page_title="Churn Risk Scorer", page_icon="📉", layout="wide")


@st.cache_resource
def load():
    pipe = joblib.load(MODEL_PATH)
    return pipe, shap.TreeExplainer(pipe.named_steps["clf"])


pipe, explainer = load()

st.title("📉 Customer Churn Risk Scorer")
st.caption("XGBoost model trained on the IBM Telco Customer Churn dataset. "
           "Enter a customer profile to estimate churn probability.")

YN = ["Yes", "No"]
with st.sidebar:
    st.header("Customer profile")
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly = st.slider("Monthly charges ($)", 18.0, 120.0, 70.0, step=0.5)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
    payment = st.selectbox("Payment method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    paperless = st.selectbox("Paperless billing", YN)
    senior = st.selectbox("Senior citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
    partner = st.selectbox("Partner", YN, index=1)
    dependents = st.selectbox("Dependents", YN, index=1)
    gender = st.selectbox("Gender", ["Female", "Male"])
    phone = st.selectbox("Phone service", YN)
    multi = st.selectbox("Multiple lines", ["No", "Yes"])
    sec = st.selectbox("Online security", ["No", "Yes"])
    backup = st.selectbox("Online backup", ["No", "Yes"])
    protect = st.selectbox("Device protection", ["No", "Yes"])
    tech = st.selectbox("Tech support", ["No", "Yes"])
    tv = st.selectbox("Streaming TV", ["No", "Yes"])
    movies = st.selectbox("Streaming movies", ["No", "Yes"])

row = pd.DataFrame([{
    "gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
    "tenure": tenure, "PhoneService": phone, "MultipleLines": multi, "InternetService": internet,
    "OnlineSecurity": sec, "OnlineBackup": backup, "DeviceProtection": protect,
    "TechSupport": tech, "StreamingTV": tv, "StreamingMovies": movies, "Contract": contract,
    "PaperlessBilling": paperless, "PaymentMethod": payment, "MonthlyCharges": monthly,
    "TotalCharges": round(tenure * monthly, 2),
}])
row = add_features(row)[CATEGORICAL + NUMERIC]

proba = float(pipe.predict_proba(row)[0, 1])
band, icon = (("HIGH", "🔴") if proba >= 0.6 else ("MEDIUM", "🟠") if proba >= 0.3 else ("LOW", "🟢"))

c1, c2 = st.columns([1, 2])
with c1:
    st.metric("Churn probability", f"{proba:.1%}")
    st.subheader(f"{icon} {band} risk")
    st.progress(min(proba, 1.0))
    if band == "HIGH":
        st.warning("Suggested action: retention offer, e.g. a contract-upgrade discount.")
    elif band == "MEDIUM":
        st.info("Suggested action: proactive check-in or tech-support outreach.")
    else:
        st.success("No action needed.")

with c2:
    st.subheader("Why this score?")
    pre = pipe.named_steps["prep"]
    names = list(pre.get_feature_names_out())
    X = pd.DataFrame(pre.transform(row), columns=names)
    sv = explainer(X)
    sv.feature_names = [clean_name(n) for n in names]
    plt.figure()
    shap.plots.waterfall(sv[0], max_display=8, show=False)
    st.pyplot(plt.gcf(), clear_figure=True)
    st.caption("Red bars push churn risk up, blue bars push it down (log-odds scale).")
