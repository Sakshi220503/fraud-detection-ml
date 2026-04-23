# 1. Imports
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 2. Load model + data
model = joblib.load("C:/Users/Sakshi/fraud-detection-project/model/fraud_model.pkl")
df = pd.read_csv("C:/Users/Sakshi/fraud-detection-project/data/creditcard.csv")

# 3. Title
st.title("💳 Credit Card Fraud Detection")

# =========================
# 🔹 MANUAL INPUT
# =========================
st.subheader("Enter Transaction Amount")

amount = st.number_input("Amount", min_value=0.0)

if st.button("Check Manual Transaction"):
    features = np.zeros((1, 30))
    features[0, -1] = amount

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Fraud Detected (Confidence: {probability:.2f})")
    else:
        st.success(f"✅ Safe Transaction (Confidence: {1 - probability:.2f})")

# =========================
# 🔹 RANDOM TRANSACTION
# =========================
st.subheader("Test with Real Data")

if st.button("Check Random Transaction"):
    sample = df.sample(1)

    features = sample.drop("Class", axis=1).values
    actual = sample["Class"].values[0]

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    st.write("### Transaction Details")
    st.write(sample)

    st.write(f"🧾 Actual: {'Fraud' if actual == 1 else 'Normal'}")

    if prediction == 1:
        st.error(f"⚠️ Predicted Fraud (Confidence: {probability:.2f})")
    else:
        st.success(f"✅ Predicted Safe (Confidence: {1 - probability:.2f})")

# =========================
# 🔥 ADD HERE (Fraud Test)
# =========================
st.subheader("Force Fraud Testing")

if st.button("⚠️ Test Fraud Case"):

    sample = df[df["Class"] == 1].sample(1)

    features = sample.drop("Class", axis=1).values
    actual = sample["Class"].values[0]

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    st.write("### Fraud Transaction Details")
    st.write(sample)

    if prediction == 1:
        st.error(f"✅ Correctly Detected Fraud (Confidence: {probability:.2f})")
    else:
        st.warning(f"❌ Missed Fraud (Confidence: {probability:.2f})")