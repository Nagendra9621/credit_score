# credit_score_app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# Set page title
st.set_page_config(page_title="Credit Risk Scoring", layout="centered")

st.title("💳 Credit Scoring Risk Predictor")
st.write("Fill in the financial details below to estimate default risk.")

# Generate synthetic dataset
np.random.seed(0)
n = 1000
data = pd.DataFrame({
    'age': np.random.randint(21, 65, size=n),
    'income': np.random.randint(20000, 100000, size=n),
    'loan_amount': np.random.randint(1000, 30000, size=n),
    'credit_score': np.random.randint(300, 850, size=n),
    'default': np.random.choice([0, 1], size=n, p=[0.85, 0.15])
})

X = data[['age', 'income', 'loan_amount', 'credit_score']]
y = data['default']

# Train logistic regression model
model = LogisticRegression()
model.fit(X, y)

# User input sliders
st.sidebar.header("📥 Input Details")

age = st.sidebar.slider("Age", 21, 65, 30)
income = st.sidebar.slider("Annual Income ($)", 20000, 100000, 50000, step=1000)
loan_amount = st.sidebar.slider("Loan Amount ($)", 1000, 30000, 10000, step=500)
credit_score = st.sidebar.slider("Credit Score", 300, 850, 600)

# Create DataFrame from user input
user_input = pd.DataFrame({
    'age': [age],
    'income': [income],
    'loan_amount': [loan_amount],
    'credit_score': [credit_score]
})

# Make prediction
prob_default = model.predict_proba(user_input)[0][1]
prediction = model.predict(user_input)[0]

# Show result
st.subheader("📊 Result")
st.write(f"**Probability of Default:** `{prob_default:.2%}`")
st.write("**Prediction:**", "🔴 High Risk" if prediction == 1 else "🟢 Low Risk")

# Show feature influence (manual explanation)
st.subheader("🔍 Factors Considered")
coef = model.coef_[0]
features = X.columns.tolist()

for i in range(len(features)):
    st.write(f"- **{features[i].capitalize()}**: Weight = `{coef[i]:.4f}`")

st.info("Positive weights contribute to higher risk. Negative weights reduce the risk.")
