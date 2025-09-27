import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib

# ---- MODELİ YÜKLE ----
model = joblib.load("best_churn_model.pkl")

# ---- APP BAŞLIK ----
st.title("💳 Bank Customer Churn Prediction App")

st.write("Bu uygulama, bir müşterinin bankadan ayrılıp ayrılmayacağını tahmin eder.")

# ---- Kullanıcı Input ----
credit_score = st.number_input("Kredi Skoru", min_value=300, max_value=900, value=600)
age = st.number_input("Yaş", min_value=18, max_value=100, value=35)
tenure = st.number_input("Bankada Çalışma Süresi (Yıl)", min_value=0, max_value=20, value=5)
balance = st.number_input("Bakiye", min_value=0.0, value=50000.0)
num_of_products = st.number_input("Ürün Sayısı", min_value=1, max_value=4, value=1)
has_cr_card = st.selectbox("Kredi Kartı Var mı?", [0, 1])
is_active_member = st.selectbox("Aktif Üye mi?", [0, 1])
estimated_salary = st.number_input("Tahmini Maaş", min_value=0.0, value=50000.0)

geography = st.selectbox("Ülke", ["France", "Germany", "Spain"])
gender = st.selectbox("Cinsiyet", ["Female", "Male"])

# ---- ENCODING ----
if geography == "France":
    geo_germany, geo_spain = 0, 0
elif geography == "Germany":
    geo_germany, geo_spain = 1, 0
else:  # Spain
    geo_germany, geo_spain = 0, 1

gender_encoded = 1 if gender == "Male" else 0

# ---- FEATURE VECTOR ----
columns = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
           'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
           'Gender_encoded', 'Geography_Germany', 'Geography_Spain']

input_data = pd.DataFrame([[credit_score, age, tenure, balance, num_of_products,
                            has_cr_card, is_active_member, estimated_salary,
                            gender_encoded, geo_germany, geo_spain]],
                          columns=columns)

# ---- TAHMİN ----
if st.button("Tahmin Et"):
    proba = model.predict_proba(input_data)[0][1]
    prediction = model.predict(input_data)[0]

    st.write("📊 Churn Olasılığı:", round(proba, 2))

    if prediction == 1:
        st.error("⚠️ Müşteri bankadan ayrılabilir!")
    else:
        st.success("✅ Müşteri bankada kalacak gibi görünüyor.")

