# Bank Customer Churn Prediction

This project aims to predict whether a bank customer is likely to churn (leave the bank) based on demographic and financial information such as credit score, age, account balance, and activity status. It combines a machine learning model with a simple Streamlit web application for interactive use.

The machine learning model (`best_churn_model.pkl`) was trained and evaluated in a Jupyter Notebook (`churn.ipynb`). The Streamlit app (`app.py`) provides an interface where users can input customer details and instantly receive a churn probability and prediction.

Live App (Local Run): [http://localhost:8501/](http://localhost:8501/)

---

## Features
- Interactive Streamlit web application
- Pre-trained machine learning model integrated into the app
- Outputs both probability score and classification (Churn / Not Churn)
- Easy input of customer data through a user-friendly interface

---

## Installation & Usage
Clone the repository and install the dependencies:

```bash
git clone https://github.com/yourusername/churn-prediction.git
cd churn-prediction
pip install -r requirements.txt

---

