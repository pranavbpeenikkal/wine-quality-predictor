import streamlit as st
import pandas as pd
import joblib
import os

st.title("Wine Quality Predictor")
st.write("This app predicts whether a red wine is good (Quality >= 7) based on its chemical properties.")

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

@st.cache_data
def load_data():
    df = pd.read_csv("dataset.csv", sep=";")
    X = df.drop(['quality'], axis=1, errors='ignore')
    if 'target' in X.columns:
        X = X.drop('target', axis=1)
    return X

try:
    model = load_model()
    X = load_data()
    
    st.sidebar.header("Input Features")
    input_data = {}
    
    for col in X.columns:
        min_val = float(X[col].min())
        max_val = float(X[col].max())
        mean_val = float(X[col].mean())
        
        # Determine step based on the range
        step = float((max_val - min_val) / 100.0)
        if step == 0: step = 0.1
            
        input_data[col] = st.sidebar.slider(col, min_val, max_val, mean_val, step=step)
        
    if st.button("Predict"):
        input_df = pd.DataFrame([input_data])
        
        prediction = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]
        
        if prediction == 1:
            st.success(f"🍷 **Good Wine!** (Probability: {prob:.2f})")
        else:
            st.error(f"🍇 **Not Good Wine.** (Probability: {1-prob:.2f})")
            
except Exception as e:
    st.error(f"Error loading model or data: {e}")
    st.write("Make sure to run main.py first to generate 'model.pkl' and download the dataset.")
