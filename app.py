import os
os.environ["STREAMLIT_WATCHDOG_OBSERVER_TYPE"] = "polling"
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

# Step 1: Import libraries
import streamlit as st
import pickle
import pandas as pd

# Step 2: Load the trained model
with open("model-reg-67130701919.pkl", "rb") as file:
    model = pickle.load(file)

# Step 3: Streamlit UI
st.title("💼 Advertising Sales Prediction App")

st.write("Enter your advertising budget for each platform to estimate the sales:")

# Input fields
youtube = st.number_input("YouTube budget", min_value=0.0, value=50.0)
tiktok = st.number_input("TikTok budget", min_value=0.0, value=50.0)
instagram = st.number_input("Instagram budget", min_value=0.0, value=50.0)

# Step 4: Predict when button is clicked
if st.button("Predict Sales"):
    new_data = pd.DataFrame({
        "youtube": [youtube],
        "tiktok": [tiktok],
        "instagram": [instagram]
    })
    predicted_sales = model.predict(new_data)
    st.success(f"📊 Estimated Sales: {predicted_sales[0]:.2f}")

st.caption("This app uses a Linear Regression model trained on advertising data.")
