import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pickle

model = pickle.load(open('rfr.pkl','rb'))
# Sample dataset creation
def create_sample_data():
    data = {
        "Car Name": ["Toyota Corolla", "Honda Civic", "Ford Focus", "Hyundai Elantra", "BMW 3 Series"],
        "Year": [2015, 2017, 2016, 2018, 2020],
        "Mileage": [50000, 30000, 40000, 25000, 15000],
        "Engine Size": [1.8, 2.0, 1.5, 1.6, 2.5],
        "Horsepower": [132, 158, 160, 147, 255],
        "Fuel Efficiency (mpg)": [30, 32, 28, 33, 25],
        "Previous Owners": [1, 1, 2, 1, 0],
        "Price": [15000, 18000, 12000, 20000, 35000],
    }
    return pd.DataFrame(data)

# Load dataset
data = create_sample_data()


# Web App
st.title("Car Price Predictor")

st.sidebar.header("User Input Features")
car_name = st.sidebar.selectbox("Select Car Name", data["Car Name"].unique())
year = st.sidebar.slider("Year of Manufacture", min_value=2000, max_value=2024, value=2015)
mileage = st.sidebar.number_input("Mileage (in miles)", value=30000, step=500)
engine_size = st.sidebar.number_input("Engine Size (in liters)", value=1.8, step=0.1)
horsepower = st.sidebar.number_input("Horsepower", value=150, step=5)
fuel_efficiency = st.sidebar.number_input("Fuel Efficiency (mpg)", value=30, step=1)
previous_owners = st.sidebar.number_input("Number of Previous Owners", min_value=0, max_value=5, value=1)
# Split data into features and target
X = data.drop(columns=["Price", "Car Name"])
y = data["Price"]

# Train a simple model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)
# Predict button
if st.sidebar.button("Predict"):
    # Prepare input for prediction
    input_data = pd.DataFrame(
        {
            "Year": [year],
            "Mileage": [mileage],
            "Engine Size": [engine_size],
            "Horsepower": [horsepower],
            "Fuel Efficiency (mpg)": [fuel_efficiency],
            "Previous Owners": [previous_owners],
        }
    )
    prediction = model.predict(input_data)
    st.write(f"The estimated price for a {car_name} is: ${prediction[0]:,.2f}")

# Display dataset for reference
st.subheader("Sample Dataset")
st.dataframe(data)
