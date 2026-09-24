import streamlit as st
import pandas as pd
import joblib


# Load trained model
model = joblib.load("model/house_price_model.pkl")


# Page settings
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered"
)


# Title
st.title("🏠 House Price Prediction")

st.write(
    "Enter the property details below to estimate the house price."
)


# -----------------------------
# USER INPUTS
# -----------------------------

area = st.number_input(
    "Area (Square Feet)",
    min_value=500,
    max_value=10000,
    value=1500,
    step=100
)

bedrooms = st.number_input(
    "Number of Bedrooms",
    min_value=1,
    max_value=10,
    value=3,
    step=1
)

bathrooms = st.number_input(
    "Number of Bathrooms",
    min_value=1,
    max_value=10,
    value=2,
    step=1
)

floors = st.number_input(
    "Number of Floors",
    min_value=1,
    max_value=5,
    value=1,
    step=1
)

parking = st.number_input(
    "Parking Spaces",
    min_value=0,
    max_value=5,
    value=1,
    step=1
)

age = st.number_input(
    "Property Age (Years)",
    min_value=0,
    max_value=100,
    value=5,
    step=1
)


# -----------------------------
# PREDICTION
# -----------------------------

if st.button("Predict House Price"):

    input_data = pd.DataFrame({
        "area": [area],
        "bedrooms": [bedrooms],
        "bathrooms": [bathrooms],
        "floors": [floors],
        "parking": [parking],
        "age": [age]
    })

    prediction = model.predict(input_data)[0]

    st.success(
        f"Estimated House Price: ₹{prediction:,.2f}"
    )


    # -----------------------------
    # INPUT SUMMARY
    # -----------------------------

    st.subheader("Input Summary")

    st.write(f"**Area:** {area} sq.ft")
    st.write(f"**Bedrooms:** {bedrooms}")
    st.write(f"**Bathrooms:** {bathrooms}")
    st.write(f"**Floors:** {floors}")
    st.write(f"**Parking:** {parking}")
    st.write(f"**Property Age:** {age} years")


    # -----------------------------
    # INTERPRETATION
    # -----------------------------

    st.subheader("Prediction Interpretation")

    st.info(
        "The predicted price is an estimated value generated "
        "by the Linear Regression model. Actual house prices "
        "may differ depending on location, market conditions "
        "and other property characteristics."
    )