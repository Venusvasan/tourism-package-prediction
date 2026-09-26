import os
import streamlit as st
import pandas as pd
import joblib

# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(os.path.dirname(__file__), "best_model.joblib")
model = joblib.load(model_path)

st.title("Tourism Package Purchase Predictor")
st.write("""
Predict whether a prospective customer will purchase the newly introduced "
    "**Wellness Tourism Package** prior to direct outreach.
""")
# Organize input form into 3 clear columns
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Customer Demographics")
    age = st.slider("Age", min_value=18, max_value=75, value=35)
    gender = st.selectbox("Gender", ["Male", "Female"])
    occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Freelancer"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
    monthly_income = st.number_input("Monthly Income ($)", min_value=1000, max_value=200000, value=25000, step=1000)
    city_tier = st.selectbox("City Tier", [1, 2, 3], index=0)

with col2:
    st.subheader("Travel Profile")
    num_person = st.slider("Number of Person Visiting", min_value=1, max_value=10, value=2)
    preferred_star = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0], index=0)
    num_trips = st.slider("Number of Annual Trips", min_value=1, max_value=25, value=3)
    passport = st.selectbox("Valid Passport?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    own_car = st.selectbox("Owns a Car?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    num_children = st.slider("Children Visiting (under age 5)", min_value=0, max_value=5, value=0)
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])

with col3:
    st.subheader("Pitch & Sales Interactions")
    pitch_satisfaction = st.slider("Pitch Satisfaction Score (1-5)", min_value=1, max_value=5, value=3)
    product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    num_followups = st.slider("Number of Follow-ups", min_value=1, max_value=10, value=3)
    duration_pitch = st.slider("Duration of Pitch (Minutes)", min_value=5, max_value=120, value=20)
    type_contact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])

input_data = pd.DataFrame([{
    "Age": age,
    "TypeofContact": type_contact,
    "CityTier": city_tier,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": num_person,
    "PreferredPropertyStar": preferred_star,
    "MaritalStatus": marital_status,
    "NumberOfTrips": num_trips,
    "Passport": passport,
    "OwnCar": own_car,
    "NumberOfChildrenVisiting": num_children,
    "Designation": designation,
    "MonthlyIncome": monthly_income,
    "PitchSatisfactionScore": pitch_satisfaction,
    "ProductPitched": product_pitched,
    "NumberOfFollowups": num_followups,
    "DurationOfPitch": duration_pitch
}])

st.divider()

if st.button("Predict Purchase"):
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    purchase_prob = probabilities[1]
    st.subheader("Prediction Result:")
    if prediction == 1:
        st.success(
            f"### Highly likely to Purchase!\n"
            f"**Likelihood to Purchase: {purchase_prob:.1%}**\n\n"
            f"Recommendation: Prioritize this lead for direct marketing contact."
        )
        st.balloons()
    else:
        st.info(
            f"### Less Likely to Purchase\n"
            f"**Likelihood to Purchase: {purchase_prob:.1%}**\n\n"
            f"Recommendation: Deprioritize  outreach to preserve sales resources."
        )
