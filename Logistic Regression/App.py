import streamlit as st
import pandas as pd
import pickle
import os
import numpy as np
from sklearn.preprocessing import StandardScaler  # Ensure scikit-learn is available

def load_model():
    file_path = "logistic_regression_titanic.pkl"  # Ensure correct model file name
    if not os.path.exists(file_path):
        st.error(f"Model file '{file_path}' not found. Please upload the model file.")
        return None
    try:
        with open(file_path, "rb") as file:
            model = pickle.load(file)
        return model
    except ModuleNotFoundError as e:
        st.error("ModuleNotFoundError: The model may have been saved with missing dependencies.")
        st.error(str(e))
        return None

model = load_model()

# Function to make predictions
def predict_survival(features):
    if model is None:
        return None, None
    
    # Ensure all required features are present
    expected_features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    for feature in expected_features:
        if feature not in features:
            st.error(f"Missing feature: {feature}. Please provide all required inputs.")
            return None, None
    
    features_df = pd.DataFrame([features])
    prediction = model.predict(features_df)
    probability = model.predict_proba(features_df)[:, 1]
    return prediction[0], probability[0]

# Streamlit UI
st.set_page_config(page_title="Titanic Survival Predictor", layout="centered")
st.title("🚢 Titanic Survival Prediction App")
st.write("Enter passenger details to predict survival probability.")

# Sidebar for user input
st.sidebar.header("User Input")

pclass = st.sidebar.selectbox("Passenger Class", [1, 2, 3])
sex = st.sidebar.selectbox("Sex", ["male", "female"])
age = st.sidebar.slider("Age", 1, 100, 30)
sibsp = st.sidebar.slider("Siblings/Spouses Aboard", 0, 8, 0)
parch = st.sidebar.slider("Parents/Children Aboard", 0, 6, 0)
fare = st.sidebar.slider("Fare Paid", 0.0, 500.0, 30.0)
embarked = st.sidebar.selectbox("Embarked Port", ["Cherbourg", "Queenstown", "Southampton"])

# Convert categorical values
sex = 1 if sex == "male" else 0
embarked_dict = {"Cherbourg": 0, "Queenstown": 1, "Southampton": 2}
embarked = embarked_dict[embarked]

features = {
    "Pclass": pclass,
    "Sex": sex,
    "Age": age,
    "SibSp": sibsp,
    "Parch": parch,
    "Fare": fare,
    "Embarked": embarked,
}

if st.sidebar.button("Predict Survival"):
    prediction, probability = predict_survival(features)
    
    if prediction is None:
        st.error("Prediction cannot be made because the model file is missing or incompatible.")
    else:
        st.subheader("Prediction Result")
        st.write(f"Survival Prediction: {'Survived' if prediction == 1 else 'Did Not Survive'}")
        st.write(f"Survival Probability: {probability:.2%}")

st.markdown("---")
st.write("Built with Streamlit | Logistic Regression Model")
