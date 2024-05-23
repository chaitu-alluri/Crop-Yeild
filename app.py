import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


# Load the dataset
@st.cache_resource
def load_data():
    df = pd.read_csv("crop_yield.csv")  # Replace "crop_yield.csv" with your actual dataset file path
    return df


# Data Preprocessing
def preprocess_data(df):
    # Encode categorical variables
    df = pd.get_dummies(df, columns=["Crop", "Season", "State"])

    # Check if 'Crop_Year' column is present in the DataFrame
    if 'Crop_Year' in df.columns:
        # Drop 'Crop_Year' column if it's present
        df = df.drop(columns=["Crop_Year"])

    # Split data into features and target variable
    X = df.drop(columns=["Yield"])
    y = df["Yield"]

    return X, y


# Model Training
def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


# Model Evaluation
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2


# Prediction
def predict_yield(model, input_features):
    # Predict yield based on preprocessed input features
    prediction = model.predict(input_features)
    return prediction


def main():
    st.title("Crop Yield Prediction App")
    st.divider()

    # Load data
    df = load_data()

    # Preprocess data
    X, y = preprocess_data(df)
    columns = X.columns

    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)

    st.header("Crop Yield Prediction")
    st.subheader("Select Input Parameters")

    # Dropdown menu for selecting state
    selected_state = st.selectbox("Select State", df["State"].unique())

    # Dropdown menu for selecting crop
    selected_crop = st.selectbox("Select Crop", df["Crop"].unique())

    # Dropdown menu for selecting season
    selected_season = st.selectbox("Select Season", df["Season"].unique())

    # Slider for selecting Production
    min_production = df['Production'].min()
    max_production = df['Production'].max()
    production = st.slider("Select Production", min_value=min_production, max_value=max_production, value=min_production)

    # Slider for selecting Annual Rainfall
    min_rainfall = df['Annual_Rainfall'].min()
    max_rainfall = df['Annual_Rainfall'].max()
    annual_rainfall = st.slider("Select Annual Rainfall", min_value=min_rainfall, max_value=max_rainfall, value=min_rainfall)

    # Slider for selecting Area
    min_area = df['Area'].min()
    max_area = df['Area'].max()
    area = st.slider("Select Area", min_value=min_area, max_value=max_area, value=min_area)

    # Slider for selecting Fertilizer
    min_fertilizer = df['Fertilizer'].min()
    max_fertilizer = df['Fertilizer'].max()
    fertilizer = st.slider("Select Fertilizer", min_value=min_fertilizer, max_value=max_fertilizer, value=min_fertilizer)

    # Slider for selecting Pesticide
    min_pesticide = df['Pesticide'].min()
    max_pesticide = df['Pesticide'].max()
    pesticide = st.slider("Select Pesticide", min_value=min_pesticide, max_value=max_pesticide, value=min_pesticide)

    # Predict yield based on user input
    if st.button("Predict Yield"):
        input_features = pd.DataFrame({
            'State': [selected_state],
            'Crop': [selected_crop],
            'Season': [selected_season],
            'Production': [production],
            'Annual_Rainfall': [annual_rainfall],
            'Area': [area],
            'Fertilizer': [fertilizer],
            'Pesticide': [pesticide]
        })

        # One-hot encode categorical variables
        input_features = pd.get_dummies(input_features, columns=["Crop", "Season", "State"])

        # Ensure input features contain all columns present during training
        for col in columns:
            if col not in input_features.columns:
                input_features[col] = 0

        # Reorder columns to match training data
        input_features = input_features[columns]

        # Predict yield based on preprocessed input features
        prediction = predict_yield(model, input_features)
        st.subheader(f"Predicted Yield: {prediction[0]}")

    # Display model evaluation metrics
    #st.subheader("Model Evaluation Metrics:")
    #mse, r2 = evaluate_model(model, X_test, y_test)
    #st.write(f"- Mean Squared Error: {mse}")
    #st.write(f"- R-squared: {r2}")


if __name__ == "__main__":
    main()
