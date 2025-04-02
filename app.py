from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle
import os

app = Flask(__name__)

def load_or_train_model():
    model_path = 'model.pkl'
    csv_path = 'car_data.csv'
    print(f"Checking for model at {model_path}")

    if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
        print("Loading existing model...")
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            df = pd.read_csv(csv_path)
            if df.empty:
                print("Error: car_data.csv is empty!")
                return None, None
            categorical_cols = ['brand', 'fuel_type', 'seller_type', 'transmission', 'owner_type']
            df = pd.get_dummies(df, columns=categorical_cols)
            features = df.drop('price', axis=1)
            return model, features.columns
        except (EOFError, pickle.UnpicklingError):
            print("Error: model.pkl is corrupted. Will train a new model.")

    print("Training new model...")
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found!")
        return None, None
    
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            print("Error: car_data.csv is empty!")
            return None, None
    except pd.errors.EmptyDataError:
        print("Error: car_data.csv is empty or invalid!")
        return None, None
    
    categorical_cols = ['brand', 'fuel_type', 'seller_type', 'transmission', 'owner_type']
    df = pd.get_dummies(df, columns=categorical_cols)
    X = df.drop('price', axis=1)
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print("Model trained and saved")
    return model, X.columns

model, feature_columns = load_or_train_model()
if model is None:
    print("Failed to load or train model. Exiting.")
    exit(1)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = {
        'brand': request.form['brand'],
        'year': float(request.form['year']),
        'km_driven': float(request.form['km_driven']),
        'fuel_type': request.form['fuel_type'],
        'seller_type': request.form['seller_type'],
        'transmission': request.form['transmission'],
        'owner_type': request.form['owner_type'],
        'mileage': float(request.form['mileage']),
        'engine_cc': float(request.form['engine_cc']),
        'max_power': float(request.form['max_power']),
        'seats': float(request.form['seats'])
    }
    input_df = pd.DataFrame([input_data])
    
    # Apply the same one-hot encoding as during training
    categorical_cols = ['brand', 'fuel_type', 'seller_type', 'transmission', 'owner_type']
    input_df = pd.get_dummies(input_df, columns=categorical_cols)
    
    # Drop any unseen columns not in training data
    input_df = input_df.drop(columns=[col for col in input_df.columns if col not in feature_columns], errors='ignore')
    
    # Align input_df with the training feature columns
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_columns]  # Ensure exact match with training columns
    
    prediction = model.predict(input_df)[0]  # Price is already in INR
    return render_template('index.html', data=input_data, prediction=round(prediction, 1), show_result=True)

if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(debug=True)