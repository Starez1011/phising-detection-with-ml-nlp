import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import pickle
import FeatureExtraction
from sklearn.preprocessing import LabelEncoder

def preprocess_data(data):
    # Drop non-numeric columns
    numeric_data = data.select_dtypes(include=['int64', 'float64'])
    
    # Handle any missing values
    numeric_data = numeric_data.fillna(0)
    
    return numeric_data

def load_data():
    # Load your training data
    legitimate_urls = pd.read_csv("../extracted_csv_files/legitimate_websites_1.11.csv")
    phishing_urls = pd.read_csv("../extracted_csv_files/phishing_websites_1.11.csv")
    
    # Combine the datasets
    data = pd.concat([legitimate_urls, phishing_urls])
    
    # Separate features and target
    X = data.drop('label', axis=1)
    y = data['label']
    
    # Preprocess the features
    X = preprocess_data(X)
    
    return X, y

def train_and_save_models():
    print("Loading data...")
    X, y = load_data()
    
    print("Training XGBoost model...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    xgb_model.fit(X, y)
    
    print("Training Random Forest model...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    rf_model.fit(X, y)
    
    print("Saving models...")
    # Save XGBoost model
    xgb_model.save_model('XGBoostModel_12000.sav')
    
    # Save Random Forest model
    with open('RFmodel_12000.sav', 'wb') as f:
        pickle.dump(rf_model, f)
    
    print("Models saved successfully!")

if __name__ == "__main__":
    train_and_save_models() 