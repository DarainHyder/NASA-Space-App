import os
import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from sklearn.impute import SimpleImputer
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
import json

app = Flask(__name__)

# --- MODEL LOADING LOGIC ---
xgb_model = None
dt_pipeline = None

def create_fallback_dt_model():
    return DecisionTreeClassifier(max_depth=4, min_samples_split=30, min_samples_leaf=15, max_features="sqrt", random_state=42)

def create_fallback_dt_pipeline():
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("clf", create_fallback_dt_model())])

# Load models (Hugging Face expects these in the same directory or Models/)
try:
    model_path = os.path.join(os.path.dirname(__file__), 'Models', 'xgb_exoplanet_model.pkl')
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            xgb_model = pickle.load(f)
            print("XGBoost model loaded")
    
    # Create fallback for DT
    dt_pipeline = create_fallback_dt_pipeline()
    column_names = [
        'sy_snum', 'sy_pnum', 'disc_year', 'pl_orbper', 'pl_orbsmax', 'pl_rade', 'pl_radj', 'pl_bmasse', 'pl_bmassj', 
        'pl_orbeccen', 'pl_insol', 'pl_eqt', 'st_teff', 'st_rad', 'st_mass', 'st_met', 'st_logg', 'sy_dist', 
        'star_planet_size', 'temp_diff', 'log_insol', 'planets_per_system', 'stars_in_system', 
        'discoverymethod_MICROLENSING', 'discoverymethod_RADIAL VELOCITY', 'discoverymethod_TRANSIT'
    ]
    dummy_X = pd.DataFrame(np.random.rand(10, 26), columns=column_names)
    dummy_y = np.random.randint(0, 2, 10)
    dt_pipeline.fit(dummy_X, dummy_y)
    print("Decision Tree fallback ready")
except Exception as e:
    print(f"Loading error: {e}")

def preprocess_data(df):
    df_processed = df.copy()
    for col in df_processed.select_dtypes(include="object").columns:
        df_processed[col] = df_processed[col].str.strip().str.upper()
    
    required_columns = ['sy_snum', 'sy_pnum', 'disc_year', 'pl_orbper', 'pl_orbsmax', 'pl_rade', 'pl_radj', 'pl_bmasse', 'pl_bmassj', 'pl_orbeccen', 'pl_insol', 'pl_eqt', 'st_teff', 'st_rad', 'st_mass', 'st_met', 'st_logg', 'sy_dist']
    for col in required_columns:
        if col not in df_processed.columns: df_processed[col] = 0
    
    num_cols = df_processed.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        df_processed[col] = df_processed[col].fillna(df_processed[col].median())
    
    df_processed["star_planet_size"] = df_processed["st_rad"] * df_processed["pl_rade"]
    df_processed["temp_diff"] = df_processed["st_teff"] - df_processed["pl_eqt"]
    df_processed["log_insol"] = np.log10(df_processed["pl_insol"].replace(0, np.nan))
    df_processed["planets_per_system"] = df_processed["sy_pnum"]
    df_processed["stars_in_system"] = df_processed["sy_snum"]
    
    # Discovery methods
    df_processed['discoverymethod_MICROLENSING'] = 0
    df_processed['discoverymethod_RADIAL VELOCITY'] = 0
    df_processed['discoverymethod_TRANSIT'] = 1
    
    prediction_columns = [
        'sy_snum', 'sy_pnum', 'disc_year', 'pl_orbper', 'pl_orbsmax', 'pl_rade', 'pl_radj', 'pl_bmasse', 'pl_bmassj', 
        'pl_orbeccen', 'pl_insol', 'pl_eqt', 'st_teff', 'st_rad', 'st_mass', 'st_met', 'st_logg', 'sy_dist', 
        'star_planet_size', 'temp_diff', 'log_insol', 'planets_per_system', 'stars_in_system', 
        'discoverymethod_MICROLENSING', 'discoverymethod_RADIAL VELOCITY', 'discoverymethod_TRANSIT'
    ]
    return df_processed[prediction_columns].fillna(0)

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.json
    try:
        if input_data['type'] == 'manual':
            df = pd.DataFrame([input_data['data']])
        else:
            df = pd.read_json(input_data['data'])
        
        df_processed = preprocess_data(df)
        
        # Prediction
        res = {}
        if xgb_model:
            pred = xgb_model.predict(df_processed)
            prob = xgb_model.predict_proba(df_processed)[:, 1]
            res['xgb_prediction'] = 'Confirmed' if pred[0] == 1 else 'Candidate'
            res['xgb_confidence'] = f"{prob[0]:.2%}"
            
            # Batch results formatting
            if input_data['type'] == 'batch':
                df_display = df.copy()
                df_display['xgb_prediction'] = ['Confirmed' if p == 1 else 'Candidate' for p in pred]
                df_display['xgb_confidence'] = [f"{p:.2%}" for p in prob]
                res['df_html'] = df_display.head(100).to_html(classes='table table-striped table-hover', index=False)
                res['shape'] = df.shape
                res['model_name'] = "XGBoost"
        
        # Simple Viz Data
        res['visualizations'] = {
            'features': {
                'orbper': df['pl_orbper'].tolist() if 'pl_orbper' in df.columns else [],
                'rade': df['pl_rade'].tolist() if 'pl_rade' in df.columns else [],
                'teff': df['st_teff'].tolist() if 'st_teff' in df.columns else [],
                'dist': df['sy_dist'].tolist() if 'sy_dist' in df.columns else []
            }
        }
        if input_data['type'] == 'batch':
             res['visualizations']['predictions'] = { 'candidate': int((pred == 0).sum()), 'confirmed': int((pred == 1).sum()) }

        return jsonify(res)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)
