import os
import requests
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-here')

# Backend API configuration
BACKEND_URL = os.getenv('BACKEND_URL', 'http://localhost:8000')

def get_backend_data(endpoint):
    """Helper function to get data from backend API"""
    try:
        response = requests.get(f"{BACKEND_URL}{endpoint}", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        flash(f"Error connecting to backend: {str(e)}", 'error')
        return None

def post_backend_data(endpoint, data):
    """Helper function to post data to backend API"""
    try:
        response = requests.post(f"{BACKEND_URL}{endpoint}", 
                               json=data, 
                               headers={'Content-Type': 'application/json'},
                               timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        flash(f"Error connecting to backend: {str(e)}", 'error')
        return None

@app.route('/')
def index():
    """Welcome screen with model selection"""
    # Get available models from backend
    health_data = get_backend_data('/health')
    if not health_data:
        return render_template('error.html', 
                             error="Unable to connect to backend service")
    
    models = health_data.get('models', [])
    return render_template('index.html', models=models)

@app.route('/model/<model_name>')
def model_config(model_name):
    """Model configuration page"""
    # Get schema information
    schema_data = get_backend_data('/schema')
    if not schema_data:
        return render_template('error.html', 
                             error="Unable to get model schema")
    
    # Model-specific parameters from config
    model_params = {
        'xgb': {
            'n_estimators': {'type': 'number', 'default': 500, 'min': 1, 'max': 2000},
            'max_depth': {'type': 'number', 'default': 8, 'min': 1, 'max': 20},
            'learning_rate': {'type': 'number', 'default': 0.1, 'min': 0.01, 'max': 1.0, 'step': 0.01}
        },
        'lgbm': {
            'n_estimators': {'type': 'number', 'default': 800, 'min': 1, 'max': 2000},
            'learning_rate': {'type': 'number', 'default': 0.05, 'min': 0.01, 'max': 1.0, 'step': 0.01}
        },
        'catboost': {
            'iterations': {'type': 'number', 'default': 800, 'min': 1, 'max': 2000},
            'depth': {'type': 'number', 'default': 8, 'min': 1, 'max': 20},
            'learning_rate': {'type': 'number', 'default': 0.1, 'min': 0.01, 'max': 1.0, 'step': 0.01}
        },
        'lccde': {
            'description': 'LCCDE is an ensemble method that combines XGBoost, LightGBM, and CatBoost models'
        }
    }
    
    features = schema_data.get('features', [])
    classes = schema_data.get('classes', [])
    available_models = schema_data.get('models', [])
    
    if model_name not in available_models:
        flash(f"Model '{model_name}' is not available", 'error')
        return redirect(url_for('index'))
    
    params = model_params.get(model_name, {})
    
    return render_template('model_config.html', 
                         model_name=model_name,
                         features=features,
                         classes=classes,
                         parameters=params)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    model_name = request.form.get('model_name')
    inputs = {}
    
    # Extract feature values from form
    for key, value in request.form.items():
        if key != 'model_name' and key.startswith('feature_'):
            feature_name = key.replace('feature_', '')
            try:
                inputs[feature_name] = float(value)
            except ValueError:
                flash(f"Invalid value for {feature_name}: {value}", 'error')
                return redirect(url_for('model_config', model_name=model_name))
    
    # Make prediction request to backend
    prediction_data = {
        'model': model_name,
        'inputs': inputs
    }
    
    result = post_backend_data('/predict', prediction_data)
    if result is None:
        return redirect(url_for('model_config', model_name=model_name))
    
    if 'error' in result:
        flash(f"Prediction error: {result['error']}", 'error')
        return redirect(url_for('model_config', model_name=model_name))
    
    # Get classes from backend schema
    schema_data = get_backend_data('/schema')
    logs = get_backend_data('/logs')
    comparisons = []
    for log in logs:
        if log['model_name'] == model_name:
            comparisons.append(log)
    classes = schema_data.get('classes', []) if schema_data else []
    return render_template('prediction_result.html', 
                         model_name=model_name,
                         prediction=result['prediction'],
                         meta=result.get('meta', {}),
                         classes=classes,
                         comparisons=comparisons)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
