# app/main.py - Flask web service for Stroke Risk Prediction
# Using Gradient Boosting model

from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load model
MODEL_PATH = "../models/model.bin"
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

print(f"✅ Model loaded from {MODEL_PATH}")

@app.route('/')
def home():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Stroke Risk Prediction API',
        'model': 'Gradient Boosting (Tuned)',
        'version': '1.0',
        'endpoints': {
            'predict': 'POST /predict'
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint
    
    Expected JSON:
    {
        "gender": "Male",
        "age": 67,
        "hypertension": 0,
        "heart_disease": 1,
        "ever_married": "Yes",
        "work_type": "Private",
        "Residence_type": "Urban",
        "avg_glucose_level": 228.69,
        "bmi": 36.6,
        "smoking_status": "formerly smoked"
    }
    """
    try:
        # Get patient data
        patient = request.get_json()
        
        # Validate required fields
        required_fields = [
            'gender', 'age', 'hypertension', 'heart_disease',
            'ever_married', 'work_type', 'Residence_type',
            'avg_glucose_level', 'bmi', 'smoking_status'
        ]
        
        missing_fields = [field for field in required_fields if field not in patient]
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {missing_fields}'
            }), 400
        
        # Convert to DataFrame
        df = pd.DataFrame([patient])
        
        # Predict
        proba = model.predict_proba(df)[0, 1]
        prediction = int(proba >= 0.5)
        
        # Risk categorization
        if proba >= 0.7:
            risk_level = 'VERY HIGH'
        elif proba >= 0.5:
            risk_level = 'HIGH'
        elif proba >= 0.3:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'
        
        # Prepare response
        response = {
            'stroke_probability': round(float(proba), 4),
            'stroke_prediction': prediction,
            'risk_level': risk_level,
            'model': 'Gradient Boosting',
            'patient_data': patient
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)