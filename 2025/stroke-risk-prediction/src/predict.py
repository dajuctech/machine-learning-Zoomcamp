# src/predict.py - Prediction functions
# Using Gradient Boosting model

import pickle
import pandas as pd

MODEL_PATH = "../models/model.bin"

# Load model once (global)
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

print(f"✅ Model loaded from {MODEL_PATH}")

def predict_single(patient):
    """
    Predict stroke risk for a single patient
    
    Args:
        patient (dict): Patient data with keys:
            - gender: str ('Male', 'Female', 'Other')
            - age: float
            - hypertension: int (0 or 1)
            - heart_disease: int (0 or 1)
            - ever_married: str ('Yes' or 'No')
            - work_type: str ('Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked')
            - Residence_type: str ('Urban' or 'Rural')
            - avg_glucose_level: float
            - bmi: float
            - smoking_status: str ('formerly smoked', 'never smoked', 'smokes', 'Unknown')
    
    Returns:
        dict: Prediction results with probability, prediction, and risk level
    """
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
    
    return {
        'stroke_probability': float(proba),
        'stroke_prediction': prediction,
        'risk_level': risk_level
    }

# Test function
if __name__ == "__main__":
    print("\n" + "="*60)
    print("TESTING PREDICTION FUNCTION")
    print("="*60)
    
    # Test patient 1: High risk
    test_patient_high = {
        'gender': 'Male',
        'age': 67.0,
        'hypertension': 0,
        'heart_disease': 1,
        'ever_married': 'Yes',
        'work_type': 'Private',
        'Residence_type': 'Urban',
        'avg_glucose_level': 228.69,
        'bmi': 36.6,
        'smoking_status': 'formerly smoked'
    }
    
    result1 = predict_single(test_patient_high)
    print("\nTest 1 - High Risk Patient:")
    print(f"  Age: {test_patient_high['age']}, Heart Disease: Yes, BMI: {test_patient_high['bmi']}")
    print(f"  Stroke Probability: {result1['stroke_probability']:.4f}")
    print(f"  Prediction: {result1['stroke_prediction']}")
    print(f"  Risk Level: {result1['risk_level']}")
    
    # Test patient 2: Low risk
    test_patient_low = {
        'gender': 'Female',
        'age': 25.0,
        'hypertension': 0,
        'heart_disease': 0,
        'ever_married': 'No',
        'work_type': 'Private',
        'Residence_type': 'Urban',
        'avg_glucose_level': 85.0,
        'bmi': 22.0,
        'smoking_status': 'never smoked'
    }
    
    result2 = predict_single(test_patient_low)
    print("\nTest 2 - Low Risk Patient:")
    print(f"  Age: {test_patient_low['age']}, Heart Disease: No, BMI: {test_patient_low['bmi']}")
    print(f"  Stroke Probability: {result2['stroke_probability']:.4f}")
    print(f"  Prediction: {result2['stroke_prediction']}")
    print(f"  Risk Level: {result2['risk_level']}")
    
    print("\n" + "="*60)
    print("✅ PREDICTION TESTS COMPLETE")
    print("="*60)