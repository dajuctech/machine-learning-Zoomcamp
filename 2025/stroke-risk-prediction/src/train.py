# src/train.py - Training script for Stroke Risk Prediction
# Using best model: Gradient Boosting (Tuned)

import pandas as pd
import numpy as np
import pickle
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, precision_score, recall_score, f1_score

# Configuration
RAW_DATA_PATH = "../data/raw/healthcare-dataset-stroke-data.csv"
MODEL_OUTPUT_PATH = "../models/model.bin"
RANDOM_STATE = 42

# Best hyperparameters from tuning
BEST_PARAMS = {
    'n_estimators': 200,
    'learning_rate': 0.1,
    'max_depth': 5,
    'min_samples_split': 50,
    'min_samples_leaf': 20
}

def load_data():
    """Load raw data"""
    print("Loading data...")
    if not os.path.exists(RAW_DATA_PATH):
        print(f"❌ ERROR: Data file not found at {RAW_DATA_PATH}")
        print(f"Current directory: {os.getcwd()}")
        sys.exit(1)
    
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"✅ Loaded {len(df):,} records")
    return df

def prepare_data(df):
    """Prepare and split data"""
    print("\nPreparing data...")
    
    # Drop ID
    df = df.drop(columns=['id'])
    
    # Split: 80% train+val, 20% test
    train_val, test = train_test_split(
        df, test_size=0.2, stratify=df['stroke'], random_state=RANDOM_STATE
    )
    
    # Split train_val: 75% train, 25% val (60/20/20 overall)
    train, val = train_test_split(
        train_val, test_size=0.25, stratify=train_val['stroke'], random_state=RANDOM_STATE
    )
    
    print(f"Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")
    return train, val, test

def handle_missing_values(train, val, test):
    """Handle missing BMI values"""
    print("\nHandling missing values...")
    train_bmi_median = train['bmi'].median()
    
    train['bmi'] = train['bmi'].fillna(train_bmi_median)
    val['bmi'] = val['bmi'].fillna(train_bmi_median)
    test['bmi'] = test['bmi'].fillna(train_bmi_median)
    
    print(f"✅ Filled BMI with train median: {train_bmi_median:.2f}")
    return train, val, test

def build_model():
    """Build preprocessing pipeline and model"""
    print("\nBuilding Gradient Boosting model...")
    
    numeric_cols = ['age', 'avg_glucose_level', 'bmi']
    categorical_cols = ['gender', 'hypertension', 'heart_disease', 
                       'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ],
        remainder='drop'
    )
    
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', GradientBoostingClassifier(
            random_state=RANDOM_STATE,
            **BEST_PARAMS
        ))
    ])
    
    print("✅ Model pipeline created (Gradient Boosting)")
    return model

def train_model(model, X_train, y_train):
    """Train the model"""
    print("\nTraining model...")
    model.fit(X_train, y_train)
    print("✅ Model trained")
    return model

def evaluate_model(model, X_val, y_val, X_test, y_test):
    """Evaluate model on validation and test sets"""
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    # Validation set
    y_val_pred = model.predict(X_val)
    y_val_proba = model.predict_proba(X_val)[:, 1]
    
    val_acc = accuracy_score(y_val, y_val_pred)
    val_auc = roc_auc_score(y_val, y_val_proba)
    
    print(f"\nValidation Set:")
    print(f"  Accuracy: {val_acc:.4f}")
    print(f"  ROC AUC:  {val_auc:.4f}")
    
    # Test set
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    test_acc = accuracy_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, y_test_proba)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    
    print(f"\nTest Set (Final):")
    print(f"  Accuracy:  {test_acc:.4f}")
    print(f"  ROC AUC:   {test_auc:.4f}")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall:    {test_recall:.4f}")
    print(f"  F1 Score:  {test_f1:.4f}")
    
    print("\nTest Set Classification Report:")
    print(classification_report(y_test, y_test_pred, target_names=['No Stroke', 'Stroke']))
    print("="*60)
    
    return test_auc

def save_model(model, path):
    """Save trained model"""
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    print(f"\nSaving model to {path}...")
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print("✅ Model saved successfully")
    
    # Verify file was created
    if os.path.exists(path):
        file_size = os.path.getsize(path) / (1024 * 1024)  # Convert to MB
        print(f"✅ Model file verified: {file_size:.2f} MB")
    else:
        print("❌ ERROR: Model file was not created!")

def main():
    """Main training pipeline"""
    print("="*60)
    print("STROKE RISK PREDICTION - MODEL TRAINING")
    print("Model: Gradient Boosting (Tuned)")
    print("="*60)
    
    try:
        # Load data
        df = load_data()
        
        # Prepare data
        train, val, test = prepare_data(df)
        
        # Handle missing values
        train, val, test = handle_missing_values(train, val, test)
        
        # Separate features and target
        X_train = train.drop(columns=['stroke'])
        y_train = train['stroke']
        
        X_val = val.drop(columns=['stroke'])
        y_val = val['stroke']
        
        X_test = test.drop(columns=['stroke'])
        y_test = test['stroke']
        
        # Build and train model
        model = build_model()
        model = train_model(model, X_train, y_train)
        
        # Evaluate
        test_auc = evaluate_model(model, X_val, y_val, X_test, y_test)
        
        # Save model
        save_model(model, MODEL_OUTPUT_PATH)
        
        print("\n" + "="*60)
        print(f"✅ TRAINING COMPLETE")
        print(f"Model: Gradient Boosting (Tuned)")
        print(f"Final Test ROC AUC: {test_auc:.4f}")
        print(f"Model saved: {MODEL_OUTPUT_PATH}")
        print("="*60)
        
    except Exception as e:
        print("\n" + "="*60)
        print("❌ ERROR DURING TRAINING")
        print("="*60)
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        print("="*60)
        sys.exit(1)

if __name__ == "__main__":
    main()