import streamlit as st
import requests
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Stroke Risk Predictor",
    page_icon="🏥",
    layout="wide"
)

# Title
st.title("🏥 Stroke Risk Prediction System")
st.markdown("Enter patient information below to predict stroke risk")
st.markdown("---")

# API endpoint
API_URL = "http://localhost:9696/predict"

# Create two columns for input
col1, col2 = st.columns(2)

# ============================================================
# LEFT COLUMN - Demographic Variables
# ============================================================
with col1:
    st.subheader("📋 Demographic Information")
    
    # Variable 1: Gender
    gender = st.selectbox(
        "Gender *",
        options=["Male", "Female", "Other"],
        index=0,
        help="Select patient's gender"
    )
    
    # Variable 2: Age
    age = st.slider(
        "Age (years) *",
        min_value=0,
        max_value=100,
        value=50,
        step=1,
        help="Patient's age in years"
    )
    
    # Variable 3: Ever Married
    ever_married = st.radio(
        "Ever Married *",
        options=["Yes", "No"],
        horizontal=True,
        help="Has the patient ever been married?"
    )
    
    # Variable 4: Work Type
    work_type = st.selectbox(
        "Work Type *",
        options=["Private", "Self-employed", "Govt_job", "children", "Never_worked"],
        index=0,
        help="Patient's type of employment"
    )
    
    # Variable 5: Residence Type
    residence_type = st.radio(
        "Residence Type *",
        options=["Urban", "Rural"],
        horizontal=True,
        help="Type of residence area"
    )

# ============================================================
# RIGHT COLUMN - Health Variables
# ============================================================
with col2:
    st.subheader("🩺 Health Metrics")
    
    # Variable 6: Hypertension
    hypertension = st.radio(
        "Hypertension *",
        options=["No", "Yes"],
        horizontal=True,
        help="Does the patient have high blood pressure?"
    )
    hypertension = 1 if hypertension == "Yes" else 0
    
    # Variable 7: Heart Disease
    heart_disease = st.radio(
        "Heart Disease *",
        options=["No", "Yes"],
        horizontal=True,
        help="Does the patient have any heart disease?"
    )
    heart_disease = 1 if heart_disease == "Yes" else 0
    
    # Variable 8: Average Glucose Level
    avg_glucose_level = st.number_input(
        "Average Glucose Level (mg/dL) *",
        min_value=50.0,
        max_value=300.0,
        value=100.0,
        step=1.0,
        help="Average glucose level in blood (normal: 70-100)"
    )
    
    # Variable 9: BMI
    bmi = st.number_input(
        "BMI (Body Mass Index) *",
        min_value=10.0,
        max_value=60.0,
        value=25.0,
        step=0.1,
        help="Body Mass Index (normal: 18.5-24.9)"
    )
    
    # Variable 10: Smoking Status
    smoking_status = st.selectbox(
        "Smoking Status *",
        options=["never smoked", "formerly smoked", "smokes", "Unknown"],
        index=0,
        help="Patient's smoking history"
    )

st.markdown("---")

# ============================================================
# Display Current Variable Values
# ============================================================
with st.expander("👁️ View Current Input Values"):
    st.markdown("**Variables that will be passed to the API:**")
    
    variable_data = {
        "Variable": [
            "gender", "age", "hypertension", "heart_disease", 
            "ever_married", "work_type", "Residence_type", 
            "avg_glucose_level", "bmi", "smoking_status"
        ],
        "Value": [
            gender, age, hypertension, heart_disease,
            ever_married, work_type, residence_type,
            avg_glucose_level, bmi, smoking_status
        ],
        "Type": [
            type(gender).__name__, type(age).__name__, 
            type(hypertension).__name__, type(heart_disease).__name__,
            type(ever_married).__name__, type(work_type).__name__, 
            type(residence_type).__name__, type(avg_glucose_level).__name__,
            type(bmi).__name__, type(smoking_status).__name__
        ]
    }
    
    df_vars = pd.DataFrame(variable_data)
    st.dataframe(df_vars, use_container_width=True)

# ============================================================
# Predict Button - Pass Variables to API
# ============================================================
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])

with col_btn2:
    predict_button = st.button(
        "🔮 Predict Stroke Risk",
        type="primary",
        use_container_width=True
    )

if predict_button:
    # ============================================================
    # CREATE DICTIONARY WITH ALL VARIABLES
    # ============================================================
    patient_data = {
        "gender": gender,                          # ← Variable passed
        "age": float(age),                         # ← Variable passed
        "hypertension": int(hypertension),         # ← Variable passed
        "heart_disease": int(heart_disease),       # ← Variable passed
        "ever_married": ever_married,              # ← Variable passed
        "work_type": work_type,                    # ← Variable passed
        "Residence_type": residence_type,          # ← Variable passed
        "avg_glucose_level": float(avg_glucose_level),  # ← Variable passed
        "bmi": float(bmi),                         # ← Variable passed
        "smoking_status": smoking_status           # ← Variable passed
    }
    
    st.markdown("---")
    
    # Show what's being sent
    with st.expander("📤 Data Being Sent to API"):
        st.json(patient_data)
    
    # ============================================================
    # SEND VARIABLES TO FLASK API
    # ============================================================
    with st.spinner("🔄 Sending data to prediction API..."):
        try:
            # Make POST request with variables
            response = requests.post(
                API_URL,
                json=patient_data  # ← Variables passed here as JSON
            )
            
            if response.status_code == 200:
                # ============================================================
                # RECEIVE RESPONSE VARIABLES FROM API
                # ============================================================
                result = response.json()
                
                st.success("✅ Prediction received successfully!")
                
                # Display results
                st.markdown("---")
                st.subheader("📊 Prediction Results")
                
                # Show received variables
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                
                with metric_col1:
                    st.metric(
                        label="Stroke Probability",
                        value=f"{result['stroke_probability']:.2%}",  # ← Variable received
                        help="Probability of stroke occurrence"
                    )
                
                with metric_col2:
                    prediction_value = result['stroke_prediction']  # ← Variable received
                    prediction_text = "⚠️ STROKE RISK" if prediction_value == 1 else "✅ NO STROKE RISK"
                    st.metric(
                        label="Prediction",
                        value=prediction_text,
                        help="Binary prediction (0 or 1)"
                    )
                
                with metric_col3:
                    risk_level = result['risk_level']  # ← Variable received
                    st.metric(
                        label="Risk Level",
                        value=risk_level,
                        help="Categorized risk level"
                    )
                
                # Risk level interpretation
                st.markdown("---")
                risk_prob = result['stroke_probability']
                
                if risk_level == "VERY HIGH":
                    st.error(f"""
                    ### ⚠️ VERY HIGH RISK
                    **Probability:** {risk_prob:.2%}
                    
                    **Recommendation:** Immediate medical consultation recommended.
                    """)
                elif risk_level == "HIGH":
                    st.warning(f"""
                    ### ⚠️ HIGH RISK
                    **Probability:** {risk_prob:.2%}
                    
                    **Recommendation:** Schedule medical evaluation soon.
                    """)
                elif risk_level == "MEDIUM":
                    st.info(f"""
                    ### ℹ️ MEDIUM RISK
                    **Probability:** {risk_prob:.2%}
                    
                    **Recommendation:** Regular health monitoring advised.
                    """)
                else:
                    st.success(f"""
                    ### ✅ LOW RISK
                    **Probability:** {risk_prob:.2%}
                    
                    **Recommendation:** Maintain healthy lifestyle.
                    """)
                
                # Show complete API response
                with st.expander("📥 Complete API Response"):
                    st.json(result)
            
            else:
                st.error(f"❌ API Error: Status code {response.status_code}")
                st.json(response.json())
        
        except requests.exceptions.ConnectionError:
            st.error("""
            ### ❌ Connection Error
            
            Could not connect to the Flask API. Please ensure:
            
            1. Flask API is running: `python app/main.py`
            2. API is accessible at: http://localhost:9696
            
            **Start Flask API in another terminal:**
            ```bash
            cd app
            python main.py
            ```
            """)
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# ============================================================
# Sidebar - Quick Test Profiles
# ============================================================
with st.sidebar:
    st.header("🚀 Quick Actions")
    
    st.markdown("### Test Profiles")
    st.markdown("Load pre-filled patient data:")
    
    if st.button("👨 High Risk Profile", use_container_width=True):
        st.session_state['test_profile'] = 'high'
        st.rerun()
    
    if st.button("👩 Low Risk Profile", use_container_width=True):
        st.session_state['test_profile'] = 'low'
        st.rerun()
    
    if st.button("🔄 Clear Form", use_container_width=True):
        st.session_state.clear()
        st.rerun()
    
    st.markdown("---")
    
    st.markdown("### ℹ️ Variable Information")
    st.markdown("""
    **10 Variables Required:**
    1. Gender (categorical)
    2. Age (numeric)
    3. Hypertension (binary)
    4. Heart Disease (binary)
    5. Ever Married (categorical)
    6. Work Type (categorical)
    7. Residence Type (categorical)
    8. Avg Glucose Level (numeric)
    9. BMI (numeric)
    10. Smoking Status (categorical)
    """)
    
    st.markdown("---")
    
    st.markdown("### 📊 Model Info")
    st.markdown("""
    **Algorithm:** Gradient Boosting
    
    **Performance:**
    - ROC AUC: 0.8383
    - Accuracy: 0.7456
    
    **Backend:** Flask API
    
    **Frontend:** Streamlit
    """)

# ============================================================
# Handle Quick Test Profiles
# ============================================================
if 'test_profile' in st.session_state:
    if st.session_state['test_profile'] == 'high':
        st.info("📝 Loaded High Risk Profile - Click Predict!")
        # Variables auto-filled via session state
    elif st.session_state['test_profile'] == 'low':
        st.info("📝 Loaded Low Risk Profile - Click Predict!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p><strong>ML Zoomcamp 2025 - Stroke Risk Prediction</strong></p>
    <p>Built with Streamlit (Frontend) + Flask (Backend) + Gradient Boosting (ML Model)</p>
</div>
""", unsafe_allow_html=True)