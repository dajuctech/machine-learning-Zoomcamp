import pickle
import numpy as np

# Load the pipeline
with open('pipeline_v1.bin', 'rb') as f:
    pipeline = pickle.load(f)

# Score the record for Q3
record = {
    "lead_source": "paid_ads",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0
}

# Get probability
probability = pipeline.predict_proba([record])[0, 1]
print(f"Q3 - Probability of conversion: {probability:.3f}")

# Check against options: 0.333, 0.533, 0.733, 0.933
options_q3 = [0.333, 0.533, 0.733, 0.933]
closest_q3 = min(options_q3, key=lambda x: abs(x - probability))
print(f"Q3 Answer: {closest_q3}")