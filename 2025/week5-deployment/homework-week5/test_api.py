import requests
import time

# Wait for server to start, then test
time.sleep(2)

url = "http://localhost:8000/predict"
client = {
    "lead_source": "organic_search",
    "number_of_courses_viewed": 4,
    "annual_income": 80304.0
}

try:
    response = requests.post(url, json=client)
    result = response.json()
    probability = result['probability']
    
    print(f"Q4 - Probability: {probability:.3f}")
    
    # Check against options: 0.334, 0.534, 0.734, 0.934
    options_q4 = [0.334, 0.534, 0.734, 0.934]
    closest_q4 = min(options_q4, key=lambda x: abs(x - probability))
    print(f"Q4 Answer: {closest_q4}")
    
except Exception as e:
    print(f"Error: {e}")
    print("Make sure FastAPI server is running: uvicorn app:app --host 0.0.0.0 --port 8000")