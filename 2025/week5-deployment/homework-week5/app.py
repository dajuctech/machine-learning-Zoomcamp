import pickle
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# Load the model
with open('pipeline_v1.bin', 'rb') as f:
    pipeline = pickle.load(f)

app = FastAPI()

class LeadData(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float

@app.get("/")
def read_root():
    return {"message": "Lead Scoring API - Week 5 Homework"}

@app.post("/predict")
def predict(lead: LeadData):
    # Convert to dictionary
    lead_dict = lead.dict()
    
    # Get probability
    probability = pipeline.predict_proba([lead_dict])[0, 1]
    
    return {
        "probability": float(probability),
        "converted": probability > 0.5
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)