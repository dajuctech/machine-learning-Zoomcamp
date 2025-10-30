import subprocess
import time
import requests
import os

print("="*60)
print("WEEK 5 DEPLOYMENT HOMEWORK - COMPLETE SOLUTION")
print("="*60)

# Q1: uv version (run manually)
print("\nQ1: Check uv version with: uv --version")

# Q2: Scikit-learn hash (check manually)  
print("Q2: Check scikit-learn hash with: cat uv.lock | grep -A 10 scikit-learn")

# Q3: Model scoring
print("\nQ3: Testing model scoring...")
exec(open('score_model.py').read())

# Q4: FastAPI test
print("\nQ4: Testing FastAPI...")
print("Start server with: uvicorn app:app --host 0.0.0.0 --port 8000")
print("Then run: python test_api.py")

# Q5: Docker image size
print("\nQ5: Check Docker image size with: docker images agrigorev/zoomcamp-model:2025")

# Q6: Docker container test  
print("\nQ6: Build and test Docker container...")
print("Build with: docker build -t week5-homework .")
print("Run with: docker run -p 8000:8000 week5-homework")
print("Test with: python test_docker.py")

print("\n🎯 All scripts ready! Run each step manually for homework submission.")