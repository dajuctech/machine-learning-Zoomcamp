# scripts/download_data.py
import os
import urllib.request

os.makedirs("data/raw", exist_ok=True)

#url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00296/diabetic_data.csv"
url = "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip"
dest_path = "data/raw/diabetic_data.csv"

print("Downloading dataset...")
urllib.request.urlretrieve(url, dest_path)
print(f"Saved to {dest_path}")

