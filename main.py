import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import pandas as pd
import json
from pydantic import BaseModel
import os
import requests
from dotenv import load_dotenv
import re


model = joblib.load("model/car_price_model.pkl")
label_encoders = joblib.load("model/label_encoders.pkl")
scaler = joblib.load("model/scaler.pkl")

with open("model/feature_order.json", "r") as f:
    feature_order = json.load(f)  

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CarFeatures(BaseModel):
    make: str
    model: str
    trim: str
    body_type: str
    drivetrain: str
    transmission: str
    miles: int
    car_age: int
    engine_size: float

@app.post("/predict")
def predict_price(car: CarFeatures):
    try:
        car_df = pd.DataFrame([car.dict()])

        for col in label_encoders:
            if col in car_df.columns:
                if car_df[col].iloc[0] in label_encoders[col].classes_:
                    car_df[col] = label_encoders[col].transform(car_df[col])
                else:
                    car_df[col] = -1

        num_features = ["miles", "car_age", "engine_size"]
        car_df[num_features] = scaler.transform(car_df[num_features])

        car_df = car_df[feature_order]

        input_data = car_df.to_numpy()

        predicted_price = model.predict(input_data)[0]

        predicted_price = float(round(predicted_price, -2))

        return {"predicted_price": predicted_price}

    except Exception as e:
        return {"error": str(e)}

load_dotenv()
API_TUBE_KEY = os.getenv("API_TUBE_KEY")
API_TUBE_URL = "https://api.apitube.io/v1/news/everything"

def normalize_text(text, length):
    return re.sub(r'\s+', ' ', text.strip().lower())[:length]

def remove_duplicates(articles):
    seen_titles = set()
    seen_hrefs = set()
    seen_descriptions = set()
    unique_articles = []

    for article in articles:
        title = normalize_text(article.get("title", ""), 25)
        href = article.get("href", "").strip()
        description = normalize_text(article.get("description", ""), 50)

        if not title or not href:
            continue

        if title in seen_titles or href in seen_hrefs or description in seen_descriptions:
            continue

        seen_titles.add(title)
        seen_hrefs.add(href)
        seen_descriptions.add(description)
        unique_articles.append(article)

    return unique_articles


@app.get("/news")
def get_news():
    try:
        response = requests.get(API_TUBE_URL, params={
            "topic.id": "industry.automotive_news",
            "language.code": "en",
            "per_page": 75,
            "api_key": API_TUBE_KEY
        })

        if response.status_code != 200:
            return {"error": f"Failed to fetch news, status code: {response.status_code}"}

        news_data = response.json()

        if "results" not in news_data or not news_data["results"]:
            return {"error": "No articles found"}

        filtered_articles = remove_duplicates(news_data["results"])

        return {"results": filtered_articles}

    except requests.exceptions.RequestException as e:
        return {"error": f"API request failed: {str(e)}"}

    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5050)
