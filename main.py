import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import pandas as pd
import json
from pydantic import BaseModel


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

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5050)
