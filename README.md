# Car Price Prediction API

This project is a FastAPI-based web service that provides car price predictions based on various vehicle features. Additionally, it includes an endpoint to fetch automotive news from an external API.

## Features

- **Car Price Prediction**: Predicts the price of a car based on its attributes.
- **Automotive News Fetching**: Retrieves automotive industry news from an external API.
- **CORS Support**: Allows communication with a frontend application.

## Requirements

Before running the project, ensure you have the required dependencies installed.

### Prerequisites

- Python 3.8+
- `pip` (Python package manager)

### Install Dependencies

Run the following command to install the required Python packages:

```bash
pip install -r requirements.txt
```

## Running the Application

You can start the FastAPI server using `uvicorn`:

```bash
uvicorn main:app --host 127.0.0.1 --port 5050 --reload
```

### API Endpoints

#### **1. Car Price Prediction**

- **Endpoint:** `POST /predict`
- **Description:** Predicts the price of a car based on provided features.
- **Request Body:**

```json
{
  "make": "Toyota",
  "model": "Corolla",
  "trim": "LE",
  "body_type": "Sedan",
  "drivetrain": "FWD",
  "transmission": "Automatic",
  "miles": 50000,
  "car_age": 5,
  "engine_size": 1.8
}
```

- **Response Example:**

```json
{
  "predicted_price": 15000
}
```

#### **2. Fetch Automotive News**

- **Endpoint:** `GET /news`
- **Description:** Retrieves the latest automotive industry news from an external API.

## Environment Variables

Create a `.env` file in the project directory and add your API key for the external news API:

```env
API_TUBE_KEY=your_api_key_here
```

## Getting an API Key

To get an API key, register with APITube at [https://apitube.io/#sign-in](https://apitube.io/#sign-in). The API key will be automatically generated and available in the Dashboard.

## Directory Structure

```
project_root/
│── main.py         # Main FastAPI application file
│── model/
│   ├── car_price_model.pkl   # Trained ML model
│   ├── label_encoders.pkl    # Encoders for categorical features
│   ├── scaler.pkl            # Scaler for numerical features
│   ├── feature_order.json    # Feature order for model input
│── requirements.txt  # List of dependencies
│── .env              # API key configuration
```

## Notes

- Ensure all model files are placed in the `model/` directory.
- The application is designed to work with a frontend that interacts via API requests.
