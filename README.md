# 📈 XAU/USD Direction Classifier

> Machine Learning project for predicting the next-hour direction of XAU/USD (Gold vs US Dollar) using engineered technical features and a FastAPI inference service.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-API-green)
![Pytest](https://img.shields.io/badge/Pytest-Testing-yellow)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

# Project Overview

Financial markets are highly dynamic and difficult to predict. This project develops an end-to-end machine learning pipeline that predicts the **next-hour movement of XAU/USD** using historical price action and engineered technical features.

The project covers the complete machine learning workflow:

- Data preprocessing
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Model Training
- Model Evaluation
- Model Analysis
- FastAPI Deployment
- Automated API Testing

Rather than focusing only on model accuracy, the project emphasizes building a production-style machine learning workflow that is modular, reproducible, and deployable.

---

# Project Workflow

```
Historical XAU/USD Data
          │
          ▼
Data Cleaning & Processing
          │
          ▼
Feature Engineering
(26 Technical Features)
          │
          ▼
Chronological Train/Test Split
          │
          ▼
Model Training
(Logistic Regression & Gradient Boosting)
          │
          ▼
Model Evaluation
          │
          ▼
Model Analysis
          │
          ▼
FastAPI Deployment
          │
          ▼
REST API Prediction
```

---

# Dataset

- Asset: **XAU/USD (Gold vs US Dollar)**
- Frequency: **1 Hour**
- Historical Market Data
- Time-series forecasting setup
- Chronological train-test split to prevent data leakage

---

# Feature Engineering

The final model uses **26 engineered features**, including:

### Price Returns

- return_1h
- return_3h
- return_6h
- return_12h
- return_24h

### Momentum

- momentum_3h
- momentum_6h
- momentum_12h
- momentum_24h

### Candle Features

- body_size
- range_size
- upper_wick_ratio
- lower_wick_ratio
- up_candle_ratio_12h

### Technical Indicators

- RSI (14)
- ATR Ratio
- SMA 6 Ratio
- SMA 12 Ratio
- SMA 24 Ratio

### Volatility Features

- volatility_6h
- volatility_12h
- volatility_24h

### Trend Features

- trend_strength_24h

### Time Features

- hour
- day_of_week

### Price Position

- close_position

---

# Machine Learning Models

The following models were evaluated:

- Logistic Regression
- Gradient Boosting Classifier

The final deployed model is:

**Logistic Regression**

Selected due to stable performance, simplicity, interpretability, and consistent generalization on the chronological test set.

---

# Model Performance

| Metric | Score |
|---------|-------|
| Accuracy | **0.5079** |
| Precision | **0.5251** |
| Recall | **0.4392** |
| F1 Score | **0.4784** |

---

# Key Findings

- Performance is slightly above random classification.
- Financial market direction remains inherently difficult to predict.
- Prediction confidence is generally concentrated near the decision threshold.
- The model demonstrates stable behaviour across the chronological test period.

---

# FastAPI Deployment

The trained model is deployed as a REST API using **FastAPI**.

### Available Endpoints

### Root

```
GET /
```

Returns API status.

---

### Health Check

```
GET /health
```

Returns:

- API status
- Model loading status
- Number of engineered features

---

### Prediction

```
POST /predict
```

Accepts engineered feature values and returns:

```json
{
  "predicted_class": 1,
  "predicted_direction": "UP",
  "confidence": 0.5626
}
```

---

# API Documentation

Interactive Swagger documentation is automatically available at:

```
http://127.0.0.1:8000/docs
```

---

# Automated Testing

API endpoints are tested using **pytest**.

Verified endpoints:

- Root endpoint
- Health endpoint
- Prediction endpoint

All API tests pass successfully.

---

# Project Structure

```
📦 xauusd-direction-classifier
│
├── 📂 api
│   ├── __init__.py
│   └── main.py
│
├── 📂 data
│   ├── raw
│   └── processed
├── 📂 images
├── 📂 models
│
├── 📂 notebooks
│
├── 📂 src
│
├── 📂 tests
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/yourusername/xauusd-direction-classifier.git

cd xauusd-direction-classifier
```

Create virtual environment

```bash
python -m venv .venv
```

Activate environment

Windows

```bash
.venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Run the API

```bash
uvicorn api.main:app --reload
```

Open

```
http://127.0.0.1:8000/docs
```

---

# Run Tests

```bash
pytest tests/test_api.py -v
```

---

## 🚀 Images

images/banner.png
images/eda.png
images/model_analysis.png
images/swagger.png
images/prediction_api.png

---


## 🛠 Tech Stack

| Category | Technologies |
|----------|--------------|
| Language | Python |
| ML | Scikit-Learn |
| API | FastAPI |
| Testing | Pytest |
| Visualization | Matplotlib |
| Data | Pandas, NumPy |
| Deployment | Uvicorn |

---

# Future Improvements

## 🚀 Future Work

- Hyperparameter Optimization
- XGBoost & LightGBM
- LSTM-based Forecasting
- Transformer Models
- Live Market Prediction
- Docker Deployment
- CI/CD Pipeline
- Cloud Deployment (AWS)
- Real-time Dashboard
---

# Limitations

- Financial markets exhibit significant randomness and non-stationarity.
- The model should not be interpreted as a trading strategy.
- Prediction performance remains only slightly above random classification.
- Additional market indicators and macroeconomic features could improve performance.

---

# License

This project is provided for educational and research purposes.
