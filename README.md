# 🚗 Car Price Prediction System (CarDekho)

[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0%2B-black.svg)](https://flask.palletsprojects.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red.svg)](https://streamlit.io/)
[![R2 Score](https://img.shields.io/badge/R%C2%B2%20Score-95.9%25-brightgreen.svg)]()

An end-to-end Machine Learning web application and data pipeline designed to predict the fair selling price of used cars based on historical transaction data from **CarDekho**. The system includes a **Flask REST API Backend**, a responsive **Glassmorphism Web Frontend**, an interactive **Streamlit Dashboard**, a complete **Jupyter Notebook**, and a hyperparameter-tuned **Random Forest Regressor** achieving a **95.9% $R^2$ Score**.

---

## 📌 Table of Contents
1. [Project Overview](#-project-overview)
2. [Workflow Architecture](#-workflow-architecture)
3. [Full-Stack Architecture (Frontend & Backend)](#-full-stack-architecture-frontend--backend)
4. [Dataset & Preprocessing](#-dataset--preprocessing)
5. [Machine Learning Models & Evaluation](#-machine-learning-models--evaluation)
6. [Visualizations & Key Insights](#-visualizations--key-insights)
7. [Directory Structure](#-directory-structure)
8. [Setup & Installation Instructions](#-setup--installation-instructions)
9. [API Documentation](#-api-documentation)
10. [Future Enhancements](#-future-enhancements)

---

## 🚘 Project Overview

When selling or purchasing a pre-owned vehicle, determining an accurate, transparent valuation is challenging due to complex multi-factor depreciation involving age, showroom price, odometer reading, fuel type, transmission, and ownership history.

This project delivers:
- **Accurate Price Valuation**: Estimates selling price in ₹ Lakhs and full Indian Rupees (₹).
- **Depreciation Analysis**: Computes the exact depreciation percentage and retained value.
- **Valuation Tiers**: Categorizes vehicles into *High Value Retention*, *Fair Market*, or *Budget/Value* tiers.
- **Dual User Interfaces**: A custom automotive glassmorphism web app (HTML5/CSS3/JS) and a Streamlit dashboard.

---

## 🔄 Workflow Architecture

```text
              CAR DATASET (CarDekho)
                        ↓
             Data Cleaning & Imputation
                        ↓
        Feature Engineering (Car_Age Calculation)
                        ↓
            Categorical One-Hot Encoding
                        ↓
             80 / 20 Train-Test Split
                        ↓
        ┌───────────────────────────────────┐
        │       Model Comparison & Tuning   │
        │  Linear Regression vs Random Forest│
        └───────────────────────────────────┘
                        ↓
         Hyperparameter Optimization (5-Fold CV)
                        ↓
          Model Evaluation (RMSE, MAE, R²)
                        ↓
            Model Serialization (.pkl)
                        ↓
       ┌─────────────────┴─────────────────┐
       ↓                                   ↓
Flask REST API Backend             Streamlit Web Dashboard
       ↓
Modern Responsive Web Frontend
```

---

## 💻 Full-Stack Architecture (Frontend & Backend)

The project separates concerns cleanly across the presentation, application, and machine learning layers:

### 1. Frontend (`/frontend`)
- **Modern UI**: Automotive dark glassmorphism theme using Plus Jakarta Sans typography.
- **Real-Time Calculation**: Dynamic INR formatting hints while entering showroom prices.
- **Interactive Quick Presets**: One-click chips to pre-fill test vehicles (*Maruti Swift*, *Hyundai i20*, *Honda City*, *Toyota Fortuner*, *Maruti Alto*).
- **Visual Feedback**: Retained value progress bar, price badge indicators, and depreciation loss breakdown.
- **Dynamic Charts**: Interactive horizontal bar chart rendered with **Chart.js** displaying real-time feature importances.

### 2. Backend (`/backend/server.py`)
- **Flask REST API**: Decoupled, CORS-enabled endpoints serving predictions and analytical metadata.
- **Input Validation**: Bounds checking on manufacturing year (1995–2024), non-negative mileage, and valid showroom prices.
- **Static Asset Serving**: Serves the frontend web app and generated charts directly from local routes.

### 3. Streamlit Application (`app.py`)
- Alternative dashboard providing sliders, dropdowns, live inference, and tabs for feature importance, regression plots, and dataset distributions.

---

## 📊 Dataset & Preprocessing

The model is trained on the benchmark **CarDekho Used Car Dataset** (`data/car_data.csv`) comprising 301 records and 9 features:

| Feature Name | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `Car_Name` | Categorical | Brand and model title | *Hyundai i20, Swift* |
| `Year` | Numeric | Year of original manufacture | `2019` |
| `Present_Price` | Numeric | Original ex-showroom price in ₹ Lakhs | `7.50` (₹7,50,000) |
| `Kms_Driven` | Numeric | Total distance driven in kilometers | `35000` |
| `Fuel_Type` | Categorical | Fuel system (`Petrol`, `Diesel`, `CNG`) | `Petrol` |
| `Seller_Type` | Categorical | Vendor channel (`Dealer`, `Individual`) | `Dealer` |
| `Transmission` | Categorical | Gearbox type (`Manual`, `Automatic`) | `Manual` |
| `Owner` | Numeric | Count of previous owners (`0`, `1`, `2`, `3+`) | `0` (First owner) |
| **`Selling_Price`** | **Target** | **Actual market selling price (₹ Lakhs)** | **`5.34`** |

### Preprocessing & Feature Engineering Steps:
1. **Age Derivation**: Converted static manufacturing year into dynamic vehicle age:
   $$\text{Car\_Age} = 2024 - \text{Year}$$
2. **Column Pruning**: Dropped high-cardinality `Car_Name` (since `Present_Price` captures price segment) and redundant `Year`.
3. **One-Hot Encoding**: Converted categorical variables with `drop_first=True` to eliminate dummy variable trap multicollinearity:
   - `Fuel_Type` $\rightarrow$ `Fuel_Type_Diesel`, `Fuel_Type_Petrol` (CNG as reference)
   - `Seller_Type` $\rightarrow$ `Seller_Type_Individual` (Dealer as reference)
   - `Transmission` $\rightarrow$ `Transmission_Manual` (Automatic as reference)
4. **Data Splitting**: 80% training set (240 cars) and 20% holdout test set (61 cars) with fixed `random_state=42`.

---

## 🤖 Machine Learning Models & Evaluation

Two models were developed and compared:
1. **Linear Regression (Baseline)**: Ordinary Least Squares establishing benchmark linear relationships.
2. **Random Forest Regressor (Tuned)**: Non-linear ensemble model optimized via `RandomizedSearchCV` (125 cross-validated fits) exploring tree depth, leaf splits, and estimator count.

### 📈 Evaluation Metrics Summary

| Model | MAE (₹ Lakhs) | RMSE (₹ Lakhs) | $R^2$ Score | Performance |
| :--- | :---: | :---: | :---: | :---: |
| **Linear Regression** | 1.2162 | 1.8652 | 0.8490 (84.9%) | Baseline |
| **Random Forest Regressor (Tuned)** 🏆 | **0.6390** | **0.9661** | **0.9595 (95.9%)** | **Best Fit** |

### Mathematical Metrics:
- **Root Mean Squared Error (RMSE)**: Penalizes large estimation errors:
  $$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2} = \mathbf{0.9661 \text{ Lakhs}}$$
- **Mean Absolute Error (MAE)**: Measures average absolute divergence:
  $$\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i| = \mathbf{0.6390 \text{ Lakhs}}$$
- **Coefficient of Determination ($R^2$)**: Explains variance in selling price:
  $$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2} = \mathbf{0.9595 \ (95.95\%)}$$

---

## 📉 Visualizations & Key Insights

The training script automatically produces high-resolution plots saved in `images/`:

1. **`images/feature_importance.png`**:
   - **Key Finding**: `Present_Price` is the single strongest predictor (~86.5% relative weight), followed by `Car_Age` (~8.2%) and `Kms_Driven` (~3.4%).
2. **`images/actual_vs_predicted.png`**:
   - Demonstrates close clustering around the $y = x$ ideal line, confirming minimal residual variance.
3. **`images/price_distribution.png`**:
   - Reveals right-skewed pricing distribution typical of Indian passenger car markets.
4. **`images/correlation_heatmap.png`**:
   - Illustrates strong positive correlation between Showroom Price and Resale Price ($r = 0.88$) and negative correlation with Car Age ($r = -0.24$).

---

## 📁 Directory Structure

```text
car-price-prediction/
│
├── data/
│   └── car_data.csv                   # CarDekho used cars dataset (301 rows, 9 cols)
│
├── notebooks/
│   └── car_price_prediction.ipynb     # Step-by-step EDA & modeling Jupyter notebook
│
├── src/
│   ├── __init__.py
│   ├── train_model.py                 # Pipeline: data prep, training, tuning, plotting, saving
│   └── predict.py                     # Standalone prediction engine with safety bounding
│
├── backend/
│   ├── __init__.py
│   └── server.py                      # Flask REST API backend + static file server
│
├── frontend/
│   ├── index.html                     # Responsive glassmorphism web interface
│   ├── style.css                      # Custom automotive theme styling
│   └── app.js                         # Dynamic form logic, presets, API calls, Chart.js
│
├── app.py                             # Interactive Streamlit web application
│
├── model/
│   └── car_price_model.pkl            # Serialized model artifact & feature metadata
│
├── images/
│   ├── price_distribution.png         # Target & present price distribution plot
│   ├── actual_vs_predicted.png        # Scatter plot (test set vs predictions)
│   ├── feature_importance.png         # Feature importance horizontal bar chart
│   └── correlation_heatmap.png        # Feature correlation matrix
│
├── run_app.bat                        # One-click Windows application launcher
├── requirements.txt                   # Dependency specifications
└── README.md                          # Comprehensive project documentation
```

---

## 🚀 Setup & Installation Instructions

### Prerequisites
- Python 3.9+ installed
- Git or ZIP extractor

### Step 1: Clone or Navigate to Project
```bash
cd car-price-prediction
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: (Optional) Retrain Model
To rerun preprocessing, hyperparameter search, and regenerate plots:
```bash
python src/train_model.py
```

### Step 4: Run the Application

#### Option A: Run Full-Stack Web Application (Frontend + Backend)
```bash
python backend/server.py
```
Then open your web browser at:
```
http://localhost:5000
```

#### Option B: Run Streamlit Application
```bash
streamlit run app.py
```
Then open your web browser at:
```
http://localhost:8501
```

#### Option C: Windows One-Click Launcher
Double-click `run_app.bat` or execute in terminal:
```cmd
run_app.bat
```

---

## 🔌 API Documentation

The backend exposes the following RESTful endpoints:

### 1. Health Check
- **Endpoint**: `GET /api/health`
- **Response**:
```json
{
  "model_loaded": true,
  "service": "Car Price Prediction API",
  "status": "healthy",
  "version": "1.0.0"
}
```

### 2. Vehicle Valuation Prediction
- **Endpoint**: `POST /api/predict`
- **Headers**: `Content-Type: application/json`
- **Request Body**:
```json
{
  "car_name": "Hyundai i20 Asta",
  "present_price": 7.50,
  "year": 2019,
  "kms_driven": 35000,
  "fuel_type": "Petrol",
  "seller_type": "Dealer",
  "transmission": "Manual",
  "owner": 0
}
```
- **Response**:
```json
{
  "car_age_years": 5,
  "car_name": "Hyundai i20 Asta",
  "depreciation_lakhs": 2.16,
  "depreciation_pct": 28.8,
  "predicted_price_inr": "₹5,34,000",
  "predicted_price_lakhs": 5.34,
  "present_price_inr": "₹7,50,000",
  "present_price_lakhs": 7.5,
  "status": "success",
  "tier_badge": "Excellent",
  "valuation_tier": "High Value Retention (Like New / Low Depreciation)"
}
```

### 3. Evaluation Metrics
- **Endpoint**: `GET /api/metrics`
- **Response**: Returns $R^2$, RMSE, and MAE comparing Linear Regression and Random Forest.

---

## 🔮 Future Enhancements

1. **Brand & Model Embeddings**: Incorporate high-cardinality vehicle brand and trim levels using target encoding or learned neural embeddings.
2. **City & Regional Demand**: Add location-based price differentials across tier-1, tier-2, and tier-3 automotive markets.
3. **Advanced Ensemble Models**: Compare with XGBoost, LightGBM, and CatBoost regressors with Bayesian hyperparameter optimization.
4. **Computer Vision Damage Assessment**: Allow users to upload car exterior photos to adjust depreciation based on scratch or dent severity.
5. **Cloud Deployment**: Containerize with Docker and deploy to AWS ECS, Google Cloud Run, or Render.
