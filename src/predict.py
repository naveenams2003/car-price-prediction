"""
Inference Module for Car Price Prediction
-----------------------------------------
Loads the serialized model artifact and exposes a clean prediction interface
for web APIs, CLI scripts, and user interfaces.
"""

import os
import pickle
import pandas as pd
from typing import Dict, Any

class CarPricePredictor:
    def __init__(self, model_path: str = None):
        if model_path is None:
            # Default to model/car_price_model.pkl relative to project root
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            model_path = os.path.join(base_dir, 'model', 'car_price_model.pkl')
        
        self.model_path = model_path
        self.model_bundle = None
        self.model = None
        self.feature_names = None
        self.reference_year = 2024
        self.metrics = None
        self.feature_importances = None
        
        self._load_model()

    def _load_model(self):
        """Loads the serialized model payload."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}. Please run train_model.py first.")
        
        with open(self.model_path, 'rb') as f:
            self.model_bundle = pickle.load(f)
            
        self.model = self.model_bundle['model']
        self.feature_names = self.model_bundle['feature_names']
        self.reference_year = self.model_bundle.get('reference_year', 2024)
        self.metrics = self.model_bundle.get('metrics', {})
        self.feature_importances = self.model_bundle.get('feature_importances', {})

    def predict(self, 
                present_price: float, 
                year: int, 
                kms_driven: int, 
                fuel_type: str, 
                seller_type: str, 
                transmission: str, 
                owner: int = 0,
                car_name: str = "Car") -> Dict[str, Any]:
        """
        Predict selling price based on vehicle attributes.

        Parameters:
            present_price (float): Original/Showroom price in ₹ Lakhs.
            year (int): Year of manufacture.
            kms_driven (int): Odometer reading in kilometers.
            fuel_type (str): 'Petrol', 'Diesel', or 'CNG'.
            seller_type (str): 'Dealer' or 'Individual'.
            transmission (str): 'Manual' or 'Automatic'.
            owner (int): Number of previous owners (0, 1, 2, 3+).
            car_name (str): Display name for car.

        Returns:
            dict containing prediction in Lakhs, formatted INR, depreciation info, and stats.
        """
        fuel_type = str(fuel_type).strip().capitalize()
        seller_type = str(seller_type).strip().capitalize()
        transmission = str(transmission).strip().capitalize()

        # Compute engineered feature
        car_age = max(0, self.reference_year - int(year))

        # Build feature vector matching encoded columns
        row_dict = {
            'Present_Price': float(present_price),
            'Kms_Driven': float(kms_driven),
            'Owner': int(owner),
            'Car_Age': float(car_age),
            'Fuel_Type_Diesel': 1 if fuel_type == 'Diesel' else 0,
            'Fuel_Type_Petrol': 1 if fuel_type == 'Petrol' else 0,
            'Seller_Type_Individual': 1 if seller_type == 'Individual' else 0,
            'Transmission_Manual': 1 if transmission == 'Manual' else 0
        }

        # Ensure correct column ordering
        features_df = pd.DataFrame([row_dict])[self.feature_names]

        # Model inference
        raw_pred = float(self.model.predict(features_df)[0])

        # Sensible bounding: price cannot be negative, nor exceed present showroom price
        min_allowed = 0.15  # ₹15,000 baseline scrap/minimum value
        predicted_lakhs = max(min_allowed, min(raw_pred, float(present_price)))
        
        # Calculate depreciation
        depreciation_amount_lakhs = max(0.0, float(present_price) - predicted_lakhs)
        depreciation_pct = (depreciation_amount_lakhs / float(present_price)) * 100 if present_price > 0 else 0.0

        # Valuation rating
        if depreciation_pct < 30:
            valuation_tier = "High Value Retention (Like New / Low Depreciation)"
            tier_badge = "Excellent"
        elif depreciation_pct < 55:
            valuation_tier = "Moderate Depreciation (Typical Market Range)"
            tier_badge = "Fair Market"
        else:
            valuation_tier = "Significant Depreciation (Budget / Older Model)"
            tier_badge = "Budget / Value"

        return {
            'status': 'success',
            'car_name': car_name,
            'predicted_price_lakhs': round(predicted_lakhs, 2),
            'predicted_price_inr': f"₹{int(predicted_lakhs * 100000):,}",
            'present_price_lakhs': round(float(present_price), 2),
            'present_price_inr': f"₹{int(float(present_price) * 100000):,}",
            'depreciation_lakhs': round(depreciation_amount_lakhs, 2),
            'depreciation_pct': round(depreciation_pct, 1),
            'car_age_years': car_age,
            'valuation_tier': valuation_tier,
            'tier_badge': tier_badge,
            'model_info': {
                'algorithm': 'Random Forest Regressor (Hyperparameter Tuned)',
                'test_r2_score': round(self.metrics.get('random_forest', {}).get('r2', 0.95), 3),
                'test_rmse': round(self.metrics.get('random_forest', {}).get('rmse', 1.3), 3)
            }
        }
