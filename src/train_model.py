"""
Car Price Prediction - Model Training Pipeline
----------------------------------------------
Author: Machine Learning Engineer
Dataset: CarDekho Used Car Dataset
Description:
    This script loads the CarDekho dataset, performs exploratory data analysis,
    feature engineering, data preprocessing, trains both Linear Regression
    and Random Forest Regressor models (with hyperparameter tuning),
    evaluates their performance (MAE, RMSE, R2), generates publication-ready
    visualizations, and serializes the best performing model.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 300

REFERENCE_YEAR = 2024

def load_data(filepath: str) -> pd.DataFrame:
    """Load dataset from CSV file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    df = pd.read_csv(filepath)
    print(f"[*] Dataset successfully loaded from {filepath}")
    print(f"[*] Raw dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def preprocess_data(df: pd.DataFrame):
    """
    Perform cleaning, feature engineering, and encoding.
    
    1. Feature Engineering: Compute Car_Age = REFERENCE_YEAR - Year
    2. Drop Car_Name and Year columns
    3. One-hot encode Fuel_Type, Seller_Type, Transmission (drop_first=True)
    """
    data = df.copy()
    
    # Check for missing values
    missing = data.isnull().sum().sum()
    print(f"[*] Total missing values found: {missing}")
    
    # Feature Engineering: Age of the car
    data['Car_Age'] = REFERENCE_YEAR - data['Year']
    
    # Drop irrelevant columns for direct tabular regression
    data.drop(columns=['Car_Name', 'Year'], inplace=True)
    
    # One-Hot Encoding for categorical features
    # Fuel_Type: Petrol, Diesel, CNG -> Diesel, Petrol (CNG dropped as reference)
    # Seller_Type: Dealer, Individual -> Individual (Dealer dropped as reference)
    # Transmission: Manual, Automatic -> Manual (Automatic dropped as reference)
    data_encoded = pd.get_dummies(data, drop_first=True, dtype=int)
    
    print(f"[*] Encoded dataset shape: {data_encoded.shape[0]} rows, {data_encoded.shape[1]} columns")
    print(f"[*] Features: {list(data_encoded.columns)}")
    
    return data_encoded

def generate_visualizations(df_raw: pd.DataFrame, df_encoded: pd.DataFrame, 
                            y_test, y_pred_rf, feature_names, importances, output_dir: str):
    """Generate and save project charts."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Price Distribution Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(df_raw['Selling_Price'], kde=True, color='#2563eb', ax=axes[0], bins=25)
    axes[0].set_title('Distribution of Selling Price (₹ Lakhs)', fontsize=13, fontweight='bold', pad=10)
    axes[0].set_xlabel('Selling Price (₹ Lakhs)')
    axes[0].set_ylabel('Car Count')

    sns.histplot(df_raw['Present_Price'], kde=True, color='#059669', ax=axes[1], bins=25)
    axes[1].set_title('Distribution of Present / Showroom Price (₹ Lakhs)', fontsize=13, fontweight='bold', pad=10)
    axes[1].set_xlabel('Present Price (₹ Lakhs)')
    axes[1].set_ylabel('Car Count')

    plt.tight_layout()
    dist_path = os.path.join(output_dir, 'price_distribution.png')
    plt.savefig(dist_path, bbox_inches='tight')
    plt.close()
    print(f"[*] Saved: {dist_path}")

    # 2. Actual vs Predicted Price Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred_rf, color='#2563eb', alpha=0.7, edgecolors='k', s=65, label='Predicted Points')
    min_val = min(y_test.min(), y_pred_rf.min())
    max_val = max(y_test.max(), y_pred_rf.max())
    plt.plot([min_val, max_val], [min_val, max_val], color='#dc2626', linestyle='--', linewidth=2, label='Perfect Prediction (y=x)')
    plt.title('Actual vs Predicted Selling Price (Random Forest)', fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('Actual Selling Price (₹ Lakhs)', fontsize=11)
    plt.ylabel('Predicted Selling Price (₹ Lakhs)', fontsize=11)
    plt.legend(frameon=True)
    plt.tight_layout()
    scatter_path = os.path.join(output_dir, 'actual_vs_predicted.png')
    plt.savefig(scatter_path, bbox_inches='tight')
    plt.close()
    print(f"[*] Saved: {scatter_path}")

    # 3. Feature Importance Plot
    feat_series = pd.Series(importances, index=feature_names).sort_values(ascending=True)
    plt.figure(figsize=(9, 5))
    bars = plt.barh(feat_series.index, feat_series.values, color='#3b82f6', edgecolor='#1d4ed8')
    plt.title('Feature Importances in Car Price Prediction', fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('Relative Importance Score', fontsize=11)
    plt.ylabel('Features', fontsize=11)
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 0.01, bar.get_y() + bar.get_height() / 2, f'{w:.3f}', ha='left', va='center', fontsize=9)
    plt.xlim(0, max(feat_series.values) * 1.15)
    plt.tight_layout()
    feat_path = os.path.join(output_dir, 'feature_importance.png')
    plt.savefig(feat_path, bbox_inches='tight')
    plt.close()
    print(f"[*] Saved: {feat_path}")

    # 4. Correlation Heatmap (Bonus insight)
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_encoded.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix of Processed Features', fontsize=14, fontweight='bold', pad=12)
    plt.tight_layout()
    corr_path = os.path.join(output_dir, 'correlation_heatmap.png')
    plt.savefig(corr_path, bbox_inches='tight')
    plt.close()
    print(f"[*] Saved: {corr_path}")

def train_and_evaluate(base_dir: str):
    """Main training routine."""
    data_path = os.path.join(base_dir, 'data', 'car_data.csv')
    model_dir = os.path.join(base_dir, 'model')
    images_dir = os.path.join(base_dir, 'images')
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    # 1. Load Data
    df_raw = load_data(data_path)

    # 2. Preprocess Data
    df_encoded = preprocess_data(df_raw)

    # 3. Train-Test Split
    X = df_encoded.drop(columns=['Selling_Price'])
    y = df_encoded['Selling_Price']
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"[*] Training sample size: {X_train.shape[0]}, Testing sample size: {X_test.shape[0]}")

    # -------------------------------------------------------------
    # 4. Model 1: Linear Regression (Baseline)
    # -------------------------------------------------------------
    print("\n" + "="*50)
    print("Training Model 1: Linear Regression")
    print("="*50)
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)

    mae_lr = mean_absolute_error(y_test, y_pred_lr)
    mse_lr = mean_squared_error(y_test, y_pred_lr)
    rmse_lr = np.sqrt(mse_lr)
    r2_lr = r2_score(y_test, y_pred_lr)

    print(f"Linear Regression Results:")
    print(f"  - MAE:  {mae_lr:.4f} Lakhs")
    print(f"  - RMSE: {rmse_lr:.4f} Lakhs")
    print(f"  - R² Score: {r2_lr:.4f}")

    # -------------------------------------------------------------
    # 5. Model 2: Random Forest Regressor (Tuned)
    # -------------------------------------------------------------
    print("\n" + "="*50)
    print("Training Model 2: Random Forest Regressor with Hyperparameter Tuning")
    print("="*50)

    # Hyperparameter Grid
    param_grid = {
        'n_estimators': [100, 150, 200, 250, 300],
        'max_features': [1.0, 'sqrt', 'log2'],
        'max_depth': [5, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    rf = RandomForestRegressor(random_state=42)
    rf_random = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_grid,
        scoring='neg_mean_squared_error',
        n_iter=25,
        cv=5,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    rf_random.fit(X_train, y_train)

    best_rf_model = rf_random.best_estimator_
    print(f"[*] Best Random Forest Parameters: {rf_random.best_params_}")

    y_pred_rf = best_rf_model.predict(X_test)

    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    rmse_rf = np.sqrt(mse_rf)
    r2_rf = r2_score(y_test, y_pred_rf)

    print(f"\nRandom Forest Results:")
    print(f"  - MAE:  {mae_rf:.4f} Lakhs")
    print(f"  - RMSE: {rmse_rf:.4f} Lakhs")
    print(f"  - R² Score: {r2_rf:.4f}")

    # -------------------------------------------------------------
    # 6. Comparison Summary
    # -------------------------------------------------------------
    print("\n" + "="*50)
    print("Model Performance Comparison")
    print("="*50)
    comparison_df = pd.DataFrame({
        'Model': ['Linear Regression', 'Random Forest Regressor'],
        'MAE (Lakhs)': [mae_lr, mae_rf],
        'RMSE (Lakhs)': [rmse_lr, rmse_rf],
        'R² Score': [r2_lr, r2_rf]
    })
    print(comparison_df.to_string(index=False))

    # Feature Importances
    importances = best_rf_model.feature_importances_
    feat_imp_dict = {name: float(imp) for name, imp in zip(feature_names, importances)}

    # -------------------------------------------------------------
    # 7. Generate Visualizations
    # -------------------------------------------------------------
    print("\n[*] Generating visualization charts...")
    generate_visualizations(
        df_raw=df_raw,
        df_encoded=df_encoded,
        y_test=y_test,
        y_pred_rf=y_pred_rf,
        feature_names=feature_names,
        importances=importances,
        output_dir=images_dir
    )

    # -------------------------------------------------------------
    # 8. Serialize Trained Model Artifact
    # -------------------------------------------------------------
    model_payload = {
        'model': best_rf_model,
        'linear_regression_model': lr_model,
        'feature_names': feature_names,
        'reference_year': REFERENCE_YEAR,
        'metrics': {
            'linear_regression': {'mae': float(mae_lr), 'rmse': float(rmse_lr), 'r2': float(r2_lr)},
            'random_forest': {'mae': float(mae_rf), 'rmse': float(rmse_rf), 'r2': float(r2_rf)}
        },
        'feature_importances': feat_imp_dict,
        'best_params': rf_random.best_params_
    }

    model_filepath = os.path.join(model_dir, 'car_price_model.pkl')
    with open(model_filepath, 'wb') as f:
        pickle.dump(model_payload, f)
    print(f"\n[OK] Best model bundle successfully saved to: {model_filepath}")

    # Verify a test prediction
    sample_input = pd.DataFrame([{
        'Present_Price': 7.5,
        'Kms_Driven': 35000,
        'Owner': 0,
        'Car_Age': REFERENCE_YEAR - 2019,
        'Fuel_Type_Diesel': 0,
        'Fuel_Type_Petrol': 1,
        'Seller_Type_Individual': 0,
        'Transmission_Manual': 1
    }])[feature_names]

    sample_pred = best_rf_model.predict(sample_input)[0]
    print(f"[*] Verification sample prediction (Hyundai i20 2019 Petrol Manual @ 35k km, Present Price 7.5 Lakhs):")
    print(f"    Estimated Price: {sample_pred:.2f} Lakhs (Rs. {sample_pred * 100000:,.0f})")

if __name__ == '__main__':
    base_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    train_and_evaluate(base_directory)
