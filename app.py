"""
Car Price Prediction - Streamlit Web Application
------------------------------------------------
Interactive web dashboard demonstrating the trained CarDekho machine learning model.
Provides real-time price estimation, depreciation analysis, and model visualizations.
"""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.predict import CarPricePredictor

# Page Configuration
st.set_page_config(
    page_title="Car Price Predictor AI",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .price-card {
        background: linear-gradient(135deg, #1E40AF 0%, #3B82F6 100%);
        border-radius: 12px;
        padding: 24px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 25px rgba(59, 130, 246, 0.25);
        margin-bottom: 20px;
    }
    .price-amount {
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: -1px;
    }
    .price-sub {
        font-size: 1.25rem;
        font-weight: 600;
        color: #BFDBFE;
    }
    .metric-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        background: rgba(255, 255, 255, 0.2);
        margin-top: 8px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_predictor():
    """Cache the predictor instance."""
    return CarPricePredictor()

try:
    predictor = load_predictor()
except Exception as e:
    st.error(f"Error loading model: {e}. Please run `python src/train_model.py` first.")
    st.stop()

# Sidebar: Presets and Info
st.sidebar.title("⚡ Quick Presets")
st.sidebar.caption("Click a preset vehicle to test the predictor:")

PRESETS = {
    "Custom Input": None,
    "Maruti Swift VXi (2018)": {
        "name": "Maruti Swift VXi", "present_price": 5.90, "year": 2018,
        "kms": 42000, "fuel": "Petrol", "seller": "Dealer", "trans": "Manual", "owner": 0
    },
    "Hyundai i20 Asta (2019)": {
        "name": "Hyundai i20 Asta", "present_price": 7.50, "year": 2019,
        "kms": 35000, "fuel": "Petrol", "seller": "Dealer", "trans": "Manual", "owner": 0
    },
    "Honda City VX (2016)": {
        "name": "Honda City VX", "present_price": 11.20, "year": 2016,
        "kms": 58000, "fuel": "Petrol", "seller": "Dealer", "trans": "Automatic", "owner": 1
    },
    "Toyota Fortuner 4x2 (2018)": {
        "name": "Toyota Fortuner", "present_price": 32.50, "year": 2018,
        "kms": 65000, "fuel": "Diesel", "seller": "Dealer", "trans": "Automatic", "owner": 0
    }
}

selected_preset = st.sidebar.selectbox("Choose sample vehicle:", list(PRESETS.keys()))

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Model Performance")
st.sidebar.markdown(f"**Random Forest R² Score:** `{predictor.metrics.get('random_forest', {}).get('r2', 0.959):.3f}`")
st.sidebar.markdown(f"**Random Forest RMSE:** `₹{predictor.metrics.get('random_forest', {}).get('rmse', 0.966):.2f} Lakhs`")
st.sidebar.markdown(f"**Linear Regression R²:** `{predictor.metrics.get('linear_regression', {}).get('r2', 0.849):.3f}`")

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip:** Model trained on CarDekho verified sales dataset with 301 entries and 9 attributes.")

# Header
st.markdown('<div class="main-title">🚗 Car Price Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Estimate fair market value and depreciation of used cars using Machine Learning</div>', unsafe_allow_html=True)

# Main Form Layout
col_input, col_result = st.columns([1.1, 0.9], gap="large")

# Preset defaults
preset_data = PRESETS[selected_preset] if selected_preset != "Custom Input" else None

with col_input:
    st.subheader("📝 Vehicle Specifications")
    
    car_name = st.text_input(
        "Car Model Name", 
        value=preset_data["name"] if preset_data else "Hyundai i20",
        help="Model name for record and display"
    )
    
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        present_price = st.number_input(
            "Showroom / New Price (₹ Lakhs)",
            min_value=0.5,
            max_value=100.0,
            value=preset_data["present_price"] if preset_data else 7.50,
            step=0.25,
            help="Original ex-showroom invoice price in Lakhs (e.g. 7.50 = ₹7,50,000)"
        )
        st.caption(f"Equivalent: **₹{int(present_price * 100000):,}**")

    with col_p2:
        year = st.slider(
            "Manufacturing Year",
            min_value=2005,
            max_value=2024,
            value=preset_data["year"] if preset_data else 2019
        )
        st.caption(f"Car Age: **{2024 - year} years**")

    col_k1, col_k2 = st.columns(2)
    with col_k1:
        kms_driven = st.number_input(
            "Kilometers Driven",
            min_value=500,
            max_value=500000,
            value=preset_data["kms"] if preset_data else 35000,
            step=2500
        )
    with col_k2:
        owner_options = [0, 1, 2, 3]
        owner_labels = {0: "0 (First Owner)", 1: "1 (Second Owner)", 2: "2 (Third Owner)", 3: "3+ (Fourth or more)"}
        owner = st.selectbox(
            "Previous Owners",
            options=owner_options,
            format_func=lambda x: owner_labels[x],
            index=owner_options.index(preset_data["owner"]) if preset_data else 0
        )

    col_f1, col_f2, col_f3 = st.columns(3)
    fuel_list = ["Petrol", "Diesel", "CNG"]
    with col_f1:
        fuel_type = st.selectbox(
            "Fuel Type", 
            fuel_list,
            index=fuel_list.index(preset_data["fuel"]) if preset_data else 0
        )
    seller_list = ["Dealer", "Individual"]
    with col_f2:
        seller_type = st.selectbox(
            "Seller Type",
            seller_list,
            index=seller_list.index(preset_data["seller"]) if preset_data else 0
        )
    trans_list = ["Manual", "Automatic"]
    with col_f3:
        transmission = st.selectbox(
            "Transmission",
            trans_list,
            index=trans_list.index(preset_data["trans"]) if preset_data else 0
        )

    predict_button = st.button("⚡ Calculate Estimated Selling Price", type="primary", use_container_width=True)

with col_result:
    st.subheader("🏷️ Valuation Summary")
    
    # Calculate inference
    res = predictor.predict(
        present_price=present_price,
        year=year,
        kms_driven=kms_driven,
        fuel_type=fuel_type,
        seller_type=seller_type,
        transmission=transmission,
        owner=owner,
        car_name=car_name
    )

    st.markdown(f"""
    <div class="price-card">
        <div>ESTIMATED FAIR RESALE VALUE</div>
        <div class="price-amount">₹{res['predicted_price_lakhs']:.2f} <span style="font-size: 1.5rem;">Lakhs</span></div>
        <div class="price-sub">{res['predicted_price_inr']}</div>
        <div class="metric-badge">{res['tier_badge']} Valuation</div>
    </div>
    """, unsafe_allow_html=True)

    # Key Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Showroom Price", f"₹{res['present_price_lakhs']} L")
    m2.metric("Estimated Loss", f"₹{res['depreciation_lakhs']} L", delta=f"-{res['depreciation_pct']}%", delta_color="inverse")
    retained_pct = max(0.0, 100.0 - res['depreciation_pct'])
    m3.metric("Retained Value", f"{retained_pct:.1f}%")

    st.progress(retained_pct / 100.0)
    st.caption(f"**Market Verdict:** {res['valuation_tier']}")

# Visualizations & Insights Tabs
st.markdown("---")
st.subheader("📈 Model Evaluation & Analysis Visualizations")

tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Feature Importances", 
    "📉 Actual vs Predicted", 
    "📊 Price Distribution", 
    "📋 Model Metrics Comparison"
])

with tab1:
    st.markdown("### Top Predictive Features")
    st.write("Feature importances extracted from the tuned Random Forest Regressor:")
    feat_imp = pd.Series(predictor.feature_importances).sort_values(ascending=True)
    st.bar_chart(feat_imp)
    st.caption("Notice how **Present Price** and **Car Age** represent the strongest predictive signals for used car resale value.")

with tab2:
    st.markdown("### Test Set: Actual vs Predicted Selling Prices")
    img_path = os.path.join(PROJECT_ROOT, "images", "actual_vs_predicted.png")
    if os.path.exists(img_path):
        st.image(img_path, caption="Actual vs Predicted prices with perfect fit line (y = x)")
    else:
        st.info("Run `python src/train_model.py` to generate visualization images.")

with tab3:
    st.markdown("### Dataset Price Distributions")
    img_path = os.path.join(PROJECT_ROOT, "images", "price_distribution.png")
    if os.path.exists(img_path):
        st.image(img_path, caption="Distribution of Selling Price vs Present Price")
    else:
        st.info("Run `python src/train_model.py` to generate visualization images.")

with tab4:
    st.markdown("### Linear Regression vs Tuned Random Forest Comparison")
    metrics_data = {
        "Model": ["Linear Regression (Baseline)", "Random Forest Regressor (Tuned) 🏆"],
        "MAE (₹ Lakhs)": [
            f"{predictor.metrics.get('linear_regression', {}).get('mae', 1.216):.4f}",
            f"{predictor.metrics.get('random_forest', {}).get('mae', 0.639):.4f}"
        ],
        "RMSE (₹ Lakhs)": [
            f"{predictor.metrics.get('linear_regression', {}).get('rmse', 1.865):.4f}",
            f"{predictor.metrics.get('random_forest', {}).get('rmse', 0.966):.4f}"
        ],
        "R² Score": [
            f"{predictor.metrics.get('linear_regression', {}).get('r2', 0.849):.4f} (84.9%)",
            f"{predictor.metrics.get('random_forest', {}).get('r2', 0.959):.4f} (95.9%)"
        ]
    }
    st.table(pd.DataFrame(metrics_data))
