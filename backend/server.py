"""
Flask REST API Backend for Car Price Prediction
-----------------------------------------------
Provides prediction, evaluation metrics, feature importances,
and serves the web frontend and visualization assets.
"""

import os
import sys
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Add root project path to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.predict import CarPricePredictor

app = Flask(
    __name__,
    static_folder=os.path.join(PROJECT_ROOT, 'frontend'),
    static_url_path=''
)
CORS(app)

# Initialize predictor
try:
    predictor = CarPricePredictor()
    print("[*] Predictor initialized successfully.")
except Exception as e:
    print(f"[!] Warning: Could not initialize predictor directly: {e}")
    predictor = None

# Sample car presets for quick user evaluation
PRESETS = [
    {
        "id": "swift",
        "car_name": "Maruti Swift VXi",
        "present_price": 5.90,
        "year": 2018,
        "kms_driven": 42000,
        "fuel_type": "Petrol",
        "seller_type": "Dealer",
        "transmission": "Manual",
        "owner": 0
    },
    {
        "id": "i20",
        "car_name": "Hyundai i20 Asta",
        "present_price": 7.50,
        "year": 2019,
        "kms_driven": 35000,
        "fuel_type": "Petrol",
        "seller_type": "Dealer",
        "transmission": "Manual",
        "owner": 0
    },
    {
        "id": "city",
        "car_name": "Honda City VX",
        "present_price": 11.20,
        "year": 2016,
        "kms_driven": 58000,
        "fuel_type": "Petrol",
        "seller_type": "Dealer",
        "transmission": "Automatic",
        "owner": 1
    },
    {
        "id": "fortuner",
        "car_name": "Toyota Fortuner 4x2 AT",
        "present_price": 32.50,
        "year": 2018,
        "kms_driven": 65000,
        "fuel_type": "Diesel",
        "seller_type": "Dealer",
        "transmission": "Automatic",
        "owner": 0
    },
    {
        "id": "alto",
        "car_name": "Maruti Alto 800 LXi",
        "present_price": 3.80,
        "year": 2015,
        "kms_driven": 62000,
        "fuel_type": "Petrol",
        "seller_type": "Individual",
        "transmission": "Manual",
        "owner": 1
    }
]

@app.route('/')
def index():
    """Serve frontend web app."""
    return send_from_directory(os.path.join(PROJECT_ROOT, 'frontend'), 'index.html')

@app.route('/images/<path:filename>')
def serve_image(filename):
    """Serve plot images from images directory."""
    return send_from_directory(os.path.join(PROJECT_ROOT, 'images'), filename)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Service health and model metadata check."""
    return jsonify({
        'status': 'healthy',
        'service': 'Car Price Prediction API',
        'model_loaded': predictor is not None,
        'version': '1.0.0'
    })

@app.route('/api/presets', methods=['GET'])
def get_presets():
    """Return preconfigured car samples for one-click testing."""
    return jsonify({
        'status': 'success',
        'presets': PRESETS
    })

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """Return model evaluation performance metrics."""
    if not predictor:
        return jsonify({'status': 'error', 'message': 'Model not loaded'}), 500
    
    return jsonify({
        'status': 'success',
        'metrics': predictor.metrics,
        'feature_importances': predictor.feature_importances
    })

@app.route('/api/predict', methods=['POST'])
def predict_price():
    """
    Handle car valuation prediction requests.
    Expected JSON:
    {
        "car_name": "Hyundai i20",
        "present_price": 7.5,
        "year": 2019,
        "kms_driven": 35000,
        "fuel_type": "Petrol",
        "seller_type": "Dealer",
        "transmission": "Manual",
        "owner": 0
    }
    """
    if not predictor:
        return jsonify({'status': 'error', 'message': 'Model not loaded. Train the model first.'}), 500

    data = request.get_json(silent=True)
    if not data:
        return jsonify({'status': 'error', 'message': 'Invalid JSON request payload'}), 400

    try:
        car_name = data.get('car_name', 'Used Car')
        present_price = float(data.get('present_price', 0))
        year = int(data.get('year', 2018))
        kms_driven = int(data.get('kms_driven', 0))
        fuel_type = str(data.get('fuel_type', 'Petrol'))
        seller_type = str(data.get('seller_type', 'Dealer'))
        transmission = str(data.get('transmission', 'Manual'))
        owner = int(data.get('owner', 0))

        if present_price <= 0:
            return jsonify({'status': 'error', 'message': 'Present showroom price must be greater than 0.'}), 400
        
        if year < 1995 or year > predictor.reference_year:
            return jsonify({'status': 'error', 'message': f'Manufacturing year must be between 1995 and {predictor.reference_year}.'}), 400
        
        if kms_driven < 0:
            return jsonify({'status': 'error', 'message': 'Kilometers driven cannot be negative.'}), 400

        result = predictor.predict(
            present_price=present_price,
            year=year,
            kms_driven=kms_driven,
            fuel_type=fuel_type,
            seller_type=seller_type,
            transmission=transmission,
            owner=owner,
            car_name=car_name
        )

        return jsonify(result)

    except Exception as err:
        return jsonify({'status': 'error', 'message': str(err)}), 500

def run_server(host='127.0.0.1', port=5000, debug=False):
    """Run Flask development server."""
    print(f"[*] Starting Car Price Prediction server on http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)

if __name__ == '__main__':
    run_server()
