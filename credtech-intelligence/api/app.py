# Entry point for Flask API
from flask import Flask, jsonify, request
from flask_cors import CORS
import logging
import os
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Configuration
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'dev-secret-key-change-in-production')
app.config['DEBUG'] = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'

@app.route('/')
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'service': 'Credit Intelligence API',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '1.0.0'
    })

@app.route('/api/status')
def api_status():
    """API status endpoint."""
    return jsonify({
        'api_status': 'online',
        'services': {
            'news_collector': 'available',
            'edgar_collector': 'available', 
            'credit_model': 'available',
            'real_time_monitor': 'available'
        },
        'timestamp': datetime.utcnow().isoformat()
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Credit scoring prediction endpoint."""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Placeholder for actual prediction logic
        prediction = {
            'credit_score': 650,
            'risk_level': 'medium',
            'confidence': 0.85,
            'factors': {
                'income_ratio': 0.7,
                'payment_history': 0.9,
                'credit_utilization': 0.6
            }
        }
        
        return jsonify({
            'prediction': prediction,
            'timestamp': datetime.utcnow().isoformat(),
            'model_version': '1.0.0'
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(
        host='0.0.0.0',
        port=port,
        debug=app.config['DEBUG']
    )
