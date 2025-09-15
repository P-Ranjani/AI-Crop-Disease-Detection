from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras 
import base64
from io import BytesIO
import json

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variable to store the model
model = None
class_names = ['Healthy', 'Powdery', 'Rust']

# Disease treatment recommendations
treatments = {
    'Healthy': {
        'description': 'Your plant appears to be healthy!',
        'recommendations': [
            'Continue current care routine',
            'Ensure adequate sunlight and water',
            'Monitor for any changes in appearance',
            'Maintain proper soil nutrition'
        ],
        'prevention': [
            'Regular inspection of leaves',
            'Proper watering schedule',
            'Adequate spacing between plants',
            'Good air circulation'
        ]
    },
    'Powdery': {
        'description': 'Your plant shows signs of Powdery Mildew disease.',
        'recommendations': [
            'Apply fungicide containing sulfur or potassium bicarbonate',
            'Remove and destroy infected plant parts',
            'Improve air circulation around the plant',
            'Avoid overhead watering',
            'Apply neem oil as a natural treatment'
        ],
        'prevention': [
            'Plant resistant varieties when possible',
            'Ensure proper spacing between plants',
            'Avoid overcrowding',
            'Water at the base of plants, not on leaves',
            'Maintain good air circulation'
        ]
    },
    'Rust': {
        'description': 'Your plant shows signs of Rust disease.',
        'recommendations': [
            'Remove and destroy infected leaves immediately',
            'Apply copper-based fungicide',
            'Improve air circulation',
            'Avoid overhead watering',
            'Apply sulfur-based fungicide as preventive measure'
        ],
        'prevention': [
            'Plant resistant varieties',
            'Ensure proper spacing',
            'Water early in the day',
            'Remove plant debris regularly',
            'Avoid working with plants when they are wet'
        ]
    }
}

def load_model():
    """Load the trained model"""
    global model
    try:
        model = keras.models.load_model('plant_disease_model.h5')
        print("Model loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def preprocess_image(image):
    """Preprocess image for prediction"""
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize to model input size
    image = image.resize((224, 224))
    
    # Convert to numpy array and normalize
    img_array = np.array(image) / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def predict_disease(image):
    """Predict plant disease from image"""
    if model is None:
        return None, 0.0, None
    
    try:
        # Preprocess image
        processed_image = preprocess_image(image)
        
        # Make prediction
        predictions = model.predict(processed_image, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        predicted_class = class_names[predicted_class_idx]
        
        # Get all class probabilities
        class_probabilities = {
            class_names[i]: float(predictions[0][i]) 
            for i in range(len(class_names))
        }
        
        return predicted_class, confidence, class_probabilities
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None, 0.0, None

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        if file:
            # Read image
            image = Image.open(file.stream)
            
            # Make prediction
            predicted_class, confidence, class_probabilities = predict_disease(image)
            
            if predicted_class is None:
                return jsonify({'error': 'Prediction failed'}), 500
            
            # Get treatment information
            treatment_info = treatments.get(predicted_class, {})
            
            # Convert image to base64 for display
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            result = {
                'prediction': predicted_class,
                'confidence': confidence,
                'class_probabilities': class_probabilities,
                'treatment': treatment_info,
                'image': img_str
            }
            
            return jsonify(result)
    
    except Exception as e:
        print(f"Error in predict route: {e}")
        return jsonify({'error': 'An error occurred during prediction'}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    model_status = "loaded" if model is not None else "not loaded"
    return jsonify({
        'status': 'healthy',
        'model_status': model_status
    })

if __name__ == '__main__':
    # Load model on startup
    if load_model():
        print("Starting Flask application...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to load model. Please train the model first.")
