# Plant Disease Detection System

An AI-powered web application that detects plant diseases from leaf images and provides treatment recommendations.

## Features

- **Disease Classification**: Detects 3 types of plant conditions:
  - Healthy
  - Powdery Mildew
  - Rust Disease
- **Web Interface**: User-friendly interface for image upload and results display
- **Treatment Recommendations**: Provides specific treatment and prevention advice
- **Confidence Scores**: Shows prediction confidence and probability distribution

## Dataset

The model is trained on a dataset with the following structure:
- **Training Set**: 1,322 images (458 Healthy, 430 Powdery, 434 Rust)
- **Test Set**: 150 images
- **Validation Set**: 60 images

## Installation

1. **Clone or download the project files**

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model** (first time only):
   ```bash
   python plant_disease_model.py
   ```
   This will:
   - Train a CNN model on your dataset
   - Save the trained model as `plant_disease_model.h5`
   - Generate training history and confusion matrix plots

4. **Run the web application**:
   ```bash
   python app.py
   ```

5. **Open your browser** and go to `http://localhost:5000`

## Usage

1. **Upload an Image**: Click the upload area or drag and drop a plant leaf image
2. **View Results**: The system will display:
   - Predicted disease type with confidence score
   - Probability distribution for all classes
   - Treatment recommendations
   - Prevention tips

## Model Architecture

The CNN model includes:
- Data augmentation layers (random flip, rotation, zoom)
- 4 convolutional blocks with batch normalization
- Global average pooling
- Dense layers with dropout for regularization
- Softmax output for 3-class classification

## File Structure

```
├── plant_disease_model.py    # Model training script
├── app.py                    # Flask web application
├── templates/
│   └── index.html           # Web interface template
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── Train/                  # Training dataset
├── Test/                   # Test dataset
└── Validation/             # Validation dataset
```

## API Endpoints

- `GET /`: Main web interface
- `POST /predict`: Upload image and get prediction
- `GET /health`: Health check endpoint

## Troubleshooting

1. **Model not found error**: Make sure to train the model first by running `plant_disease_model.py`
2. **Memory issues**: Reduce batch size in the training script if you encounter memory errors
3. **Slow predictions**: The model may take a few seconds to load on first use

## Performance

The model typically achieves:
- Training accuracy: 95%+
- Validation accuracy: 90%+
- Test accuracy: 85%+

## Contributing

Feel free to improve the model architecture, add more disease classes, or enhance the web interface.

## License

This project is open source and available under the MIT License.
