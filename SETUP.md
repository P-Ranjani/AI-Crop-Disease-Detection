# Setup Guide - Plant Disease Detection System

## Quick Start (Windows)

1. **Install Python** (if not already installed):
   - Download Python 3.8+ from [python.org](https://python.org)
   - Make sure to check "Add Python to PATH" during installation

2. **Install Dependencies**:
   ```cmd
   pip install -r requirements.txt
   ```

3. **Train the Model** (first time only):
   ```cmd
   python train_model.py
   ```
   This will take 30-60 minutes depending on your computer.

4. **Start the Web App**:
   ```cmd
   python app.py
   ```
   Or double-click `start_app.bat` for automatic setup.

5. **Open Browser**: Go to `http://localhost:5000`

## Manual Setup Steps

### Step 1: Install Dependencies
```bash
pip install tensorflow==2.13.0
pip install flask==2.3.3
pip install pillow==10.0.1
pip install numpy==1.24.3
pip install scikit-learn==1.3.0
pip install matplotlib==3.7.2
pip install seaborn==0.12.2
```

### Step 2: Train the Model
```bash
python train_model.py
```

### Step 3: Test the Model (Optional)
```bash
python demo.py
python predict_batch.py
```

### Step 4: Run the Web Application
```bash
python app.py
```

## File Structure After Setup

```
Crop/
├── plant_disease_model.h5      # Trained model (created after training)
├── plant_disease_model.py      # Model training code
├── app.py                      # Web application
├── train_model.py              # Simple training script
├── demo.py                     # Demo script
├── predict_batch.py            # Batch prediction script
├── start_app.bat               # Windows startup script
├── requirements.txt            # Dependencies
├── README.md                   # Documentation
├── SETUP.md                    # This file
├── templates/
│   └── index.html             # Web interface
├── Train/Train/               # Training data
├── Test/Test/                 # Test data
└── Validation/Validation/     # Validation data
```

## Troubleshooting

### Common Issues:

1. **"Module not found" errors**:
   - Make sure you installed all dependencies: `pip install -r requirements.txt`

2. **"Model not found" error**:
   - Train the model first: `python train_model.py`

3. **Memory errors during training**:
   - Close other applications
   - Reduce batch size in `plant_disease_model.py` (line with `batch_size=32`)

4. **Slow training**:
   - This is normal for the first time. Training takes 30-60 minutes.
   - The model will be saved and reused for future runs.

5. **Web app won't start**:
   - Check if port 5000 is available
   - Try a different port by modifying `app.py` (change `port=5000` to `port=5001`)

### System Requirements:

- **Python**: 3.8 or higher
- **RAM**: 8GB recommended (4GB minimum)
- **Storage**: 2GB free space
- **OS**: Windows 10/11, macOS, or Linux

## Usage Examples

### Using the Web Interface:
1. Open `http://localhost:5000`
2. Upload a plant leaf image
3. View disease prediction and treatment recommendations

### Using the Command Line:
```bash
# Train model
python train_model.py

# Test with demo images
python demo.py

# Batch prediction on test set
python predict_batch.py
```

## Next Steps

After successful setup:
1. Try uploading different plant leaf images
2. Check the accuracy on your test images
3. Customize the treatment recommendations in `app.py`
4. Add more disease classes if you have additional data

## Support

If you encounter issues:
1. Check the error messages carefully
2. Ensure all dependencies are installed
3. Verify your dataset structure matches the expected format
4. Check that you have sufficient disk space and memory
