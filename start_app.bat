@echo off
echo 🌱 Plant Disease Detection System
echo ================================

echo.
echo Checking if model exists...
if not exist "plant_disease_model.h5" (
    echo ❌ Model not found! Training model first...
    echo.
    python train_model.py
    if errorlevel 1 (
        echo ❌ Training failed! Please check the error messages above.
        pause
        exit /b 1
    )
    echo.
    echo ✅ Model training completed!
) else (
    echo ✅ Model found!
)

echo.
echo 🚀 Starting web application...
echo Open your browser and go to: http://localhost:5000
echo Press Ctrl+C to stop the server
echo.

python app.py

pause
