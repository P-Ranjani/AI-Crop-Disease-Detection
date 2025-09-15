#!/usr/bin/env python3
"""
Simple script to train the plant disease classification model
"""

import os
import sys
from plant_disease_model import PlantDiseaseClassifier

def main():
    print("🌱 Plant Disease Classification Model Training")
    print("=" * 50)
    
    # Check if dataset directories exist
    required_dirs = ['Train/Train', 'Test/Test', 'Validation/Validation']
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            print(f"❌ Error: Directory '{dir_path}' not found!")
            print("Please make sure your dataset is properly organized.")
            sys.exit(1)
    
    print("✅ Dataset directories found")
    
    # Initialize classifier
    print("\n🔧 Initializing classifier...")
    classifier = PlantDiseaseClassifier()
    
    # Create model
    print("🏗️  Creating model architecture...")
    model = classifier.create_model()
    
    # Build the model to count parameters
    model.build((None, 224, 224, 3))
    print(f"📊 Model created with {model.count_params():,} parameters")
    
    # Train model
    print("\n🚀 Starting training...")
    print("This may take 30-60 minutes depending on your hardware...")
    
    try:
        history = classifier.train_model('Train/Train', epochs=30)
        print("✅ Training completed successfully!")
        
        # Save model
        print("\n💾 Saving model...")
        classifier.save_model('plant_disease_model.h5')
        
        # Evaluate on test set
        print("\n📈 Evaluating on test set...")
        test_accuracy, _ = classifier.evaluate_model('Test/Test')
        
        # Evaluate on validation set
        print("\n📈 Evaluating on validation set...")
        val_accuracy, _ = classifier.evaluate_model('Validation/Validation')
        
        print("\n🎉 Training Results:")
        print(f"   Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.1f}%)")
        print(f"   Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.1f}%)")
        
        print("\n✅ Model training completed! You can now run the web app with:")
        print("   python app.py")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
