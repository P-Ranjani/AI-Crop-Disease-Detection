#!/usr/bin/env python3
"""
Batch prediction script for testing the trained model
"""

import os
import sys
import glob
from plant_disease_model import PlantDiseaseClassifier

def main():
    print("🔍 Plant Disease Batch Prediction")
    print("=" * 40)
    
    # Check if model exists
    if not os.path.exists('plant_disease_model.h5'):
        print("❌ Error: Model file 'plant_disease_model.h5' not found!")
        print("Please train the model first by running: python train_model.py")
        sys.exit(1)
    
    # Initialize classifier and load model
    print("📂 Loading trained model...")
    classifier = PlantDiseaseClassifier()
    classifier.load_model('plant_disease_model.h5')
    
    # Get test images
    test_dirs = ['Test/Test/Healthy', 'Test/Test/Powdery', 'Test/Test/Rust']
    
    print("\n🔍 Running predictions on test images...")
    print("-" * 50)
    
    total_correct = 0
    total_images = 0
    
    for test_dir in test_dirs:
        if not os.path.exists(test_dir):
            continue
            
        true_class = os.path.basename(test_dir)
        image_files = glob.glob(os.path.join(test_dir, '*.jpg'))
        
        print(f"\n📁 Testing {true_class} images ({len(image_files)} files):")
        
        correct = 0
        for i, image_path in enumerate(image_files[:10]):  # Test first 10 images
            try:
                predicted_class, confidence, _ = classifier.predict_single_image(image_path)
                
                is_correct = predicted_class == true_class
                if is_correct:
                    correct += 1
                    total_correct += 1
                
                status = "✅" if is_correct else "❌"
                print(f"  {status} {os.path.basename(image_path)}: {predicted_class} ({confidence:.3f})")
                
            except Exception as e:
                print(f"  ❌ Error processing {os.path.basename(image_path)}: {e}")
            
            total_images += 1
        
        accuracy = correct / min(len(image_files), 10) * 100
        print(f"  📊 Accuracy for {true_class}: {accuracy:.1f}% ({correct}/{min(len(image_files), 10)})")
    
    overall_accuracy = total_correct / total_images * 100 if total_images > 0 else 0
    print(f"\n🎯 Overall Test Accuracy: {overall_accuracy:.1f}% ({total_correct}/{total_images})")
    
    print("\n✅ Batch prediction completed!")

if __name__ == "__main__":
    main()
