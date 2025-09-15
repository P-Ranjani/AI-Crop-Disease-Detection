#!/usr/bin/env python3
"""
Demo script to test the plant disease detection system
"""

import os
import glob
import random
from plant_disease_model import PlantDiseaseClassifier

def main():
    print("🌱 Plant Disease Detection Demo")
    print("=" * 40)
    
    # Check if model exists
    if not os.path.exists('plant_disease_model.h5'):
        print("❌ Model not found! Please train the model first:")
        print("   python train_model.py")
        return
    
    # Initialize classifier
    print("📂 Loading model...")
    classifier = PlantDiseaseClassifier()
    classifier.load_model('plant_disease_model.h5')
    
    # Find some test images
    test_images = []
    for class_name in ['Healthy', 'Powdery', 'Rust']:
        pattern = f"Test/Test/{class_name}/*.jpg"
        images = glob.glob(pattern)
        if images:
            test_images.extend(images[:2])  # Take 2 images from each class
    
    if not test_images:
        print("❌ No test images found!")
        return
    
    print(f"🔍 Testing with {len(test_images)} random images...")
    print("-" * 50)
    
    # Test random images
    random.shuffle(test_images)
    
    for i, image_path in enumerate(test_images[:6]):  # Test 6 images
        try:
            # Get true class from directory name
            true_class = os.path.basename(os.path.dirname(image_path))
            
            # Make prediction
            predicted_class, confidence, probabilities = classifier.predict_single_image(image_path)
            
            # Display results
            print(f"\n📸 Image {i+1}: {os.path.basename(image_path)}")
            print(f"   True Class: {true_class}")
            print(f"   Predicted: {predicted_class}")
            print(f"   Confidence: {confidence:.3f}")
            
            # Show all probabilities
            print("   All Probabilities:")
            for class_name, prob in zip(classifier.class_names, probabilities):
                print(f"     {class_name}: {prob:.3f}")
            
            # Show if correct
            is_correct = predicted_class == true_class
            status = "✅ CORRECT" if is_correct else "❌ INCORRECT"
            print(f"   Result: {status}")
            
        except Exception as e:
            print(f"❌ Error processing {image_path}: {e}")
    
    print("\n🎉 Demo completed!")
    print("\nTo use the web interface, run:")
    print("   python app.py")
    print("Then open: http://localhost:5000")

if __name__ == "__main__":
    main()
