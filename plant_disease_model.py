import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

class PlantDiseaseClassifier:
    def __init__(self, img_size=(224, 224), num_classes=3):
        self.img_size = img_size
        self.num_classes = num_classes
        self.class_names = ['Healthy', 'Powdery', 'Rust'] 
        self.model = None
        
    def create_model(self):
        """Create a CNN model for plant disease classification"""
        model = keras.Sequential([
            # Data augmentation
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            
            # Convolutional layers
            layers.Conv2D(32, 3, activation='relu', input_shape=(*self.img_size, 3)),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            
            layers.Conv2D(64, 3, activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            
            layers.Conv2D(128, 3, activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            
            layers.Conv2D(256, 3, activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            
            # Dense layers
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def load_data(self, data_dir, subset='training'):
        """Load and preprocess image data"""
        if subset in ['training', 'validation']:
            # Use validation split for training data
            datagen = keras.preprocessing.image.ImageDataGenerator(
                rescale=1./255,
                validation_split=0.2
            )
            
            generator = datagen.flow_from_directory(
                data_dir,
                target_size=self.img_size,
                batch_size=32,
                class_mode='categorical',
                subset=subset
            )
        else:
            # For test/validation directories, no split needed
            datagen = keras.preprocessing.image.ImageDataGenerator(
                rescale=1./255
            )
            
            generator = datagen.flow_from_directory(
                data_dir,
                target_size=self.img_size,
                batch_size=32,
                class_mode='categorical'
            )
        
        return generator
    
    def train_model(self, train_dir, epochs=50, validation_split=0.2):
        """Train the model"""
        print("Loading training data...")
        train_generator = self.load_data(train_dir, 'training')
        val_generator = self.load_data(train_dir, 'validation')
        
        print(f"Training samples: {train_generator.samples}")
        print(f"Validation samples: {val_generator.samples}")
        
        # Create callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        ]
        
        print("Starting training...")
        history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def evaluate_model(self, test_dir):
        """Evaluate the model on test data"""
        print("Loading test data...")
        test_generator = self.load_data(test_dir)
        
        print("Evaluating model...")
        test_loss, test_accuracy = self.model.evaluate(test_generator, verbose=1)
        
        # Get predictions
        predictions = self.model.predict(test_generator)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = test_generator.classes
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(true_classes, predicted_classes, 
                                  target_names=self.class_names))
        
        # Confusion matrix
        cm = confusion_matrix(true_classes, predicted_classes)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, 
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return test_accuracy, predictions
    
    def predict_single_image(self, image_path):
        """Predict disease for a single image"""
        # Load and preprocess image
        img = Image.open(image_path)
        img = img.convert('RGB')
        img = img.resize(self.img_size)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        prediction = self.model.predict(img_array)
        predicted_class = np.argmax(prediction[0])
        confidence = np.max(prediction[0])
        
        return self.class_names[predicted_class], confidence, prediction[0]
    
    def save_model(self, filepath):
        """Save the trained model"""
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a pre-trained model"""
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")

def main():
    # Initialize classifier
    classifier = PlantDiseaseClassifier()
    
    # Create model
    print("Creating model...")
    model = classifier.create_model()
    print(model.summary())
    
    # Train model
    print("\nTraining model...")
    history = classifier.train_model('Train/Train', epochs=30)
    
    # Save model
    classifier.save_model('plant_disease_model.h5')
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_accuracy, predictions = classifier.evaluate_model('Test/Test')
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    val_accuracy, val_predictions = classifier.evaluate_model('Validation/Validation')
    
    print(f"\nFinal Results:")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()
