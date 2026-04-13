import sys
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

# Fix path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from preprocessing.preprocess import test_data

# Load trained model
model = tf.keras.models.load_model("../../outputs/models/pneumonia_model.h5")

# Get predictions
predictions = model.predict(test_data)
predicted_classes = (predictions > 0.5).astype(int)

# True labels
true_classes = test_data.classes

# Print results
print("Confusion Matrix:")
print(confusion_matrix(true_classes, predicted_classes))

print("\nClassification Report:")
print(classification_report(true_classes, predicted_classes))