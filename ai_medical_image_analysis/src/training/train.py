import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from preprocessing.preprocess import train_data, val_data
from models.model import build_model

# Build model
model = build_model()

# Train model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=5
)

# Save model
model.save("../../outputs/models/pneumonia_model.h5")

print("✅ Model trained and saved successfully!")