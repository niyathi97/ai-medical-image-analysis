import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Dataset paths
train_dir = "../../data/chest_xray/train"
test_dir = "../../data/chest_xray/test"
val_dir = "../../data/chest_xray/val"

# Image size
IMG_SIZE = 224
BATCH_SIZE = 32

# Data generators (rescaling + augmentation)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Load datasets
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

val_data = test_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

print("✅ Data loaded successfully!")