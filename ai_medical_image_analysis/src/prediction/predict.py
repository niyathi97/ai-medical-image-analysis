import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load trained model
model = tf.keras.models.load_model("../../outputs/models/pneumonia_model.h5")

# Path to image (change this later)
img_path = "../../data/chest_xray/test/NORMAL/IM-0001-0001.jpeg"

# Load and preprocess image
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)

# Output
if prediction[0][0] > 0.5:
    print("❌ Pneumonia Detected")
else:
    print("✅ Normal")