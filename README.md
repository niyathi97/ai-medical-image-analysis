 🩺 AI-Powered Medical Image Analysis System

An end-to-end deep learning project that detects **Pneumonia from Chest X-ray images** using Transfer Learning (MobileNetV2) and provides real-time predictions through a Streamlit web application.


 📌 Overview

This project demonstrates how Artificial Intelligence can assist in healthcare by analyzing medical images. The system is trained on a real-world chest X-ray dataset and can classify images as **Normal** or **Pneumonia**.


 🚀 Features

- 🔍 Pneumonia detection using Chest X-rays  
- 🧠 Transfer Learning with MobileNetV2  
- 🧹 Image preprocessing & augmentation  
- 📊 Model evaluation (Accuracy, Confusion Matrix)  
- 🌐 Interactive Streamlit web app  
- ⚡ Real-time predictions with confidence score  

 🧠 Tech Stack

- **Programming Language:** Python  
- **Deep Learning:** TensorFlow / Keras  
- **Computer Vision:** OpenCV  
- **Data Processing:** NumPy  
- **Visualization:** Matplotlib  
- **Frontend/UI:** Streamlit  

 📂 Project Structure

ai_medical_image_analysis/
│
├── data/                  # Dataset (not uploaded to GitHub)
├── src/
│   ├── preprocessing/     # Data loading & preprocessing
│   ├── models/            # Model architecture
│   ├── training/          # Training script
│   ├── evaluation/        # Evaluation metrics
│   ├── prediction/        # Prediction script
│
├── outputs/
│   ├── models/            # Saved trained model
│   ├── plots/             # Graphs and results
│
├── app.py                 # Streamlit web app
├── requirements.txt       # Dependencies
├── README.md              # Project documentation


 🧪 How It Works

1. Load chest X-ray images  
2. Preprocess (resize, normalize, augment)  
3. Train model using MobileNetV2  
4. Evaluate performance  
5. Predict new images via UI  

 📊 Model Details

- **Model:** MobileNetV2 (Transfer Learning)  
- **Input Size:** 224 × 224  
- **Output:** Binary Classification (Normal / Pneumonia)  


🎯 Results

- Achieved reliable classification performance on test data  
- Successfully predicts pneumonia from unseen X-ray images  



 🔮 Future Improvements

- 🔥 Grad-CAM (highlight infected regions)  
- ☁️ Cloud deployment (Streamlit Cloud / AWS)  
- 🧠 Multi-disease detection  
- 📱 Mobile integration  


 👩‍💻 Author

Niyathi

If you like this project, give it a ⭐ on GitHub!
