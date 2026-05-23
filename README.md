Hand Sign Gesture Detector

A deep learning-based Hand Sign Gesture Detection system built using Python, OpenCV, and TensorFlow CNN.
This project detects and classifies hand gestures from images and can be extended for real-time gesture recognition.

📌 Features


Hand sign gesture classification using CNN
Image preprocessing with OpenCV
Deep learning model training using TensorFlow/Keras
Dataset loading and automatic class labeling
Model saving for future real-time detection
Easy to customize with your own dataset

🛠️ Technologies Used
Python
OpenCV
NumPy
TensorFlow / Keras
Scikit-learn

📂 Project Structure

Hand-Sign-Gesture-Detector/
│
├── Data/                     # Dataset folders
├── train_model.py            # Model training code
├── detect_realtime.py        # Real-time detection script
├── sign_model.h5             # Trained CNN model
├── class_names.txt           # Gesture labels
└── README.md

⚙️ How It Works

The dataset images are loaded from folders.
Images are resized and normalized.
CNN model is created and trained.
Model learns gesture patterns from images.
Trained model is saved for real-time prediction.

🧠 CNN Architecture

Conv2D Layer
MaxPooling Layer
Conv2D Layer
MaxPooling Layer
Flatten Layer
Dense Layer
Dropout Layer
Softmax Output Layer

▶️ Run the Project

Install Dependencies
pip install tensorflow opencv-python numpy scikit-learn
Train the Model
python train_model.py
Run Real-Time Detection
python detect_realtime.py

📊 Dataset

The dataset contains multiple folders where each folder represents a unique hand gesture class.

Example:

Data/
 ├── Hello
 ├── Thanks
 ├── Yes
 └── No
 
🚀 Future Improvements

Real-time webcam gesture recognition
Better accuracy with larger datasets
GUI integration
Sign language sentence generation
Mobile application support
👨‍💻 Developed by a B.Tech Software Engineering student passionate about AI, ML, and Computer Vision.

Demo video:


https://github.com/user-attachments/assets/e4801774-8a8a-445d-892b-37e5ee07f025


