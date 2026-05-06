import cv2
import mediapipe as mp
import numpy as np
import os
from tensorflow.keras.utils import to_categorical

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

class HandDataProcessor:
    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.7
        )
    
    def extract_hand_landmarks(self, image):
        """Extract 21 hand landmarks (42 values: x,y coordinates)"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_image)
        
        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0].landmark
            # Normalize coordinates to 0-1 range
            h, w = image.shape[:2]
            coords = np.array([[lm.x, lm.y] for lm in landmarks]).flatten()
            return coords
        return np.zeros(42)  # 21 landmarks * 2 coords
    
    def preprocess_dataset(self, dataset_path, img_size=(224, 224)):
        """Load images and extract landmarks"""
        X, y = [], []
        labels = sorted(os.listdir(dataset_path))
        label_to_idx = {label: i for i, label in enumerate(labels)}
        
        for label in labels:
            label_path = os.path.join(dataset_path, label)
            for img_name in os.listdir(label_path):
                img_path = os.path.join(label_path, img_name)
                image = cv2.imread(img_path)
                image = cv2.resize(image, img_size)
                
                landmarks = self.extract_hand_landmarks(image)
                X.append(landmarks)
                y.append(label_to_idx[label])
        
        return np.array(X), np.array(y), labels

# Usage
processor = HandDataProcessor()
X, y, class_names = processor.preprocess_dataset('"C:\VED\CTAI\Data"')
X = X.reshape(-1, 21, 2, 1)  # For CNN input
y = to_categorical(y, num_classes=len(class_names))