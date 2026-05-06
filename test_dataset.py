import cv2
import mediapipe as mp
import numpy as np
import os

# HandDataProcessor class (needed for testing)
class HandDataProcessor:
    def __init__(self):
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.7
        )
    
    def extract_hand_landmarks(self, image):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_image)
        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0].landmark
            h, w = image.shape[:2]
            coords = np.array([[lm.x, lm.y] for lm in landmarks]).flatten()
            return coords
        return np.zeros(42)

# TEST YOUR DATASET
processor = HandDataProcessor()
dataset_path = r"C:\VED\CTAI\Data"  # ✅ YOUR PATH (Windows raw string)

print("🔍 Testing Dataset...")
print(f"Dataset path: {dataset_path}")

if not os.path.exists(dataset_path):
    print("❌ ERROR: Dataset folder not found!")
    print("Check path: C:\\VED\\CTAI\\Data")
    exit()

labels = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
print(f"✅ Found {len(labels)} gesture folders:")
print(labels[:10])  # First 10

total_images = 0
for label in labels[:10]:  # Test first 10 folders
    folder = os.path.join(dataset_path, label)
    images = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    count = len(images)
    total_images += count
    print(f"   📁 {label}: {count} images")

print(f"\n🎉 TOTAL: {total_images} images ready for training!")