import cv2
import numpy as np
import os
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

def load_dataset(dataset_path, img_size=(64,64), max_per_class=200):
    folders = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path,d))])
    print(f"✅ Found {len(folders)} gestures: {folders[:10]}")
    
    X, y, class_names = [], [], folders
    
    for class_idx, folder in enumerate(folders):
        folder_path = os.path.join(dataset_path, folder)
        imgs = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg','.png'))]
        
        for img_file in imgs[:max_per_class]:
            img_path = os.path.join(folder_path, img_file)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, img_size)
                img = img.astype('float32') / 255.0
                X.append(img)
                y.append(class_idx)
    
    X, y = np.array(X), to_categorical(y, len(folders))
    print(f"📊 Dataset: {X.shape[0]} images, {X.shape[1:]} shape")
    return X, y, class_names

# YOUR PATH
dataset_path = r"C:\VED\CTAI\Data"

X, y, class_names = load_dataset(dataset_path)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# CNN Model
model = models.Sequential([
    layers.Conv2D(32, 3, activation='relu', input_shape=(64,64,3)),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("🚀 Training (3 mins)...")
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Save model + class names
model.save('sign_model.h5')

with open('class_names.txt', 'w') as f:
    for name in class_names:
        f.write(f"{name}\n")

print("✅ SAVED: sign_model.h5 + class_names.txt")
print("🎥 Run: python detect_realtime.py")