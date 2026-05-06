import cv2
import numpy as np
from tensorflow.keras.models import load_model

class SignDetector:
    def __init__(self):
        self.model = load_model('sign_model.h5')
        
        # Load class names
        with open('class_names.txt', 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]
        
        print(f"✅ Loaded {len(self.class_names)} gestures")
    
    def predict(self, frame):
        img = cv2.resize(frame, (64, 64))
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, 0)
        pred = self.model.predict(img, verbose=0)
        gesture_idx = np.argmax(pred)
        confidence = float(np.max(pred))
        return self.class_names[gesture_idx], confidence
    
    def run(self):
        cap = cv2.VideoCapture(0)
        print("🎥 Show HAND to screen! 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            frame = cv2.flip(frame, 1)
            gesture, conf = self.predict(frame)
            
            # Draw box
            cv2.rectangle(frame, (10,10), (400,80), (0,255,0), -1)
            cv2.putText(frame, f'{gesture}', (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.putText(frame, f'{conf:.2f}', (20,65), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            
            cv2.imshow('Sign Language', frame)
            if cv2.waitKey(1) == ord('q'): break
        
        cap.release()
        cv2.destroyAllWindows()

SignDetector().run()