# object_detector.py
import cv2
import numpy as np
import os
from keras.models import load_model

class WebcamObjectDetector:
    def __init__(self, model_path, labels_path, camera_index=1):
        self.model_path = model_path
        self.labels_path = labels_path
        self.camera_index = camera_index
        self.model = self.load_model()
        self.class_names = self.load_labels()
        print('cam_set')
        self.camera = cv2.VideoCapture(self.camera_index)
        print('cam_on')
        self.detected_labels = set()
        self.return_labels = None
        self.running = True
    
    def load_model(self):
        """모델을 로드합니다."""
        return load_model(self.model_path, compile=False)
    
    def load_labels(self):
        """레이블을 로드합니다."""
        with open(self.labels_path, "r", encoding="utf-8") as file:
            return [line.strip().split(' ', 1)[1] for line in file.readlines()]  # 숫자를 제외하고 라벨만 가져오기
    
    def preprocess_image(self, image):
        """이미지를 전처리합니다."""
        image = cv2.flip(image, 1)  # 이미지를 수평으로 반전합니다
        image_resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        image_array = np.asarray(image_resized, dtype=np.float32).reshape(1, 224, 224, 3)
        return (image_array / 127.5) - 1
    
    def predict(self, image_array):
        """모델을 사용하여 예측합니다."""
        prediction = self.model.predict(image_array)
        return prediction[0]
    
    def display_message(self, image, message):
        """이미지에 메시지를 표시합니다."""
        cv2.putText(image, message, (10, image.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
    
    def run(self):
        """웹캠에서 이미지를 캡처하고 예측을 수행합니다."""
        while self.running:
            ret, image = self.camera.read()
            if not ret:
                print("Failed to grab frame")
                continue
            
            image_array = self.preprocess_image(image)
            class_scores = self.predict(image_array)
            
            message = None
            threshold = 0.95  # 95% 이상의 확률
            
            for i, score in enumerate(class_scores):
                if score >= threshold:
                    label = self.class_names[i].strip()
                    if label.lower() != 'none':  # 'none' 레이블은 제외
                        self.detected_labels.add(label)
                        message = f"Detected: {label}"
                        break  # 첫 번째로 95%가 넘는 값을 찾으면 루프 종료
            
            if message:
                self.display_message(image, message)
            
            scale_factor = 1  # 웹캠 화면 사이즈 조절
            new_width = int(image.shape[1] * scale_factor)
            new_height = int(image.shape[0] * scale_factor)
            image_resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            
            cv2.imshow("Webcam Image", image_resized)
            
            keyboard_input = cv2.waitKey(1)
            if keyboard_input == 27:  # ESC 키
                self.running = False
            if keyboard_input == 13: # Enter 키
                if self.detected_labels:
                    self.return_labels = str(self.detected_labels)
                    print(f"{self.return_labels} : type-", type(self.return_labels))
                    return self.return_labels
                
        self.end()
    
    def end(self):
        self.camera.release()
        cv2.destroyAllWindows()
        
        if self.detected_labels:
            print("Final Detected Labels: " + ", ".join(self.detected_labels))
