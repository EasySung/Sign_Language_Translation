from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np
import os  # os 모듈 import

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# 경로로 파일 경로 정의
base_path = os.getcwd()
model_file = os.path.join(base_path, 'keras_model.h5')
lambda_file = os.path.join(base_path, 'labels.txt')
model_path = model_file
labels_path = lambda_file

# 모델 로드
model = load_model(model_path, compile=False)

# 레이블 로드
with open(labels_path, "r", encoding="utf-8") as file:
    class_names = [line.strip().split(' ', 1)[1] for line in file.readlines()]  # 숫자를 제외하고 라벨만 가져오기

# CAMERA can be 0 or 1 based on default camera of your computer
camera = cv2.VideoCapture(1)

detected_labels = set()  # 예측된 레이블을 저장할 집합

while True:
    # Grab the webcamera's image.
    ret, image = camera.read()

    if not ret:
        print("Failed to grab frame")
        continue
    
    # 이미지를 수평으로 반전합니다
    image = cv2.flip(image, 1)
    image_resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # 이미지를 numpy 배열로 변환하고 정규화
    image_array = np.asarray(image_resized, dtype=np.float32).reshape(1, 224, 224, 3)
    image_array = (image_array / 127.5) - 1

    # 모델 예측
    prediction = model.predict(image_array)
    class_scores = prediction[0]

    # 사용자 지정 문장 설정
    message = None
    threshold = 0.95  # 95% 이상의 확률
    
    for i, score in enumerate(class_scores):
        if score >= threshold:
            label = class_names[i].strip()
            if label.lower() != 'none':  # 'none' 레이블은 제외
                detected_labels.add(label)
                message = f"Detected: {label}"
                break  # 첫 번째로 95%가 넘는 값을 찾으면 루프 종료
    
    # 사용자 지정 메시지가 있는 경우 이미지에 표시
    if message:
        cv2.putText(image, message, (10, image.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

    # 화면 크기 조절
    scale_factor = 1  # 웹캠 화면 사이즈 조절
    new_width = int(image.shape[1] * scale_factor)
    new_height = int(image.shape[0] * scale_factor)
    image_resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    # Show the image in a window
    cv2.imshow("Webcam Image", image_resized)

    # 27 is the ASCII for the esc key on your keyboard.
    keyboard_input = cv2.waitKey(1)
    if keyboard_input == 27:  # ESC 키
        break

camera.release()
cv2.destroyAllWindows()

# 결과창에 나타내기
if detected_labels:
    print("Final Detected Labels: " + ", ".join(detected_labels))