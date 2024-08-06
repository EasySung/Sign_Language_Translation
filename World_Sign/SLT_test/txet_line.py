import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from collections import defaultdict

# Mediapipe 손 추적기 설정
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# 학습된 모델 로드
model = tf.keras.models.load_model('sign_language_model.h5')

# 클래스 라벨 설정 (모델 학습 시 사용했던 순서대로)
classes = ['옆쪽', '오늘', '화장실', '화재']  # 수어 단어 클래스 리스트

# 인식된 단어와 해당 단어의 검출 횟수 저장
detected_words = defaultdict(int)
recognized_words = set()

# 웹캠 캡처 시작
cap = cv2.VideoCapture(1)

while cap.isOpened():
    ret, frame = cap.read()  # 웹캠의 한 프레임 읽기
    if not ret:
        break

    # 프레임을 Mediapipe로 처리
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 프레임을 RGB로 변환
    result = hands.process(frame_rgb)  # 손 랜드마크 인식 수행

    if result.multi_hand_landmarks:
        # 양손의 랜드마크를 저장할 배열
        left_hand_landmarks = np.zeros((21, 3))
        right_hand_landmarks = np.zeros((21, 3))

        for hand_landmarks, hand_label in zip(result.multi_hand_landmarks, result.multi_handedness):
            handedness = hand_label.classification[0].label

            # 왼손과 오른손을 구분하여 저장
            if handedness == 'Left':
                left_hand_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
            elif handedness == 'Right':
                right_hand_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])

        # 양손 랜드마크를 하나의 배열로 결합
        landmarks = np.concatenate([left_hand_landmarks, right_hand_landmarks])
        landmarks = landmarks.reshape(1, 42, 3)  # 모델 입력 형식에 맞게 변환

        # 모델 예측 수행
        prediction = model.predict(landmarks)
        predicted_class = np.argmax(prediction)
        predicted_label = classes[predicted_class]

        # 검출된 단어 횟수 증가
        detected_words[predicted_label] += 1

        # 검출 횟수가 50 이상이면 리스트에 포함
        if detected_words[predicted_label] >= 10:
            recognized_words.add(predicted_label)

        # 결과를 프레임에 표시
        cv2.putText(frame, predicted_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        print(predicted_label)

        # 랜드마크를 프레임에 그리기
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # 프레임 출력
    cv2.imshow('Sign Language Recognition', frame)
    print("Recognized words:")
    print(list(recognized_words))

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()

# 인식된 단어 리스트 출력
print("Recognized words:")
print(list(recognized_words))
