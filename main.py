# main.py
import os
from object_detector import WebcamObjectDetector
from openai_chat import OpenAIChat

def main():
    # 현재 작업 디렉토리
    base_path = os.getcwd()
    model_file = os.path.join(base_path, 'keras_model.h5')
    labels_file = os.path.join(base_path, 'labels.txt')

    # OpenAI API 키와 모델 설정
    api_key = "API_KEY"
    model = "gpt-4o-mini"  # 사용할 모델명

    # OpenAIChat 클래스의 인스턴스 생성
    chat = OpenAIChat(api_key=api_key, model=model)

    # WebcamObjectDetector 인스턴스 생성
    detector = WebcamObjectDetector(model_path=model_file, labels_path=labels_file, camera_index=1)
    
    # 객체 감지 실행
    detector.running = True
    while detector.running:
        detected_labels = detector.run()
        if detected_labels:
            response = chat.generate_response(detected_labels)
            print("Assistant: " + response)

if __name__ == "__main__":
    main()
