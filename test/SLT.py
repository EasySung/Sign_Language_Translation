import openai
import os

# OpenAI API 키 설정
openai.api_key = "sk-proj-MgYQyplQwtVSJLVdcb5dT3BlbkFJCzGstWNJiHm41HAGF3DK"

# ChatCompletion API를 사용하여 대화 생성
response = openai.ChatCompletion.create(
    model="gpt-4o-mini",  # 모델 이름을 최신 모델로 설정
    messages=[
        {"role": "system", "content": "한글 단어를 한국어로 번역 후 자연스러운 한국어 문장으로 만들어줘"},
        {"role": "user", "content": "나" + "너" + "좋아"}
    ]
)

# 응답 출력
print("Assistant: " + response.choices[0].message['content'])
