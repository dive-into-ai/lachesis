import torch
from base import BaseModel


# 학습된 모델 로드
def load_model():
    model = BaseModel()  # 모델 클래스에 맞게 수정 필요
    model.load_state_dict(torch.load("models/model.pth"))
    model.eval()
    return model


# 예측 수행
def predict(model, input_data):
    with torch.no_grad():
        input_tensor = torch.tensor(input_data)
        predictions = model(input_tensor)
        return predictions.tolist()


# 응답 생성
def generate_response(model, user_input):
    with torch.no_grad():
        input_tensor = torch.tensor([user_input])  # 모델에 입력 형식에 맞게 수정 필요
        output = model(input_tensor)
        response = output[0]  # 응답 형식에 맞게 수정 필요
        return response
