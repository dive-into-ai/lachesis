from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from inference import load_model, predict

# FastAPI 애플리케이션 생성
app = FastAPI()

# 요청 데이터를 위한 모델 정의
class PredictionRequest(BaseModel):
    input_data: List[List[float]]

# 모델 로드
model = load_model()

# 예측 엔드포인트
@app.post("/predict")
def run_prediction(request: PredictionRequest):
    input_data = request.input_data
    predictions = predict(model, input_data)
    return {"predictions": predictions}
