import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import streamlit as st


# 모델 로드
def load_model():
    model_name = "t5-base"  # 예시로 T5 모델을 사용합니다. 원하는 트랜스포머 모델로 변경 가능합니다.
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


# 응답 생성
def generate_response(tokenizer, model, user_input, chat_history=None):
    if chat_history is None:
        chat_history = []
    inputs = tokenizer.encode(user_input, add_special_tokens=True)
    chat_history.extend(inputs)

    with torch.no_grad():
        inputs_tensor = torch.tensor([chat_history])

        # 다양하게 생성되도록
        outputs = model.generate(
            inputs_tensor,
            max_length=50,
            pad_token_id=tokenizer.pad_token_id,
            temperature=1.0,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            num_return_sequences=1,
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response, chat_history


# 챗봇 대화 기능
def chatbot(tokenizer, model):
    st.title("Chatbot")

    chat_history = []
    while True:
        user_input = st.text_input("Enter your message", key="user_input")
        if st.button("Send") or (user_input and st.session_state.user_input):
            chat_history.append(user_input)
            # 여기에 챗봇의 응답 생성 로직을 구현하세요
            # 챗봇의 응답을 response 변수에 할당하세요

            chat_history.append(user_input)
            response, chat_history = generate_response(
                tokenizer, model, user_input, chat_history
            )
            chat_history.append(tokenizer.encode(response, add_special_tokens=True))

            st.text_area("Chat History", "\n".join(chat_history))
            st.text_area("Chatbot Response", response)

            # 이전 입력값 저장
            st.session_state.user_input = user_input

            # 입력값 초기화
            st.text_input("Enter your message", key="user_input", value="")

        if st.button("End Chat"):
            st.stop()


# 메인 함수
def main():
    # 모델 로드
    tokenizer, model = load_model()

    # 챗봇 실행
    chatbot(tokenizer, model)


if __name__ == "__main__":
    main()
