from langchain.chains import ConversationChain
from langchain.llms import Bedrock

def get_llm(streaming_callback):
    model_kwargs =  { #Anthropic 모델
        "max_tokens_to_sample": 4000,
        "temperature": 0, 
        "top_k": 250, 
        "top_p": 0.5, 
        "stop_sequences": ["\n\nHuman:"] 
    }
    
    llm = Bedrock(
        model_id="anthropic.claude-v2:1", #파운데이션 모델 설정
        model_kwargs=model_kwargs,#Claud에 대한 속성 구성
        streaming = True,
        callbacks = [streaming_callback],
        )
    return llm


def get_streaming_response(prompt, streaming_callback):
    conversation_with_summary = ConversationChain(
        llm = get_llm(streaming_callback)
    )
    return conversation_with_summary.predict(input=prompt)
