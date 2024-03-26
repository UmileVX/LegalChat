import os
from langchain.memory import ConversationBufferWindowMemory
from langchain.llms.bedrock import Bedrock
from langchain.chains import ConversationalRetrievalChain

from langchain.embeddings import BedrockEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader


def get_llm():
        
    model_kwargs =  { #Anthropic 모델
        "max_tokens_to_sample": 1024,
        "temperature": 0, 
        "top_k": 250, 
        "top_p": 0.5, 
        "stop_sequences": ["\n\nHuman:"] 
    }
    
    llm = Bedrock(
        credentials_profile_name=os.environ.get("BWB_PROFILE_NAME"), #AWS 자격 증명에 사용할 프로필 이름 설정(기본값이 아닌 경우).
        region_name=os.environ.get("BWB_REGION_NAME"), #리전 이름 설정(기본값이 아닌 경우).
        endpoint_url=os.environ.get("BWB_ENDPOINT_URL"), #엔드포인트 URL 설정(필요한 경우)
        model_id="anthropic.claude-v2:1", #파운데이션 모델 설정
        model_kwargs=model_kwargs) #Claud에 대한 속성 구성
    
    return llm

def get_index(): #애플리케이션에서 사용할 인메모리 벡터 저장소를 생성하고 반환합니다.
    
    embeddings = BedrockEmbeddings(
        credentials_profile_name=os.environ.get("BWB_PROFILE_NAME"), #AWS 자격 증명에 사용할 프로필 이름을 설정합니다(기본값이 아닌 경우)
        region_name=os.environ.get("BWB_REGION_NAME"),  #리전 이름을 설정합니다(기본값이 아닌 경우)
        endpoint_url=os.environ.get("BWB_ENDPOINT_URL"), #엔드포인트 URL 설정(필요한 경우)
    ) #Titan Embeddings 생성합니다.
    
    #pdf_path = "2022-Shareholder-Letter.pdf" #이 이름을 가진 로컬 PDF 파일을 가정합니다.
    pdf_path = "data.pdf"

    loader = PyPDFLoader(file_path=pdf_path) #PDF 파일 로드하기
    
    text_splitter = RecursiveCharacterTextSplitter( #텍스트 분할기 만들기
        separators=["\n\n", "\n", ".", " "], #(1) 단락, (2) 줄, (3) 문장 또는 (4) 단어 순서로 청크를 분할합니다.
        chunk_size=1000, #위의 구분 기호를 사용하여 1000자 청크로 나눕니다.
        chunk_overlap=100 #이전 청크와 겹칠 수 있는 문자 수입니다.
    )
    
    index_creator = VectorstoreIndexCreator( #벡터 스토어 팩토리 만들기
        vectorstore_cls=FAISS, #데모 목적으로 인메모리 벡터 저장소를 사용합니다.
        embedding=embeddings, #Titan 임베딩 사용
        text_splitter=text_splitter, #재귀적 텍스트 분할기 사용하기
    )
    
    index_from_loader = index_creator.from_loaders([loader]) #로드된 PDF에서 벡터 스토어 인덱스를 생성합니다.
    
    return index_from_loader #클라이언트 앱에서 캐시할 인덱스를 반환합니다.


def get_memory(): #이 채팅 세션을 위한 메모리 만들기
    
    memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True) #이전 메시지의 기록을 유지합니다.
    return memory


def get_rag_chat_response(input_text, memory, index): #chat client 함수
    
    llm = get_llm()
    
    conversation_with_retrieval = ConversationalRetrievalChain.from_llm(llm, index.vectorstore.as_retriever(), memory=memory)
    
    chat_response = conversation_with_retrieval({"question": input_text}) #사용자 메시지, 기록 및 지식을 모델에 전달합니다.
    
    return chat_response['answer']
