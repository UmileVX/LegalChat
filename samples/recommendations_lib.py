import os
from langchain.llms.bedrock import Bedrock
from langchain.embeddings import BedrockEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import JSONLoader


def get_llm():
    
    model_kwargs =  { #Anthropic 모델
            "max_tokens_to_sample": 1024,
            "temperature": 0, 
            "top_k": 250, 
            "top_p": 0.5, 
            "stop_sequences": ["\n\nHuman:"] 
    }
    
    llm = Bedrock(
        credentials_profile_name=os.environ.get("BWB_PROFILE_NAME"), #AWS 자격 증명에 사용할 프로필 이름을 설정합니다(기본값이 아닌 경우)
        region_name=os.environ.get("BWB_REGION_NAME"), #리전 이름을 설정합니다(기본값이 아닌 경우)
        endpoint_url=os.environ.get("BWB_ENDPOINT_URL"), #엔드포인트 URL 설정(필요한 경우)
        model_id="anthropic.claude-v2:1", #파운데이션 모델 설정합니다
        model_kwargs=model_kwargs) #Claude의 속성을 구성합니다
    
    return llm


#함수를 사용하여 벡터스토어에서 캡처할 메타데이터를 식별하고 일치하는 콘텐츠와 함께 반환합니다.
def item_metadata_func(record: dict, metadata: dict) -> dict: 

    metadata["name"] = record.get("name")
    metadata["url"] = record.get("url")

    return metadata


def get_index(): #애플리케이션에서 사용할 인메모리 벡터 저장소를 생성하고 반환합니다.
    
    embeddings = BedrockEmbeddings(
        credentials_profile_name=os.environ.get("BWB_PROFILE_NAME"), #AWS 자격 증명에 사용할 프로필 이름을 설정합니다(기본값이 아닌 경우)
        region_name=os.environ.get("BWB_REGION_NAME"), #리전 이름을 설정합니다(기본값이 아닌 경우)
        endpoint_url=os.environ.get("BWB_ENDPOINT_URL"), #endpoint URL을 설정합니다 (필요한 경우)
    ) #Titan Embeddings 클라이언트를 생성합니다.
    
    loader = JSONLoader(
        file_path="services.json",
        jq_schema='.[]',
        content_key='description',
        metadata_func=item_metadata_func)

    text_splitter = RecursiveCharacterTextSplitter( #텍스트 분할기 만들기
        separators=["\n\n", "\n", ".", " "], #(1) 단락, (2) 줄, (3) 문장 또는 (4) 단어 순서로 청크를 분할합니다.
        chunk_size=8000, #이 콘텐츠를 기반으로 전체 항목을 원하므로 청크가 필요하지 않습니다. 콘텐츠가 너무 길면 오류가 발생할 수 있습니다.
        chunk_overlap=0 #이전 청크와 겹칠 수 있는 문자 수
    )
    
    index_creator = VectorstoreIndexCreator( #벡터 스토어 팩토리 만들기
        vectorstore_cls=FAISS, #데모 목적으로 인메모리 벡터 저장소를 사용합니다.
        embedding=embeddings, #Titan embeddings 사용하기
        text_splitter=text_splitter, #재귀적 텍스트 분할기 사용하기
    )
    
    index_from_loader = index_creator.from_loaders([loader]) #로드된 PDF에서 벡터 스토어 인덱스를 생성합니다.

    return index_from_loader #클라이언트 앱에서 캐시할 인덱스를 반환합니다.


def get_similarity_search_results(index, question):
    raw_results = index.vectorstore.similarity_search_with_score(question)
    
    llm = get_llm()
    
    results = []
    
    for res in raw_results:
        content = res[0].page_content
        prompt = f"{content}\n\n위의 서비스가 다음과 같은 요구 사항을 해결하는 방법을 요약하세요. : {question}"
        
        summary = llm(prompt)
        
        results.append({"name": res[0].metadata["name"], "url": res[0].metadata["url"], "summary": summary, "original": content})
    
    return results
