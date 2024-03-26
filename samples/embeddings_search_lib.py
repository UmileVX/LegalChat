import os
from langchain.embeddings import BedrockEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders.csv_loader import CSVLoader


def get_index(): #애플리케이션에서 사용할 인메모리 벡터 저장소를 생성하고 반환합니다.
    
    embeddings = BedrockEmbeddings(
        credentials_profile_name=os.environ.get("BWB_PROFILE_NAME"), #AWS 자격 증명에 사용할 프로필 이름을 설정합니다(기본값이 아닌 경우).
        region_name=os.environ.get("BWB_REGION_NAME"), #지역 이름을 설정합니다(기본값이 아닌 경우).
        endpoint_url=os.environ.get("BWB_ENDPOINT_URL"), #엔드포인트 URL을 설정합니다(필요한 경우).
    ) #Titan Embedding 클라이언트 생성하기
    
    loader = CSVLoader(file_path="sagemaker_answers.csv")

    index_creator = VectorstoreIndexCreator(
        vectorstore_cls=FAISS,
        embedding=embeddings,
        text_splitter=CharacterTextSplitter(chunk_size=300, chunk_overlap=0),
    )

    index_from_loader = index_creator.from_loaders([loader])
    
    return index_from_loader


def get_similarity_search_results(index, question):
    results = index.vectorstore.similarity_search_with_score(question)
    
    flattened_results = [{"content":res[0].page_content, "score":res[1]} for res in results] #더 쉽게 표시하고 다룰 수 있도록 결과를 평탄화(flatten)합니다.
    
    return flattened_results


def get_embedding(text):
    embeddings = BedrockEmbeddings(
        credentials_profile_name=os.environ.get("BWB_PROFILE_NAME"), #AWS 자격 증명에 사용할 프로필 이름을 설정합니다(기본값이 아닌 경우).
        region_name=os.environ.get("BWB_REGION_NAME"), #지역 이름을 설정합니다(기본값이 아닌 경우).
        endpoint_url=os.environ.get("BWB_ENDPOINT_URL"), #엔드포인트 URL을 설정합니다(필요한 경우).
    ) #Titan Embedding 클라이언트 생성하기
    
    return embeddings.embed_query(text)
