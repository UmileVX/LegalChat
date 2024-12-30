import os
import psycopg2
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
from uuid import uuid4
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

# ---------------------------------------
# 1. 환경 변수 등 설정
# ---------------------------------------
# OpenAI API 키를 환경 변수에서 읽어옴
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


# PostgreSQL 접속 정보
HOST = "localhost"
PORT = 5432
DBNAME = 'sihm' #os.getenv("DB_NAME", '')
USER = 'sihm' #os.getenv("USER", '')
PASSWORD = 'tech8123' #os.getenv("PW", '')

# 임베딩에 사용할 모델
EMBEDDING_MODEL = "text-embedding-3-large"


# ---------------------------------------
# 2. PostgreSQL에서 DataFrame으로 로드
# ---------------------------------------

def load_data_by_query(query):
    user = 'nonia'
    password = 'tech8120!'
    # host = 'localhost'
    host = '13.209.53.254'
    port = 3306
    database = 'sihm'

    engine = create_engine(
        f'mysql+pymysql://{user}:{password}@{host}:{port}/{database}',
    )

    df = pd.read_sql(query, engine)
    return df

def load_table_to_df():
    """
    sihm.each_paragraph 테이블에서 모든 레코드를 읽어와 
    pandas DataFrame으로 반환합니다.
    """
    print("Loading data from PostgreSQL...")
    engine = create_engine(
        f'postgresql+psycopg2://{USER}:{PASSWORD}@{HOST}:{PORT}/{DBNAME}',
    )
    query = text("""
    SELECT id, 
           name,
           part_num,
           part_name,
           chap_num,
           char_name,
           sec_num,
           sec_name,
           para_num,
           para_name,
           art_num,
           art_name,
           content
    FROM sihm.each_paragraph;
    """)
    df = pd.read_sql(query, engine)
    engine.dispose()
    return df


response = client.embeddings.create(
    model=EMBEDDING_MODEL,
    input=['hello', 'world'],
    dimensions=256
)

# ---------------------------------------
# 3. 임베딩 생성 함수 (배치 처리 예시)
# ---------------------------------------
def get_batch_embeddings(text_list, model=EMBEDDING_MODEL, dimensions=256):
    """
    text_list(문자열 리스트)에 대해 OpenAI Embedding을 요청하고,
    결과 벡터(list of float)를 반환합니다.
    
    - 한 번에 1, 2, ... ~ 수십 개씩 요청 가능. 
      (대량 요청 시 rate limit 주의)
    """
    response = client.embeddings.create(
        model=model,
        input=text_list,
        dimensions=dimensions
    )

    # "data" 키 아래에 각 문장별 embedding 이 순서대로 들어 있음
    embeddings = [item.embedding for item in response.data]
    return embeddings

# ---------------------------------------
# 4. DataFrame에 임베딩 추가
# ---------------------------------------
def create_embeddings_for_df(df, batch_size=100):
    """
    입력 DataFrame(df)의 'content' 컬럼을 기준으로 OpenAI 임베딩을 생성하여
    임베딩 컬럼을 추가한 뒤 반환합니다.
    """
    all_embeddings = []

    # batch_size 간격으로 끊어서 API 호출
    for start_idx in tqdm(range(0, len(df), batch_size), desc="Embedding batches"):
        end_idx = start_idx + batch_size
        batch_texts = df["content"].iloc[start_idx:end_idx].tolist()
        
        batch_embeddings = get_batch_embeddings(batch_texts, model=EMBEDDING_MODEL)
        all_embeddings.extend(batch_embeddings)
    
    # 생성된 임베딩을 새로운 컬럼에 할당
    df["embedding"] = all_embeddings
    print(df.head())
    return df

# ---------------------------------------
# 5. 결과 DataFrame을 새 테이블에 저장
# ---------------------------------------

def create_new_table():
    engine = create_engine(
        f'postgresql+psycopg2://{USER}:{PASSWORD}@{HOST}:{PORT}/{DBNAME}',
    )

    create_table_sql = text("""
    CREATE TABLE IF NOT EXISTS sihm.data_legal_paragraphs (
        id SERIAL PRIMARY KEY,
        name character varying(32) NOT NULL,
        part_num character varying(20),
        part_name character varying(128),
        chap_num character varying(20),
        char_name character varying(100),
        sec_num character varying(20),
        sec_name character varying(100),
        para_num character varying(20),
        para_name character varying(64),
        art_num character varying(20) NOT NULL,
        art_name character varying(128) NOT NULL,
        text TEXT,
        embedding public.vector
    );
    """)

    with engine.connect() as conn:
        conn.execute(create_table_sql)
        conn.commit()
        conn.close()


def save_df_to_new_table(df):
    """
    임베딩이 추가된 DataFrame(df)을 PostgreSQL의 sihm.sihm_paragraphs 테이블에 insert.
    """
    engine = create_engine(
        f'postgresql+psycopg2://{USER}:{PASSWORD}@{HOST}:{PORT}/{DBNAME}',
    )

    insert_sql = """
    INSERT INTO sihm.data_legal_paragraphs (
        name,
        part_num,
        part_name,
        chap_num,
        char_name,
        sec_num,
        sec_name,
        para_num,
        para_name,
        art_num,
        art_name,
        text,
        embedding
    ) VALUES (
        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
    );
    """

    # PostgreSQL pgvector 컬럼에 파이썬 list를 직접 bind 할 수 있는지 여부는
    # psycopg2 버전 및 pgvector 버전에 따라 다릅니다.
    # 일반적으로는 array 형식으로 변환 후 ::vector 캐스팅을 쓰거나,
    # psycopg2에서 지원되는 경우 list 자체를 매핑할 수 있습니다.
    #
    # 여기서는 parameter binding을 사용해 list 자체를 넘기는 예시입니다.
    # 만약 에러가 난다면, 아래 주석 처리한 변환 로직을 사용하세요.

    conn = engine.raw_connection()
    cursor = conn.cursor()

    for row in df.itertuples(index=False):
        # row.embedding 이 list인 경우
        # 만약 쿼리에서 ::vector가 필요한 경우:
        #   "ARRAY[%s]::vector" 와 같은 식으로 문자열 변환을 해야 할 수도 있습니다.
        # 예시(주석 처리):
        # embedding_list = row.embedding
        # embedding_str = "{" + ",".join(map(str, embedding_list)) + "}"  # {0.1,0.2,0.3} 형태
        # -> insert_sql을 f-string으로 만들어 "VALUES (..., '{embedding_str}'::vector)" 식으로 처리

        cursor.execute(
            insert_sql,
            (
                # row.id,
                row.name,
                row.part_num,
                row.part_name,
                row.chap_num,
                row.char_name,
                row.sec_num,
                row.sec_name,
                row.para_num,
                row.para_name,
                row.art_num,
                row.art_name,
                row.content,
                row.embedding  # pgvector가 list 바인딩 지원 시
            )
        )
    
    conn.commit()
    cursor.close()
    conn.close()


# ---------------------------------------
# 6. 전체 파이프라인 실행
# ---------------------------------------
if __name__ == "__main__":
    # 1) 테이블에서 읽어서 DataFrame 만들기
    df_original = load_table_to_df()

    # 2) 임베딩 생성
    df_with_embeddings = create_embeddings_for_df(df_original, batch_size=100)

    # save df_with_embeddings as csv
    df_with_embeddings.to_csv('sihm_paragraphs.csv', index=False)
    # df_with_embeddings = pd.read_csv('sihm_paragraphs.csv')

    # 3) 새 테이블(sihm.sihm_paragraphs)에 저장
    create_new_table()
    save_df_to_new_table(df_with_embeddings)

    print("임베딩 생성 및 테이블 삽입 완료!")
