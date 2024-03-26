import streamlit as st #모든 streamlit 명령은 "st" 별칭을 통해 사용할 수 있습니다.
import embeddings_search_lib as glib #로컬 라이브러리 스크립트에 대한 참조


st.set_page_config(page_title="Embeddings Search", layout="wide") #HTML 제목
st.title("Embeddings Search") #페이지 제목


if 'vector_index' not in st.session_state: #벡터 인덱스가 아직 생성되지 않았는지 확인합니다.
    with st.spinner("Indexing document..."): #이 블록의 코드가 실행되는 동안 스피너를 표시합니다.
        st.session_state.vector_index = glib.get_index() #지원 라이브러리를 통해 인덱스를 검색하고 앱의 세션 캐시에 저장합니다.

input_text = st.text_input("Ask a question about Amazon SageMaker:") #레이블 없이 여러 줄 텍스트 상자를 표시합니다.
go_button = st.button("Go", type="primary") #기본 버튼을 표시합니다.


if go_button: #버튼이 클릭될 때 이 if 블록의 코드가 실행됩니다.
    
    with st.spinner("Working..."): #이 블록의 코드가 실행되는 동안 스피너를 표시합니다.
        response_content = glib.get_similarity_search_results(index=st.session_state.vector_index, question=input_text)
        
        st.table(response_content) #테이블을 사용하여 텍스트가 줄 바꿈되도록 합니다.
        
        
        raw_embedding = glib.get_embedding(input_text)
        
        with st.expander("View question embedding"):
            st.json(raw_embedding)

