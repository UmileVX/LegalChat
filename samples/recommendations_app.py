import streamlit as st #모든 streamlit명령은 "st" 별칭을 통해 사용할 수 있습니다.
import recommendations_lib as glib #로컬 라이브러리 스크립트에 대한 참조


st.set_page_config(page_title="개인 맞춤형 추천", layout="wide") #HTML 제목
st.title("개인 맞춤형 추천") #페이지 제목


if 'vector_index' not in st.session_state: #벡터 인덱스가 아직 생성되지 않았는지 확인합니다.
    with st.spinner("문서 인덱싱 중..."): #이 블록의 코드가 실행되는 동안 스피너를 표시합니다.
        st.session_state.vector_index = glib.get_index() #지원 라이브러리를 통해 인덱스를 검색하고 앱의 세션 캐시에 저장합니다.


input_text = st.text_input("클라우드 서비스에서 필요한 몇 가지 주요 기능에 대해 설명하세요:") #레이블 없이 여러 줄 텍스트 상자를 표시합니다.
go_button = st.button("Go", type="primary") #기본 버튼을 표시합니다.


if go_button: #버튼이 클릭될 때 이 if 블록의 코드가 실행됩니다.
    
    with st.spinner("Working..."): #이 블록의 코드가 실행되는 동안 스피너를 표시합니다.
        response_content = glib.get_similarity_search_results(index=st.session_state.vector_index, question=input_text)
        
        for result in response_content:
            st.markdown(f"### [{result['name']}]({result['url']})")
            st.write(result['summary'])
            with st.expander("Original"):
                st.write(result['original'])
