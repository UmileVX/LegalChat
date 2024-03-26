import streamlit as st #모든 Streamlit 명령은 "st" 별칭을 통해 사용할 수 있습니다.
import streaming_lib as glib #로컬 라이브러리 스크립트에 대한 참조
from langchain.callbacks import StreamlitCallbackHandler


st.set_page_config(page_title="응답 스트리밍") #HTML 제목
st.title("응답 스트리밍") #페이지 제목

input_text = st.text_area("Input text", label_visibility="collapsed") #레이블이 없는 여러 줄 텍스트 상자를 표시합니다.
go_button = st.button("Go", type="primary") #기본 버튼을 표시합니다.

if go_button: #버튼을 클릭하면 이 if 블록의 코드가 실행됩니다.
    #스트리밍 아웃풋을 위해 빈 컨테이너를 사용합니다 
    st_callback = StreamlitCallbackHandler(st.container())
    streaming_response = glib.get_streaming_response(prompt=input_text, streaming_callback=st_callback)
