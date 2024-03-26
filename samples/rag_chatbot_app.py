import streamlit as st #모든 Streamlit 명령은 "st" 별칭을 통해 사용할 수 있습니다.
import rag_chatbot_lib as glib #로컬 라이브러리 스크립트에 대한 참조

st.set_page_config(page_title="RAG Chatbot") #HTML 제목
st.title("RAG Chatbot") #페이지 제목 

if 'memory' not in st.session_state: #메모리가 아직 생성되지 않았는지 확인합니다.
    st.session_state.memory = glib.get_memory() #메모리를 초기화합니다.

if 'chat_history' not in st.session_state: #채팅 기록이 아직 생성되지 않았는지 확인하기
    st.session_state.chat_history = [] #채팅 기록 초기화하기

if 'vector_index' not in st.session_state: #벡터 인덱스가 아직 생성되지 않았는지 확인합니다.
    with st.spinner("Indexing document..."): #이 블록의 코드가 실행되는 동안 스피너를 표시합니다.
        st.session_state.vector_index = glib.get_index() #지원 라이브러리를 통해 인덱스를 검색하고 앱의 세션 캐시에 저장합니다.

#채팅 기록 다시 렌더링(Streamlit은 이 스크립트를 다시 실행하므로, 이전 채팅 메시지를 보존하려면 이 기능이 필요합니다.)
for message in st.session_state.chat_history: #채팅 기록을 반복해서 살펴보기
    with st.chat_message(message["role"]): #지정된 역할에 대한 채팅 줄을 렌더링하며, with 블록의 모든 내용을 포함합니다.
        st.markdown(message["text"]) #채팅 콘텐츠를 표시합니다.

input_text = st.chat_input("당신의 봇과 여기서 대화하세요") #채팅 입력 상자를 표시합니다.

if input_text: #사용자가 채팅 메시지를 제출한 후 이 if 블록의 코드를 실행합니다.
    
    with st.chat_message("user"): #사용자 채팅 메시지를 표시합니다.
        st.markdown(input_text) #사용자의 최신 메시지를 렌더링합니다.
    
    st.session_state.chat_history.append({"role":"user", "text":input_text}) #사용자의 최신 메시지를 채팅 기록에 추가합니다.

    chat_response = glib.get_rag_chat_response(input_text=input_text, memory=st.session_state.memory, index=st.session_state.vector_index,) #지원 라이브러리를 통해 모델을 호출합니다.

    with st.chat_message("assistant"): #봇 채팅 메시지를 표시합니다.
        st.markdown(chat_response) #봇의 최신 답변을 표시합니다.
    
    st.session_state.chat_history.append({"role":"assistant", "text":chat_response}) #봇의 최신 메시지를 채팅 기록에 추가합니다.
