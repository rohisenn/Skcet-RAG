import warnings
warnings.filterwarnings('ignore')

import streamlit as st
from src.embeddings import get_embedding_model
from src.vector_store import load_vector_store
from src.retriever import get_retriever
from src.rag import run_rag
from src.memory import ConversationMemory
from src.loader import load_documents

st.set_page_config(
    page_title="SKCET RAG Assistant",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main container background */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        margin-bottom: 2rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .main-header h1 {
        color: #ffffff;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    .main-header p {
        color: #e0e7ff;
        font-size: 1.1rem;
        margin-top: 0.5rem;
        font-weight: 400;
    }
    
    .college-logo {
        width: 60px;
        height: 60px;
        margin-bottom: 1rem;
    }
    
    /* Chat container */
    .chat-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.15);
        min-height: 500px;
        backdrop-filter: blur(10px);
    }
    
    /* Chat messages */
    .stChatMessage {
        background: transparent !important;
        padding: 1rem 0 !important;
    }
    
    /* User message */
    [data-testid="stChatMessageContent"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        padding: 1rem 1.5rem !important;
        border-radius: 18px 18px 5px 18px !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
        max-width: 80%;
        margin-left: auto;
    }
    
    /* Assistant message */
    .stChatMessage:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stChatMessageContent"] {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%) !important;
        color: #1a202c !important;
        border-radius: 18px 18px 18px 5px !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1) !important;
        margin-left: 0;
        margin-right: auto;
    }
    
    /* Avatar styling */
    [data-testid="chatAvatarIcon-user"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border-radius: 50%;
    }
    
    [data-testid="chatAvatarIcon-assistant"] {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%) !important;
        border-radius: 50%;
    }
    
    /* Input box */
    .stChatInput {
        border-radius: 25px !important;
        border: 2px solid #667eea !important;
        padding: 0.75rem 1.5rem !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.2) !important;
        background: white !important;
    }
    
    .stChatInput:focus {
        border-color: #764ba2 !important;
        box-shadow: 0 4px 20px rgba(118, 75, 162, 0.3) !important;
    }
    
    /* Input container */
    [data-testid="stChatInput"] {
        background: transparent !important;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Info box */
    .info-box {
        background: linear-gradient(135deg, #e0e7ff 0%, #cfd9ff 100%);
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border-left: 4px solid #667eea;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
    }
    
    .info-box p {
        margin: 0;
        color: #1e3c72;
        font-weight: 500;
    }
    
    /* Typing indicator */
    .typing-indicator {
        display: inline-block;
        padding: 0.5rem 1rem;
        background: #f5f7fa;
        border-radius: 18px;
        margin: 0.5rem 0;
    }
    
    .typing-indicator span {
        height: 8px;
        width: 8px;
        background: #667eea;
        border-radius: 50%;
        display: inline-block;
        margin: 0 2px;
        animation: typing 1.4s infinite;
    }
    
    .typing-indicator span:nth-child(2) {
        animation-delay: 0.2s;
    }
    
    .typing-indicator span:nth-child(3) {
        animation-delay: 0.4s;
    }
    
    @keyframes typing {
        0%, 60%, 100% {
            transform: translateY(0);
            opacity: 0.4;
        }
        30% {
            transform: translateY(-10px);
            opacity: 1;
        }
    }
    

</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <div style="font-size: 3rem; margin-bottom: 0.5rem;">🎓</div>
    <h1>SKCET RAG Assistant</h1>
    <p>Sri Krishna College of Engineering and Technology</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
    <p>💡 Ask me anything about SKCET - courses, facilities, admissions, placements, and more!</p>
</div>
""", unsafe_allow_html=True)

if "memory" not in st.session_state:
    st.session_state.memory = ConversationMemory(max_turns=5)
if "messages" not in st.session_state:
    st.session_state.messages = []

@st.cache_resource
def load_rag():
    docs = load_documents()
    embedding_model = get_embedding_model()
    vectordb = load_vector_store(embedding_model)
    retriever = get_retriever(vectordb, docs)
    return retriever

with st.spinner("🔄 Loading knowledge base..."):
    retriever = load_rag()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if len(st.session_state.messages) == 0:
    with st.chat_message("assistant"):
        st.markdown("Hello! I'm your SKCET Knowledge Assistant. How can I help you today?")

user_input = st.chat_input("Type your question here...")

if user_input:
    
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
  
    with st.chat_message("assistant"):
        with st.spinner(""):
            
            answer = run_rag(
                user_input,
                retriever,
                st.session_state.memory
            )
        
        
        st.markdown(answer)
    
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()