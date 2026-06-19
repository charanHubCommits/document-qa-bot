import streamlit as st
import os
import chromadb
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent
load_dotenv(ROOT_DIR / ".env")

from src.ingest import ingest_documents, discover_documents
from src.query import query_rag_pipeline
from src.config import TOP_K, DATA_DIR, DB_DIR, COLLECTION_NAME

# Page configuration
st.set_page_config(
    page_title="Grounded Q&A Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

/* Application layout adjustments */
html, body, [class*="css"], .stApp {
    font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
    background: linear-gradient(135deg, #0b0f19 0%, #111827 55%, #1d152c 100%);
    color: #f1f5f9;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: rgba(17, 24, 39, 0.85) !important;
    backdrop-filter: blur(12px);
    border-right: 1px solid rgba(255, 255, 255, 0.05);
}

/* Sidebar custom divider */
.sidebar-divider {
    height: 1px;
    background: linear-gradient(90deg, rgba(255,255,255,0) 0%, rgba(255,255,255,0.1) 50%, rgba(255,255,255,0) 100%);
    margin: 15px 0;
}

/* Buttons styling */
.stButton>button {
    background: linear-gradient(135deg, #8b5cf6 0%, #6d28d9 100%) !important;
    color: #ffffff !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 8px !important;
    padding: 10px 20px !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 15px rgba(139, 92, 246, 0.25) !important;
    transition: all 0.25s ease !important;
    width: 100%;
}

.stButton>button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(139, 92, 246, 0.4) !important;
    border-color: rgba(255, 255, 255, 0.2) !important;
}

.stButton>button:active {
    transform: translateY(1px) !important;
}

/* Chat Input Styling */
.stChatInputContainer {
    border-radius: 12px !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    background-color: rgba(30, 41, 59, 0.6) !important;
    backdrop-filter: blur(8px) !important;
}

/* Expanders styling */
.streamlit-expanderHeader {
    background-color: rgba(30, 41, 59, 0.4) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)

# DB check
def get_db_document_count() -> int:
    if not DB_DIR.exists():
        return 0
    try:
        client = chromadb.PersistentClient(path=str(DB_DIR))
        collection = client.get_collection(name=COLLECTION_NAME)
        return collection.count()
    except Exception:
        return 0


if "messages" not in st.session_state:
    st.session_state.messages = []

st.markdown(
    """
    <div style="padding: 20px; border-radius: 12px; background: linear-gradient(135deg, rgba(139, 92, 246, 0.15) 0%, rgba(59, 130, 246, 0.08) 100%); border: 1px solid rgba(255, 255, 255, 0.05); margin-bottom: 25px;">
        <h1 style="margin: 0; font-size: 2.2rem; background: linear-gradient(90deg, #a78bfa, #60a5fa); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 700;">🤖 Grounded Document Q&A</h1>
        <p style="margin: 8px 0 0 0; color: #94a3b8; font-size: 1.05rem;">Query your document collection with strict verification. All answers are grounded in context with source citations.</p>
    </div>
    """,
    unsafe_allow_html=True
)


with st.sidebar:
    st.markdown("### ⚙️ Control Dashboard")
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    top_k = st.slider("Retrieval Size (TOP_K)", min_value=1, max_value=10, value=TOP_K, step=1)
    
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.markdown("### 📁 Document Library")

    docs = discover_documents()
    if not docs:
        st.warning("No files found in the data/ directory. Place PDF or DOCX files here.")
    else:
        for doc in docs:
            st.markdown(
                f"""
                <div style="background-color: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 6px; padding: 8px 12px; margin-bottom: 6px; font-size: 0.9rem; color: #cbd5e1; display: flex; align-items: center; gap: 8px;">
                    📄 {doc.name}
                </div>
                """,
                unsafe_allow_html=True
            )
            
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    
    if st.button("🔄 Re-ingest Documents"):
        with st.spinner("Processing document library..."):
            try:
                num_chunks = ingest_documents()
                st.success(f"Successfully ingested {num_chunks} chunks!")
                st.rerun()
            except Exception as e:
                st.error(f"Ingestion failed: {e}")

    db_docs = get_db_document_count()
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.caption(f"Vector Database status: **{db_docs} chunks stored**")


db_ready = get_db_document_count() > 0

if not db_ready:
    st.info("👋 Welcome! The vector database is currently empty. Place some documents in the `data/` folder and click **'Re-ingest Documents'** in the sidebar to begin.")
else:
    # Render previous messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "sources" in msg and msg["sources"]:
                with st.expander("📚 View Sources & Citations"):
                    for idx, src in enumerate(msg["sources"], start=1):
                        st.markdown(
                            f"""
                            <div style="background-color: rgba(255, 255, 255, 0.03); padding: 12px; border-radius: 8px; border-left: 3px solid #8b5cf6; margin-bottom: 10px;">
                                <span style="font-weight: 600; color: #a78bfa;">[{idx}] {src['file']}</span> 
                                <span style="font-size: 0.8rem; color: #94a3b8; float: right;">Page {src['page'] if src['page'] != -1 else 'N/A'}</span>
                                <div style="margin-top: 6px; font-size: 0.88rem; color: #cbd5e1; line-height: 1.45; white-space: pre-wrap;">
                                    {src['snippet']}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

    if user_query := st.chat_input("Ask a question about your documents..."):
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})

        with st.chat_message("assistant"):
            answer_placeholder = st.empty()
            with st.spinner("Searching documents & generating answer..."):
                try:
                    result = query_rag_pipeline(user_query=user_query, k=top_k)
                    answer = result["answer"]
                    sources = result["sources"]

                    answer_placeholder.markdown(answer)
                    
                    if sources:
                        with st.expander("📚 View Sources & Citations"):
                            for idx, src in enumerate(sources, start=1):
                                st.markdown(
                                    f"""
                                    <div style="background-color: rgba(255, 255, 255, 0.03); padding: 12px; border-radius: 8px; border-left: 3px solid #8b5cf6; margin-bottom: 10px;">
                                        <span style="font-weight: 600; color: #a78bfa;">[{idx}] {src['file']}</span> 
                                        <span style="font-size: 0.8rem; color: #94a3b8; float: right;">Page {src['page'] if src['page'] != -1 else 'N/A'}</span>
                                        <div style="margin-top: 6px; font-size: 0.88rem; color: #cbd5e1; line-height: 1.45; white-space: pre-wrap;">
                                            {src['snippet']}
                                        </div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                except Exception as e:
                    error_msg = f"An error occurred: {e}"
                    answer_placeholder.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })
