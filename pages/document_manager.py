import streamlit as st
import os
import shutil

from src.auth import check_password
from src.embeddings import get_embedding_model
from src.vector_store import load_vector_store, create_vector_store
from src.chunker import chunk_documents
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader

st.set_page_config(
    page_title="Document Manager - SKCET Admin",
    page_icon="📁",
    layout="wide",
)

if not check_password():
    st.stop()

st.title("📁 Document Manager")
st.markdown("Upload new documents or remove existing ones from the knowledge base. Changes take effect immediately.")

PDF_DIR = os.path.join("data", "pdfs")
os.makedirs(PDF_DIR, exist_ok=True)


def re_index_pdf(file_path: str):
    """Load a single PDF, chunk it, and upsert into ChromaDB."""
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    chunks = chunk_documents(docs)
    embedding_model = get_embedding_model()
    vectordb = load_vector_store(embedding_model)
    vectordb.add_documents(chunks)
    # Clear cached retriever so the main chat picks up new docs on next query
    st.cache_resource.clear()


def delete_pdf(filename: str):
    """Remove the PDF from disk and from ChromaDB by source metadata."""
    file_path = os.path.join(PDF_DIR, filename)
    if os.path.exists(file_path):
        os.remove(file_path)

    embedding_model = get_embedding_model()
    vectordb = load_vector_store(embedding_model)

    # Find and delete all chunks whose source matches this file
    try:
        results = vectordb.get(where={"source": file_path})
        ids_to_delete = results.get("ids", [])
        if ids_to_delete:
            vectordb.delete(ids=ids_to_delete)
    except Exception:
        pass  # If no matching embeddings were found, that's fine

    st.cache_resource.clear()


# ── Upload Section ─────────────────────────────────────────────────────────────
st.subheader("📤 Upload New Documents")

uploaded_files = st.file_uploader(
    "Drag and drop PDF files here",
    type=["pdf"],
    accept_multiple_files=True,
    help="Uploaded PDFs will be indexed immediately into the knowledge base."
)

if uploaded_files:
    if st.button("➕ Add to Knowledge Base", type="primary", use_container_width=True):
        for uploaded_file in uploaded_files:
            save_path = os.path.join(PDF_DIR, uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.spinner(f"Indexing `{uploaded_file.name}`..."):
                try:
                    re_index_pdf(save_path)
                    st.success(f"✅ `{uploaded_file.name}` added and indexed!")
                except Exception as e:
                    st.error(f"❌ Failed to index `{uploaded_file.name}`: {e}")

st.markdown("---")

# ── Web Scraping Section ───────────────────────────────────────────────────────
st.subheader("🌐 Add Webpage URL")
st.markdown("Enter any public URL (e.g. SKCET announcements page) to scrape and index its text into the knowledge base.")
url_input = st.text_input("Webpage URL", placeholder="https://skcet.ac.in/news...")
if st.button("➕ Index Webpage URL", type="secondary", use_container_width=True):
    if url_input:
        with st.spinner(f"Scraping `{url_input}`..."):
            try:
                loader = WebBaseLoader(url_input)
                docs = loader.load()
                for d in docs:
                    d.metadata["source"] = url_input # Normalize source name
                chunks = chunk_documents(docs)
                embedding_model = get_embedding_model()
                vectordb = load_vector_store(embedding_model)
                vectordb.add_documents(chunks)
                st.cache_resource.clear()
                st.success(f"✅ Webpage indexed: {url_input}")
            except Exception as e:
                st.error(f"❌ Failed to scrape URL: {e}")
    else:
        st.warning("Please enter a URL first.")

st.markdown("---")

# ── Existing Documents Section ─────────────────────────────────────────────────
st.subheader("📂 Existing Documents in Knowledge Base")

pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]

if not pdf_files:
    st.info("No PDF documents found in the knowledge base. Upload some above!")
else:
    for filename in sorted(pdf_files):
        file_path = os.path.join(PDF_DIR, filename)
        size_kb = os.path.getsize(file_path) / 1024

        col1, col2, col3 = st.columns([5, 2, 1])
        with col1:
            st.markdown(f"📄 **{filename}**")
        with col2:
            st.caption(f"{size_kb:.1f} KB")
        with col3:
            if st.button("🗑️", key=f"delete_{filename}", help=f"Delete {filename}"):
                with st.spinner(f"Removing `{filename}`..."):
                    delete_pdf(filename)
                st.success(f"🗑️ `{filename}` removed.")
                st.rerun()

st.markdown("---")
st.caption("ℹ️ After adding or removing documents, the main chatbot will automatically use the updated knowledge base on the next query.")
