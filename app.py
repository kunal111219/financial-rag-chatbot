import streamlit as st
import tempfile
import os
from rag_pipeline import load_and_split, create_vectorstore, build_rag_chain, query_document

st.set_page_config(
    page_title="Financial RAG Chatbot",
    page_icon="💹",
    layout="wide"
)

st.title("💹 Financial Document Chatbot")
st.caption("Powered by Groq LLaMA3 + LangChain + FAISS")

# Sidebar
with st.sidebar:
    st.header("📄 Upload Document")
    uploaded_file = st.file_uploader(
        "Upload Annual Report / Financial PDF",
        type=["pdf"]
    )
    st.markdown("---")
    st.markdown("**Suggested Questions:**")
    st.markdown("- What was the total revenue?")
    st.markdown("- What are the main risk factors?")
    st.markdown("- Summarize the CEO's message")
    st.markdown("- What is the net profit margin?")
    st.markdown("- What are the future growth plans?")

# Session state
if "chain" not in st.session_state:
    st.session_state.chain = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "doc_processed" not in st.session_state:
    st.session_state.doc_processed = False

# Process PDF
if uploaded_file and not st.session_state.doc_processed:
    with st.spinner("🔄 Processing document... please wait"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        chunks = load_and_split(tmp_path)
        vectorstore = create_vectorstore(chunks)
        chain, retriever = build_rag_chain(vectorstore)   
        st.session_state.chain = chain
        st.session_state.retriever = retriever
        st.session_state.doc_processed = True
        os.unlink(tmp_path)
    st.success(f"✅ Document processed! {len(chunks)} chunks indexed.")

# Chat
if st.session_state.chain:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if question := st.chat_input("Ask anything about the document..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                answer, sources = query_document(
                    st.session_state.chain,
                    st.session_state.retriever,
                    question
                )
                st.write(answer)
                with st.expander("📖 Source Pages"):
                    for doc in sources:
                        st.caption(f"Page {doc.metadata.get('page', 'N/A')}: {doc.page_content[:200]}...")

        st.session_state.messages.append({"role": "assistant", "content": answer})

else:
    st.info("👈 Please upload a financial PDF document to get started")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Documents", "Any PDF", "Annual Reports")
    with col2:
        st.metric("Model", "LLaMA3-8B", "via Groq")
    with col3:
        st.metric("Speed", "< 2 sec", "Groq inference")