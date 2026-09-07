import streamlit as st


# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="Intelligent PDF Q&A Chatbot",
    page_icon="📚",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .stApp {
        background: linear-gradient(
            135deg,
            #eef2ff 0%,
            #f5f3ff 50%,
            #eff6ff 100%
        );
    }

    .main-title {
        font-size: 44px;
        font-weight: 800;
        text-align: center;
        color: #4f46e5;
        margin-top: 10px;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #475569;
        margin-bottom: 30px;
    }

    .answer-card {
        padding: 22px;
        border-radius: 15px;
        background: linear-gradient(
            135deg,
            #eef2ff,
            #f5f3ff
        );
        border-left: 6px solid #6366f1;
        box-shadow: 0 5px 18px rgba(79, 70, 229, 0.12);
        margin-top: 15px;
        font-size: 17px;
        line-height: 1.6;
    }

    .source-card {
        padding: 12px 16px;
        border-radius: 10px;
        background: #ffffff;
        border-left: 5px solid #8b5cf6;
        box-shadow: 0 3px 10px rgba(139, 92, 246, 0.10);
        margin-top: 8px;
    }

    [data-testid="stFileUploader"] {
        background: white;
        border-radius: 15px;
        padding: 10px;
        border: 2px dashed #818cf8;
    }

    [data-testid="stTextInput"] input {
        border-radius: 12px;
        border: 2px solid #c4b5fd;
        padding: 12px;
    }

    [data-testid="stMetric"] {
        background: white;
        padding: 15px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(79, 70, 229, 0.10);
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #312e81,
            #4f46e5,
            #7c3aed
        );
    }

    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    [data-testid="stAlert"] {
        border-radius: 12px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# TITLE
# =========================================================

st.markdown(
    '<div class="main-title">📚 Intelligent PDF Q&A Chatbot</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Ask questions from your PDF using Retrieval-Augmented Generation (RAG)</div>',
    unsafe_allow_html=True
)


# =========================================================
# EMBEDDING MODEL
# =========================================================

@st.cache_resource
def get_embeddings():

    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


# =========================================================
# CREATE VECTOR DATABASE
# =========================================================

@st.cache_resource
def create_vector_store(pdf_bytes):

    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS

    with open("uploaded.pdf", "wb") as f:
        f.write(pdf_bytes)

    loader = PyPDFLoader("uploaded.pdf")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )

    chunks = text_splitter.split_documents(documents)

    embeddings = get_embeddings()

    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    return vector_store, len(documents), len(chunks)


# =========================================================
# GEMINI
# =========================================================

@st.cache_resource
def get_gemini_client():

    from google import genai

    api_key = st.secrets["GEMINI_API_KEY"]

    return genai.Client(
        api_key=api_key
    )


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("📄 Upload PDF")

    uploaded_file = st.file_uploader(
        "Choose your PDF",
        type=["pdf"]
    )

    st.markdown("---")

    st.subheader("⚙️ Project Details")

    st.write("🐍 **Language:** Python")
    st.write("🔗 **Framework:** LangChain")
    st.write("🔎 **Method:** RAG")
    st.write("🧠 **Embeddings:** Hugging Face")
    st.write("🗂️ **Vector DB:** FAISS")
    st.write("🤖 **LLM:** Gemini")

    st.markdown("---")

    st.info(
        "Upload a PDF and ask questions based on its content."
    )


# =========================================================
# MAIN APPLICATION
# =========================================================

if uploaded_file:

    pdf_bytes = uploaded_file.getvalue()

    with st.spinner("🔄 Processing your PDF..."):

        vector_store, page_count, chunk_count = create_vector_store(
            pdf_bytes
        )

    st.success("✅ PDF processed successfully!")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("📄 Pages", page_count)

    with col2:
        st.metric("🧩 Text Chunks", chunk_count)

    with col3:
        st.metric("🔎 Retrieved Chunks", 2)

    st.markdown("---")

    st.subheader("💬 Ask a Question")

    question = st.text_input(
        "Type your question below:",
        placeholder="Example: What is the main topic of this document?"
    )

    if question:

        with st.spinner(
            "🤖 Searching the PDF and generating answer..."
        ):

            relevant_docs = vector_store.similarity_search(
                question,
                k=2
            )

            context = "\n\n".join(
                doc.page_content
                for doc in relevant_docs
            )

            prompt = f"""
You are a PDF question-answering assistant.

Answer the user's question using ONLY the information
provided in the context below.

If the answer is not present in the context, say:
"I could not find the answer in the PDF."

Keep the answer simple and clear.

Context:
{context}

Question:
{question}
"""

            client = get_gemini_client()

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )

        st.subheader("🤖 Answer")

        st.markdown(
            f"""
            <div class="answer-card">
            {response.text}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.subheader("📚 Sources")

        shown_pages = set()

        for doc in relevant_docs:

            page_number = doc.metadata.get(
                "page",
                "Unknown"
            )

            if isinstance(page_number, int):
                page_number += 1

            if page_number not in shown_pages:

                st.markdown(
                    f"""
                    <div class="source-card">
                    📄 Source: Page {page_number}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                shown_pages.add(page_number)


# =========================================================
# WELCOME SCREEN
# =========================================================

else:

    st.info(
        "👈 Upload a PDF from the sidebar to start asking questions."
    )

    st.markdown("### 🔄 How It Works")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("### 1️⃣")
        st.write("📄 Upload PDF")

    with col2:
        st.markdown("### 2️⃣")
        st.write("🧠 Create Embeddings")

    with col3:
        st.markdown("### 3️⃣")
        st.write("🔎 Retrieve Information")

    with col4:
        st.markdown("### 4️⃣")
        st.write("🤖 Generate Answer")

    st.markdown("---")

    st.markdown("### 🚀 Technologies Used")

    tech1, tech2, tech3, tech4, tech5 = st.columns(5)

    with tech1:
        st.write("🐍 Python")

    with tech2:
        st.write("🔗 LangChain")

    with tech3:
        st.write("🗂️ FAISS")

    with tech4:
        st.write("🤗 Hugging Face")

    with tech5:
        st.write("✨ Gemini")