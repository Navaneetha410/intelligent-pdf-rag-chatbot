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

    /* ---------------- MAIN BACKGROUND ---------------- */

    .stApp {
        background: linear-gradient(
            135deg,
            #eef2ff 0%,
            #f5f3ff 50%,
            #eff6ff 100%
        );
    }


    /* ---------------- MAIN TITLE ---------------- */

    .main-title {
        font-size: 44px;
        font-weight: 800;
        text-align: center;
        color: #4f46e5;
        margin-top: 10px;
        margin-bottom: 5px;
    }


    /* ---------------- SUBTITLE ---------------- */

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #475569;
        margin-bottom: 30px;
    }


    /* ---------------- ANSWER CARD ---------------- */

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


    /* ---------------- SOURCE CARD ---------------- */

    .source-card {
        padding: 12px 16px;
        border-radius: 10px;
        background: #ffffff;
        border-left: 5px solid #8b5cf6;
        box-shadow: 0 3px 10px rgba(139, 92, 246, 0.10);
        margin-top: 8px;
    }


    /* ---------------- FILE UPLOADER ---------------- */

    [data-testid="stFileUploader"] {
        background: white;
        border-radius: 15px;
        padding: 10px;
        border: 2px dashed #818cf8;
    }


    /* ---------------- TEXT INPUT ---------------- */

    [data-testid="stTextInput"] input {
        border-radius: 12px;
        border: 2px solid #c4b5fd;
        padding: 12px;
    }


    /* ---------------- METRIC BOXES ---------------- */

    [data-testid="stMetric"] {
        background: white;
        padding: 15px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(79, 70, 229, 0.10);
    }


    /* ---------------- SIDEBAR ---------------- */

    section[data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #312e81,
            #4f46e5,
            #7c3aed
        );
    }


    /* ---------------- SIDEBAR TEXT ---------------- */

    section[data-testid="stSidebar"] * {
        color: white !important;
    }


    /* ---------------- SUCCESS / INFO BOX ---------------- */

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
# LOCAL LLM
# =========================================================

@st.cache_resource
def get_llm():

    from langchain_ollama import ChatOllama

    return ChatOllama(
        model="qwen2.5:0.5b",
        temperature=0,
        base_url="http://localhost:11434"
    )


# =========================================================
# CREATE VECTOR DATABASE
# =========================================================

@st.cache_resource
def create_vector_store(pdf_bytes):

    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS

    # Save uploaded PDF
    with open("uploaded.pdf", "wb") as f:
        f.write(pdf_bytes)

    # Load PDF
    loader = PyPDFLoader("uploaded.pdf")
    documents = loader.load()

    # Split PDF text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )

    chunks = text_splitter.split_documents(documents)

    # Create embeddings
    embeddings = get_embeddings()

    # Create FAISS vector database
    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    return vector_store, len(documents), len(chunks)


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
    st.write("🤖 **LLM:** Ollama")

    st.markdown("---")

    st.info(
        "Upload a PDF and ask questions based on its content."
    )


# =========================================================
# MAIN APPLICATION
# =========================================================

if uploaded_file:

    pdf_bytes = uploaded_file.getvalue()

    # Process PDF
    with st.spinner("🔄 Processing your PDF..."):

        vector_store, page_count, chunk_count = create_vector_store(
            pdf_bytes
        )

    st.success("✅ PDF processed successfully!")

    # =====================================================
    # PDF INFORMATION
    # =====================================================

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "📄 Pages",
            page_count
        )

    with col2:
        st.metric(
            "🧩 Text Chunks",
            chunk_count
        )

    with col3:
        st.metric(
            "🔎 Retrieved Chunks",
            2
        )

    st.markdown("---")


    # =====================================================
    # QUESTION AREA
    # =====================================================

    st.subheader("💬 Ask a Question")

    question = st.text_input(
        "Type your question below:",
        placeholder="Example: What is the main topic of this document?"
    )


    # =====================================================
    # QUESTION ANSWERING
    # =====================================================

    if question:

        with st.spinner(
            "🤖 Searching the PDF and generating answer..."
        ):

            # Retrieve relevant PDF chunks
            relevant_docs = vector_store.similarity_search(
                question,
                k=2
            )

            # Prepare context
            context = "\n\n".join(
                doc.page_content
                for doc in relevant_docs
            )


            # Prompt
            from langchain_core.prompts import ChatPromptTemplate

            prompt = ChatPromptTemplate.from_template(
                """
                Answer the question using only the information
                provided in the context below.

                If the answer is not in the context, say:

                "I could not find the answer in the PDF."

                Keep the answer simple and clear.

                Context:
                {context}

                Question:
                {question}
                """
            )


            # Load local LLM
            llm = get_llm()


            # Create chain
            chain = prompt | llm


            # Generate response
            response = chain.invoke(
                {
                    "context": context,
                    "question": question
                }
            )


        # =================================================
        # ANSWER
        # =================================================

        st.subheader("🤖 Answer")

        st.markdown(
            f"""
            <div class="answer-card">
            {response.content}
            </div>
            """,
            unsafe_allow_html=True
        )


        # =================================================
        # SOURCES
        # =================================================

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
        st.write("🦙 Ollama")