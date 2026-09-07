import streamlit as st


st.set_page_config(
    page_title="Intelligent PDF Q&A Chatbot",
    page_icon="📄"
)

st.title("📄 Intelligent PDF Q&A Chatbot")
st.write("Upload a PDF and ask questions about it.")


# Load embedding model only when needed
@st.cache_resource
def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


# Load LLM only when needed
@st.cache_resource
def get_llm():
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model="qwen2.5:0.5b",
        temperature=0,
        base_url="http://localhost:11434"
    )


# Create vector database
@st.cache_resource
def create_vector_store(pdf_bytes):

    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS

    # Save PDF
    with open("uploaded.pdf", "wb") as f:
        f.write(pdf_bytes)

    # Load PDF
    loader = PyPDFLoader("uploaded.pdf")
    documents = loader.load()

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )

    chunks = text_splitter.split_documents(documents)

    # Create embeddings
    embeddings = get_embeddings()

    # Create FAISS database
    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    return vector_store, len(documents), len(chunks)


uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)


if uploaded_file:

    pdf_bytes = uploaded_file.getvalue()

    with st.spinner("Processing PDF..."):
        vector_store, page_count, chunk_count = create_vector_store(
            pdf_bytes
        )

    st.success(
        f"PDF loaded successfully! Pages: {page_count}"
    )

    st.write(
        f"Text chunks created: {chunk_count}"
    )

    st.success("PDF is ready for questions! ✅")

    question = st.text_input(
        "Ask a question about your PDF:"
    )

    if question:

        with st.spinner("Finding answer..."):

            # Find relevant content
            relevant_docs = vector_store.similarity_search(
                question,
                k=2
            )

            # Prepare context
            context = "\n\n".join(
                doc.page_content
                for doc in relevant_docs
            )

            # Import prompt only when needed
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

            llm = get_llm()

            chain = prompt | llm

            response = chain.invoke(
                {
                    "context": context,
                    "question": question
                }
            )

        st.subheader("Answer")
        st.write(response.content)

        st.subheader("Sources")

        shown_pages = set()

        for doc in relevant_docs:

            page_number = doc.metadata.get(
                "page",
                "Unknown"
            )

            if isinstance(page_number, int):
                page_number += 1

            if page_number not in shown_pages:
                st.write(f"Page {page_number}")
                shown_pages.add(page_number)