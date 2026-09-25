import streamlit as st
import PyPDF2
from docx import Document
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔬",
    layout="wide"
)


# ==========================================
# SESSION STATE
# ==========================================

if "documents" not in st.session_state:
    st.session_state.documents = []

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "index" not in st.session_state:
    st.session_state.index = None

if "search_results" not in st.session_state:
    st.session_state.search_results = []


# ==========================================
# LOAD EMBEDDING MODEL
# ==========================================

@st.cache_resource
def load_embedding_model():

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    return model


# ==========================================
# DOCUMENT EXTRACTION
# ==========================================

def extract_pdf(file):

    text = ""

    try:

        pdf_reader = PyPDF2.PdfReader(file)

        for page in pdf_reader.pages:

            page_text = page.extract_text()

            if page_text:

                text += page_text + "\n"

    except Exception as e:

        return f"Error reading PDF: {e}"

    return text


def extract_docx(file):

    try:

        document = Document(file)

        text = []

        for paragraph in document.paragraphs:

            if paragraph.text.strip():

                text.append(paragraph.text)

        return "\n".join(text)

    except Exception as e:

        return f"Error reading DOCX: {e}"


def extract_txt(file):

    try:

        content = file.read()

        return content.decode("utf-8")

    except UnicodeDecodeError:

        return (
            "Unable to read this TXT file. "
            "Please use UTF-8 encoding."
        )

    except Exception as e:

        return f"Error reading TXT: {e}"


def extract_text(file):

    filename = file.name.lower()

    if filename.endswith(".pdf"):

        return extract_pdf(file)

    elif filename.endswith(".docx"):

        return extract_docx(file)

    elif filename.endswith(".txt"):

        return extract_txt(file)

    return "Unsupported file type."


# ==========================================
# TEXT CLEANING
# ==========================================

def clean_text(text):

    lines = text.splitlines()

    cleaned_lines = []

    for line in lines:

        line = line.strip()

        if line:

            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


# ==========================================
# TEXT CHUNKING
# ==========================================

def create_chunks(
    text,
    chunk_size=500,
    overlap=100
):

    words = text.split()

    chunks = []

    start = 0

    step = chunk_size - overlap

    while start < len(words):

        end = start + chunk_size

        chunk_words = words[start:end]

        chunk = " ".join(chunk_words)

        if chunk.strip():

            chunks.append(chunk)

        start += step

    return chunks


# ==========================================
# DOCUMENT STATISTICS
# ==========================================

def calculate_statistics(text):

    words = text.split()

    paragraphs = [
        p
        for p in text.split("\n")
        if p.strip()
    ]

    return {
        "words": len(words),
        "characters": len(text),
        "paragraphs": len(paragraphs),
        "reading_time": max(
            1,
            round(len(words) / 200)
        )
    }


# ==========================================
# CREATE EMBEDDINGS
# ==========================================

def create_embeddings(chunks, model):

    embeddings = model.encode(
        chunks,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    embeddings = embeddings.astype("float32")

    return embeddings


# ==========================================
# CREATE FAISS INDEX
# ==========================================

def create_faiss_index(embeddings):

    dimension = embeddings.shape[1]

    # Inner Product on normalized vectors
    # is equivalent to cosine similarity

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(embeddings)

    return index


# ==========================================
# SEMANTIC SEARCH
# ==========================================

def semantic_search(
    query,
    model,
    index,
    chunks,
    top_k=5
):

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    query_embedding = query_embedding.astype(
        "float32"
    )

    similarities, indices = index.search(
        query_embedding,
        min(top_k, len(chunks))
    )

    results = []

    for similarity, index_number in zip(
        similarities[0],
        indices[0]
    ):

        if index_number == -1:

            continue

        results.append({
            "chunk": chunks[index_number],
            "similarity": float(similarity),
            "chunk_number": index_number + 1
        })

    return results


# ==========================================
# HEADER
# ==========================================

st.markdown(
    """
    <h1>🔬 AI Research Assistant</h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    ### Version 3 — Embeddings + FAISS Semantic Search

    Upload research documents and search them using
    meaning instead of exact keywords.
    """
)


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.header("⚙️ Settings")

    st.subheader("📄 Supported Files")

    st.write("PDF")
    st.write("DOCX")
    st.write("TXT")

    st.divider()

    st.subheader("🧩 Chunk Settings")

    chunk_size = st.slider(
        "Words per chunk",
        300,
        2000,
        500,
        100
    )

    overlap = st.slider(
        "Chunk overlap",
        50,
        500,
        100,
        50
    )

    st.divider()

    st.subheader("🔎 Search Settings")

    top_k = st.slider(
        "Results to retrieve",
        1,
        10,
        5
    )

    st.info(
        "Version 3 uses Sentence Transformers "
        "to create embeddings and FAISS to "
        "perform semantic search."
    )


# ==========================================
# UPLOAD
# ==========================================

st.header("📚 Upload Research Documents")

uploaded_files = st.file_uploader(
    "Select one or more documents",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)


# ==========================================
# PROCESS DOCUMENTS
# ==========================================

if uploaded_files:

    if st.button(
        "🚀 Process Documents",
        use_container_width=True
    ):

        # Reset previous data

        st.session_state.documents = []
        st.session_state.chunks = []
        st.session_state.index = None
        st.session_state.search_results = []

        progress = st.progress(0)

        all_chunks = []

        total_files = len(uploaded_files)

        for file_number, file in enumerate(
            uploaded_files
        ):

            # ----------------------------------
            # Extract text
            # ----------------------------------

            raw_text = extract_text(file)

            # ----------------------------------
            # Clean text
            # ----------------------------------

            cleaned_text = clean_text(
                raw_text
            )

            # ----------------------------------
            # Create chunks
            # ----------------------------------

            document_chunks = create_chunks(
                cleaned_text,
                chunk_size,
                overlap
            )

            # ----------------------------------
            # Statistics
            # ----------------------------------

            statistics = calculate_statistics(
                cleaned_text
            )

            # ----------------------------------
            # Store document
            # ----------------------------------

            document = {
                "name": file.name,
                "type": file.type,
                "text": cleaned_text,
                "chunks": document_chunks,
                "statistics": statistics
            }

            st.session_state.documents.append(
                document
            )

            # ----------------------------------
            # Add chunks
            # ----------------------------------

            for chunk_number, chunk in enumerate(
                document_chunks
            ):

                all_chunks.append({
                    "text": chunk,
                    "document": file.name,
                    "chunk_number": chunk_number + 1
                })

            progress.progress(
                (file_number + 1) / total_files
            )

        # ======================================
        # CREATE EMBEDDINGS
        # ======================================

        if all_chunks:

            with st.spinner(
                "🧠 Creating embeddings..."
            ):

                model = load_embedding_model()

                chunk_texts = [
                    item["text"]
                    for item in all_chunks
                ]

                embeddings = create_embeddings(
                    chunk_texts,
                    model
                )

            # ==================================
            # CREATE FAISS DATABASE
            # ==================================

            with st.spinner(
                "🔎 Building FAISS vector database..."
            ):

                index = create_faiss_index(
                    embeddings
                )

                st.session_state.chunks = (
                    all_chunks
                )

                st.session_state.index = index

            st.success(
                f"Successfully processed "
                f"{total_files} document(s) and "
                f"created {len(all_chunks)} chunks."
            )

        else:

            st.warning(
                "No readable text was found "
                "in the uploaded documents."
            )


# ==========================================
# DASHBOARD
# ==========================================

if st.session_state.documents:

    st.divider()

    st.header("📊 Research Dashboard")

    documents = st.session_state.documents

    total_documents = len(documents)

    total_words = sum(
        doc["statistics"]["words"]
        for doc in documents
    )

    total_chunks = len(
        st.session_state.chunks
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "📚 Documents",
            total_documents
        )

    with col2:

        st.metric(
            "📝 Total Words",
            f"{total_words:,}"
        )

    with col3:

        st.metric(
            "🧩 Total Chunks",
            total_chunks
        )

    # ======================================
    # DOCUMENT DETAILS
    # ======================================

    st.subheader(
        "📑 Processed Documents"
    )

    for document in documents:

        stats = document["statistics"]

        with st.expander(
            f"📄 {document['name']}"
        ):

            col1, col2, col3, col4 = st.columns(4)

            with col1:

                st.write(
                    f"**Words:** "
                    f"{stats['words']:,}"
                )

            with col2:

                st.write(
                    f"**Paragraphs:** "
                    f"{stats['paragraphs']:,}"
                )

            with col3:

                st.write(
                    f"**Chunks:** "
                    f"{len(document['chunks']):,}"
                )

            with col4:

                st.write(
                    f"**Reading:** "
                    f"{stats['reading_time']} min"
                )


# ==========================================
# SEMANTIC SEARCH
# ==========================================

if st.session_state.index is not None:

    st.divider()

    st.header("🔎 Semantic Document Search")

    st.write(
        "Ask a question about your documents. "
        "The system searches by meaning rather than "
        "only matching exact words."
    )

    query = st.text_input(
        "Enter your research question",
        placeholder=(
            "Example: What are the main applications "
            "of artificial intelligence?"
        )
    )

    if st.button(
        "🔍 Search Documents",
        use_container_width=True
    ):

        if not query.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            with st.spinner(
                "🔎 Searching relevant sections..."
            ):

                model = load_embedding_model()

                results = semantic_search(
                    query,
                    model,
                    st.session_state.index,
                    st.session_state.chunks,
                    top_k
                )

                st.session_state.search_results = (
                    results
                )


# ==========================================
# SEARCH RESULTS
# ==========================================

if st.session_state.search_results:

    st.divider()

    st.header("📌 Relevant Document Sections")

    for number, result in enumerate(
        st.session_state.search_results,
        start=1
    ):

        st.subheader(
            f"Result {number}"
        )

        similarity = result["similarity"]

        relevance = similarity * 100

        st.caption(
            f"📄 Document: {result['chunk']['document']} | "
            f"🧩 Chunk: {result['chunk']['chunk_number']} | "
            f"🎯 Relevance: {relevance:.1f}%"
        )

        st.write(
            result["chunk"]["text"]
        )

        st.divider()


# ==========================================
# HOW IT WORKS
# ==========================================

st.divider()

st.header("🧠 How Version 3 Works")

st.code(
    """
Documents
    ↓
Text Extraction
    ↓
Text Cleaning
    ↓
Text Chunking
    ↓
Sentence Transformer
    ↓
Embeddings
    ↓
FAISS Vector Database
    ↓
User Question
    ↓
Question Embedding
    ↓
Semantic Search
    ↓
Relevant Chunks
""",
    language="text"
)


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "AI Research Assistant • Version 3 • "
    "Embeddings + FAISS"
)
