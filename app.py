import streamlit as st
import PyPDF2
from docx import Document


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔬",
    layout="wide"
)


# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 18px;
    color: #666;
    margin-bottom: 25px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# SESSION STATE
# ==========================================

if "documents" not in st.session_state:
    st.session_state.documents = []


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

        return "Unable to read this TXT file. Please use UTF-8 encoding."

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

    # Remove unnecessary spaces
    lines = text.splitlines()

    cleaned_lines = []

    for line in lines:

        line = line.strip()

        if line:
            cleaned_lines.append(line)

    cleaned_text = "\n".join(cleaned_lines)

    return cleaned_text


# ==========================================
# TEXT CHUNKING
# ==========================================

def create_chunks(text, chunk_size=1000, overlap=200):

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk_words = words[start:end]

        chunk = " ".join(chunk_words)

        if chunk.strip():

            chunks.append(chunk)

        # Move forward while keeping overlap
        start += chunk_size - overlap

    return chunks


# ==========================================
# DOCUMENT STATISTICS
# ==========================================

def calculate_statistics(text):

    words = text.split()

    paragraphs = [
        paragraph
        for paragraph in text.split("\n")
        if paragraph.strip()
    ]

    word_count = len(words)

    character_count = len(text)

    paragraph_count = len(paragraphs)

    reading_time = max(
        1,
        round(word_count / 200)
    )

    return {
        "words": word_count,
        "characters": character_count,
        "paragraphs": paragraph_count,
        "reading_time": reading_time
    }


# ==========================================
# HEADER
# ==========================================

st.markdown(
    '<div class="main-title">🔬 AI Research Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Version 2 — Multi-Document Processing & Text Chunking'
    '</div>',
    unsafe_allow_html=True
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
        min_value=300,
        max_value=2000,
        value=1000,
        step=100
    )

    overlap = st.slider(
        "Chunk overlap",
        min_value=50,
        max_value=500,
        value=200,
        step=50
    )

    st.info(
        "Chunking divides large documents into smaller "
        "sections. This prepares them for semantic search "
        "and RAG."
    )


# ==========================================
# UPLOAD DOCUMENTS
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
        "📥 Process Documents",
        use_container_width=True
    ):

        st.session_state.documents = []

        progress = st.progress(0)

        total_files = len(uploaded_files)

        for index, file in enumerate(uploaded_files):

            # Extract text
            raw_text = extract_text(file)

            # Clean text
            cleaned_text = clean_text(raw_text)

            # Create chunks
            chunks = create_chunks(
                cleaned_text,
                chunk_size,
                overlap
            )

            # Calculate statistics
            statistics = calculate_statistics(
                cleaned_text
            )

            # Store document
            document = {

                "name": file.name,

                "type": file.type,

                "text": cleaned_text,

                "chunks": chunks,

                "statistics": statistics

            }

            st.session_state.documents.append(
                document
            )

            progress.progress(
                (index + 1) / total_files
            )

        st.success(
            f"Successfully processed "
            f"{total_files} document(s)."
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

    total_characters = sum(
        doc["statistics"]["characters"]
        for doc in documents
    )

    total_chunks = sum(
        len(doc["chunks"])
        for doc in documents
    )


    # ======================================
    # METRICS
    # ======================================

    col1, col2, col3, col4 = st.columns(4)

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

    with col4:

        st.metric(
            "🔤 Characters",
            f"{total_characters:,}"
        )


    # ======================================
    # DOCUMENTS
    # ======================================

    st.subheader("📑 Processed Documents")

    for index, document in enumerate(documents):

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


            # ==================================
            # TEXT PREVIEW
            # ==================================

            st.write("### 👀 Text Preview")

            preview = document["text"][:3000]

            st.text_area(
                "Extracted text",
                preview,
                height=200,
                key=f"text_{index}"
            )


            # ==================================
            # CHUNK PREVIEW
            # ==================================

            st.write("### 🧩 Chunk Preview")

            if document["chunks"]:

                selected_chunk = st.selectbox(
                    "Select a chunk",
                    range(
                        len(document["chunks"])
                    ),
                    format_func=lambda x:
                    f"Chunk {x + 1}",
                    key=f"chunk_select_{index}"
                )

                chunk_text = document["chunks"][
                    selected_chunk
                ]

                st.text_area(
                    "Chunk content",
                    chunk_text,
                    height=250,
                    key=f"chunk_text_{index}"
                )


# ==========================================
# CHUNK INFORMATION
# ==========================================

if st.session_state.documents:

    st.divider()

    st.header("🧠 How Chunking Works")

    st.write(
        "The application divides each document into "
        "smaller sections so that later we can search "
        "the most relevant sections instead of sending "
        "the entire document to the AI."
    )

    st.code("""
Document
    ↓
Text Extraction
    ↓
Text Cleaning
    ↓
Text Chunking
    ↓
Chunk 1
Chunk 2
Chunk 3
Chunk 4
    ↓
Embeddings
    ↓
FAISS Vector Search
    ↓
RAG
    ↓
Gemini
    """, language="text")


else:

    st.info(
        "👆 Upload documents above to begin."
    )


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "AI Research Assistant • Version 2 • "
    "Document Chunking"
)
