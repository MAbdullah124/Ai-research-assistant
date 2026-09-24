import streamlit as st
import PyPDF2
from docx import Document
import io


# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔬",
    layout="wide"
)


# -----------------------------
# Custom Styling
# -----------------------------

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

.document-card {
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #ddd;
    margin-bottom: 15px;
}

</style>
""", unsafe_allow_html=True)


# -----------------------------
# Session State
# -----------------------------

if "documents" not in st.session_state:
    st.session_state.documents = []


# -----------------------------
# Document Extraction Functions
# -----------------------------

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

    file_type = file.name.lower()

    if file_type.endswith(".pdf"):

        return extract_pdf(file)

    elif file_type.endswith(".docx"):

        return extract_docx(file)

    elif file_type.endswith(".txt"):

        return extract_txt(file)

    else:

        return "Unsupported file type."


# -----------------------------
# Statistics
# -----------------------------

def calculate_statistics(text):

    words = text.split()

    paragraphs = [
        p for p in text.split("\n")
        if p.strip()
    ]

    word_count = len(words)

    character_count = len(text)

    paragraph_count = len(paragraphs)

    # Average reading speed
    reading_time = max(1, round(word_count / 200))

    return {
        "words": word_count,
        "characters": character_count,
        "paragraphs": paragraph_count,
        "reading_time": reading_time
    }


# -----------------------------
# Header
# -----------------------------

st.markdown(
    '<div class="main-title">🔬 AI Research Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Upload multiple research documents and prepare them for AI-powered analysis.'
    '</div>',
    unsafe_allow_html=True
)


# -----------------------------
# Sidebar
# -----------------------------

with st.sidebar:

    st.header("⚙️ Settings")

    st.write("### Supported Files")

    st.write("📄 PDF")
    st.write("📝 DOCX")
    st.write("📃 TXT")

    st.divider()

    st.info(
        "Version 1 focuses on document upload and text extraction. "
        "RAG and semantic search will be added in later versions."
    )


# -----------------------------
# Upload Section
# -----------------------------

st.header("📚 Upload Research Documents")

uploaded_files = st.file_uploader(
    "Select one or more documents",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)


# -----------------------------
# Process Documents
# -----------------------------

if uploaded_files:

    if st.button("📥 Process Documents", use_container_width=True):

        st.session_state.documents = []

        progress = st.progress(0)

        total_files = len(uploaded_files)

        for index, file in enumerate(uploaded_files):

            text = extract_text(file)

            statistics = calculate_statistics(text)

            document = {
                "name": file.name,
                "type": file.type,
                "text": text,
                "statistics": statistics
            }

            st.session_state.documents.append(document)

            progress.progress((index + 1) / total_files)

        st.success(
            f"Successfully processed {total_files} document(s)."
        )


# -----------------------------
# Dashboard
# -----------------------------

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

    total_paragraphs = sum(
        doc["statistics"]["paragraphs"]
        for doc in documents
    )


    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Documents",
            total_documents
        )

    with col2:
        st.metric(
            "Total Words",
            f"{total_words:,}"
        )

    with col3:
        st.metric(
            "Characters",
            f"{total_characters:,}"
        )

    with col4:
        st.metric(
            "Paragraphs",
            f"{total_paragraphs:,}"
        )


    # -----------------------------
    # Document List
    # -----------------------------

    st.subheader("📑 Uploaded Documents")

    for index, document in enumerate(documents):

        stats = document["statistics"]

        with st.expander(
            f"📄 {document['name']}"
        ):

            col1, col2, col3 = st.columns(3)

            with col1:
                st.write(
                    f"**Words:** {stats['words']:,}"
                )

            with col2:
                st.write(
                    f"**Paragraphs:** {stats['paragraphs']:,}"
                )

            with col3:
                st.write(
                    f"**Reading Time:** "
                    f"{stats['reading_time']} min"
                )


            st.write("### 👀 Text Preview")

            preview = document["text"][:3000]

            if preview.strip():

                st.text_area(
                    "Extracted text",
                    preview,
                    height=250,
                    key=f"preview_{index}"
                )

            else:

                st.warning(
                    "No readable text was extracted from this document."
                )


    # -----------------------------
    # Combined Text
    # -----------------------------

    st.divider()

    st.subheader("📚 Combined Research Text")

    combined_text = "\n\n".join(
        f"===== {doc['name']} =====\n\n{doc['text']}"
        for doc in documents
    )

    st.text_area(
        "All documents",
        combined_text[:10000],
        height=300
    )


else:

    st.info(
        "👆 Upload your research documents above to begin."
    )


# -----------------------------
# Footer
# -----------------------------

st.divider()

st.caption(
    "AI Research Assistant • Version 1 • "
    "Multi-Document Processing"
)
