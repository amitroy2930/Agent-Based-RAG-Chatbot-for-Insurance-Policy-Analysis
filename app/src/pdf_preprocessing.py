import os
import re
from typing import List

from dotenv import load_dotenv
from llama_parse import LlamaParse

from langchain_core.documents import Document
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import Chroma


# =========================
# 1) PDF -> Markdown Parser
# =========================

class PdfToMarkdownConverter:
    """
    Responsibility:
    - Convert PDF files into Markdown files using LlamaParse.
    """
    def __init__(self, llama_api_key: str, result_type: str = "markdown"):
        self.parser = LlamaParse(api_key=llama_api_key, result_type=result_type)

    def convert_folder(self, pdf_folder: str, md_folder: str) -> List[str]:
        """
        Converts all PDFs from pdf_folder into md_folder.
        Returns a list of markdown file paths created.
        """
        os.makedirs(md_folder, exist_ok=True)
        created_files = []

        for file_name in os.listdir(pdf_folder):
            if not file_name.lower().endswith(".pdf"):
                continue

            pdf_path = os.path.join(pdf_folder, file_name)
            md_path = os.path.join(md_folder, file_name.replace(".pdf", "-restructured.md"))

            documents = self.parser.load_data(pdf_path)

            with open(md_path, "w", encoding="utf-8") as f:
                for doc in documents:
                    f.write(doc.text + "\n\n")

            created_files.append(md_path)

        return created_files


# =========================
# 2) Markdown Loader + Metadata Enricher
# =========================

class MarkdownDocumentLoader:
    """
    Responsibility:
    - Load markdown files into LangChain Document objects
    - Enrich metadata based on filename pattern
    """
    # Example pattern you described:
    # policy-<company>-plan-<plan_number>-restructured.md
    FILENAME_PATTERN = re.compile(r"policy-([a-zA-Z_]+)-plan-([a-zA-Z_]+)", re.IGNORECASE)

    def load_folder(self, md_folder: str) -> List[Document]:
        all_docs: List[Document] = []

        for file_name in os.listdir(md_folder):
            if not file_name.lower().endswith(".md"):
                continue

            md_path = os.path.join(md_folder, file_name)
            docs = UnstructuredMarkdownLoader(md_path).load()

            for doc in docs:
                self._enrich_metadata(doc, file_name)

            all_docs.extend(docs)

        return all_docs

    def _enrich_metadata(self, doc: Document, file_name: str) -> None:
        match = self.FILENAME_PATTERN.search(file_name)
        if match:
            doc.metadata["company"] = match.group(1).upper()
            doc.metadata["plan"] = match.group(2)
        else:
            doc.metadata["source_file"] = file_name

        # Always add a consistent source
        doc.metadata["source"] = file_name


# =========================
# 3) Chunker
# =========================

class DocumentChunker:
    """
    Responsibility:
    - Split documents into smaller chunks for embedding.
    """
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def chunk(self, documents: List[Document]) -> List[Document]:
        return self.splitter.split_documents(documents)


# =========================
# 4) Vector Store Writer
# =========================

class ChromaVectorStoreWriter:
    """
    Responsibility:
    - Store chunked documents into ChromaDB (persistent).
    """
    def __init__(self, persist_directory: str, collection_name: str):
        self.persist_directory = persist_directory
        self.collection_name = collection_name

    def write(self, docs: List[Document], embedding_model: AzureOpenAIEmbeddings) -> Chroma:
        vector_store = Chroma.from_documents(
            documents=docs,
            embedding=embedding_model,
            persist_directory=self.persist_directory,
            collection_name=self.collection_name,
        )
        vector_store.persist()
        return vector_store


# =========================
# 5) Main
# =========================

def main():
    load_dotenv()

    # Read configuration directly from .env (fallback to defaults)
    pdf_folder = os.getenv("PDF_FOLDER", "./app/data/input_data/pdf/")
    md_folder = os.getenv("MD_FOLDER", "./app/data/input_data/markdown_test/")
    chroma_dir = os.getenv("CHROMA_DIR", "./app/data/vector_data/chroma_db")
    chroma_collection = os.getenv("CHROMA_COLLECTION", "policy_docs")

    llama_result_type = os.getenv("LLAMA_RESULT_TYPE", "markdown")
    chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))
    azure_embedding_deployment = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
    azure_api_version = os.getenv("AZURE_API_VERSION", "2024-02-01")

    llama_cloud_api_key = os.getenv("LLAMA_CLOUD_API_KEY")
    azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

    missing = []
    if not llama_cloud_api_key:
        missing.append("LLAMA_CLOUD_API_KEY")
    if not azure_openai_api_key:
        missing.append("AZURE_OPENAI_API_KEY")
    if not azure_openai_endpoint:
        missing.append("AZURE_OPENAI_ENDPOINT")
    if missing:
        raise RuntimeError(f"Missing env vars: {', '.join(missing)}")

    converter = PdfToMarkdownConverter(llama_cloud_api_key, result_type=llama_result_type)
    loader = MarkdownDocumentLoader()
    chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    store_writer = ChromaVectorStoreWriter(chroma_dir, chroma_collection)

    # convert_pdfs=True means: parse PDFs first, then ingest markdown.
    # If you already have markdown and want to skip parsing: set convert_pdfs=False
    convert_pdfs = False

    # 1) Convert PDFs to Markdown
    if convert_pdfs:
        created_md_files = converter.convert_folder(pdf_folder, md_folder)
        print(f"[OK] Converted PDFs -> Markdown: {len(created_md_files)} files")

    # 2) Load markdown documents
    documents = loader.load_folder(md_folder)
    print(f"[OK] Loaded Markdown documents: {len(documents)} docs")

    if not documents:
        print("[WARN] No markdown documents found. Nothing to embed.")
        return

    # 3) Chunk documents
    chunks = chunker.chunk(documents)
    print(f"[OK] Chunked documents: {len(chunks)} chunks")

    # 4) Create embedding model
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=azure_embedding_deployment,
        openai_api_version=azure_api_version,
        azure_endpoint=azure_openai_endpoint,
        api_key=azure_openai_api_key,
    )
    print("[OK] Embedding model initialized")

    # 5) Write to Chroma
    store_writer.write(chunks, embeddings)
    print(f"[OK] Stored embeddings in Chroma: {chroma_dir} (collection={chroma_collection})")


if __name__ == "__main__":
    main()