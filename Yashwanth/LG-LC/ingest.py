from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from dotenv import load_dotenv

import os
import glob

load_dotenv()

all_docs = []

# Load all PDFs
pdf_files = glob.glob("pdfs/*.pdf")

for pdf in pdf_files:

    print(f"Loading: {pdf}")

    loader = PyPDFLoader(pdf)

    documents = loader.load()

    # Add metadata
    for doc in documents:
        doc.metadata["source"] = pdf

    all_docs.extend(documents)

# Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

docs = splitter.split_documents(all_docs)

print(f"Total chunks: {len(docs)}")

# Embeddings
embeddings = OpenAIEmbeddings()

# Store in Qdrant
QdrantVectorStore.from_documents(
    docs,
    embeddings,
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    collection_name="pdf_docs",
)

print("All PDFs stored successfully!")