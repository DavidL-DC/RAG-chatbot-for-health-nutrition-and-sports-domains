from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import os
from dotenv import load_dotenv

load_dotenv()

DATA_PATH = "data/raw"
DB_PATH = "vectorstore/chroma"

documents = []

for root, _, files in os.walk(DATA_PATH):
    for file in files:
        if file.endswith(".pdf"):
            full_path = os.path.join(root, file)
            topic = os.path.relpath(root, DATA_PATH)
            loader = PyPDFLoader(full_path)
            docs = loader.load()
            for d in docs:
                d.metadata["topic"] = topic
            documents.extend(docs)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

chunks = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", chunk_size=50)
db = Chroma.from_documents(chunks, embeddings, persist_directory=DB_PATH)

print("Ingestion abgeschlossen.")
