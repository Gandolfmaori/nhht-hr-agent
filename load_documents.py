import os
from langchain.vectorstores import FAISS
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

vectordb = None

def load_documents():
    global vectordb

    # Load documents
    loader = DirectoryLoader("docs", glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    # Explicitly define the OpenAI Embedding model
    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002",  # Standard OpenAI embedding model
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    vectordb = FAISS.from_documents(docs, embedding=embeddings)
