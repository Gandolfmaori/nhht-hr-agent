import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

vectordb = None

def load_documents():
    global vectordb

    # Load documents
    loader = DirectoryLoader("docs", glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    # Use the correct embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    vectordb = FAISS.from_documents(docs, embedding=embeddings)
