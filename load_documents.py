import os
import time
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

    texts = [doc.page_content for doc in docs]

    # Setup embeddings
    embeddings_model = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    # Embed documents manually, in batches
    batch_size = 10
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        embeddings = embeddings_model.embed_documents(batch)
        all_embeddings.extend(embeddings)
        time.sleep(1)  # Sleep to avoid OpenAI rate limits

    # Build FAISS index manually (NO re-embedding now!)
    vectordb = FAISS.from_embeddings(all_embeddings, texts)
