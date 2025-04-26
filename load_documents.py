import os
import time
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import RateLimitError

vectordb = None

def load_documents():
    global vectordb

    loader = DirectoryLoader("docs", glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    texts = [doc.page_content for doc in docs]

    embeddings_model = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    all_embeddings = []

    batch_size = 1  # Only 1 text at a time!
    i = 0
    while i < len(texts):
        batch = texts[i:i + batch_size]
        try:
            embeddings = embeddings_model.embed_documents(batch)
            all_embeddings.extend(embeddings)
            i += batch_size
            time.sleep(1)  # tiny pause after success
        except RateLimitError:
            print("Rate limit hit! Sleeping 10 seconds before retrying...")
            time.sleep(10)  # faster retry
        except Exception as e:
            print(f"Embedding batch failed: {e}")
            time.sleep(5)  # fast recovery

    vectordb = FAISS.from_embeddings(all_embeddings, texts)
