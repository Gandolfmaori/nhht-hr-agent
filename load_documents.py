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

    all_embeddings = []

    batch_size = 5  # Reduce batch size to 5
    i = 0
    while i < len(texts):
        batch = texts[i:i + batch_size]
        try:
            embeddings = embeddings_model.embed_documents(batch)
            all_embeddings.extend(embeddings)
            i += batch_size  # Move to next batch
            time.sleep(2)  # Sleep after each successful batch
        except RateLimitError:
            print("Rate limit hit! Sleeping 20 seconds before retrying...")
            time.sleep(20)  # Exponential backoff
        except Exception as e:
            print(f"Embedding batch failed: {e}")
            time.sleep(10)  # Generic error pause

    vectordb = FAISS.from_embeddings(all_embeddings, texts)
