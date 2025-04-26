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

    # Setup embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    # New: Batch and sleep to avoid rate limits
    batch_size = 10  # Embed 10 documents at a time
    embedded_vectors = []

    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        try:
            vectors = embeddings.embed_documents([doc.page_content for doc in batch])
            embedded_vectors.extend(vectors)
            time.sleep(2)  # Pause 2 seconds between batches
        except Exception as e:
            print(f"Embedding batch failed: {e}")
            time.sleep(10)  # Wait longer if OpenAI complains, then continue

    # Now store into FAISS
    vectordb = FAISS.from_texts(
        [doc.page_content for doc in docs],
        embedding=embeddings
    )
