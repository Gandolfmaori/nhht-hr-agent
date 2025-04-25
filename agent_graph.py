from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from load_documents import vectordb

def run_agent(user_query):
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
        retriever=vectordb.as_retriever()
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        You are an HR advisor for Ngāti Hine Health Trust. Answer the following question, incorporating NHHT policies, MBIE employment law, and cultural values when appropriate.

        Context: {context}

        Question: {question}

        Answer:
        """
    )

    result = qa_chain.run(user_query)
    return result
