import os
import json
from dotenv import load_dotenv
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.callbacks.base import BaseCallbackHandler
import langchain

from src.langchain.parser import parse_and_load
from src.langchain.trace import interceptor
from src.evaluator.validation import ValidationEngine

load_dotenv()

def main():
    
    langchain.debug = True
    # Create synthetic documents
    docs = [
        Document(page_content="LangChain is a powerful tool for building language model applications.", metadata={"source": "doc1"}),
        Document(page_content="ChromaDB provides a fast vector store for similarity search.", metadata={"source": "doc2"}),
        Document(page_content="OpenAI offers state-of-the-art LLMs for various tasks.", metadata={"source": "doc3"})
    ]
    
    # Initialize the embeddings model (using OpenAI embeddings)
    embeddings = OpenAIEmbeddings()
    
    # Create a Chroma vector store from the documents
    vectorstore = Chroma.from_documents(docs, embeddings, collection_name="test_docs")
    
    # Initialize the OpenAI LLM
    llm = OpenAI(temperature=0, callbacks=[interceptor])
    
    # Create a retriever from the vector store
    retriever = vectorstore.as_retriever(callbacks=[interceptor])
    
    # Build the retrieveal QA chain with the callback
    qa_chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type="stuff",
        retriever = retriever,
        verbose=True,
        callbacks=[interceptor]
    )
    
    # Run a sample query against the chain
    for query in ["What does langchain do?", "What does ChromaDB provide?", "What does OPENAI do?"]:
        answer = qa_chain.run(query)
        print(f"Query: {query}")
        print(f"Answer: {answer}")
    
    # Save the logs to a json file
    json_data = interceptor.log()
    
    # Get the curated data using parse_and_load
    dataset = parse_and_load()
    
    # Initialize validation
    results, _, _ = ValidationEngine(dataset).evaluate()
    
    print(results)
    
if __name__ == "__main__":
    main()