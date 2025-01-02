"""
RAG System Generator

Usage:
  generate_rag_system.py <docs_dir>
  generate_rag_system.py (-h | --help)

Arguments:
  <docs_dir>  Path to the directory containing documentation text files.

Options:
  -h --help   Show this help message.
"""

import os
from docopt import docopt
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
#from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA


def load_documents(docs_dir):
    """Load documents from the specified directory."""
    documents = []
    for root, _, files in os.walk(docs_dir):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                loader = TextLoader(file_path)
                documents.extend(loader.load())
    return documents


def create_rag_system(docs_dir):
    """Generate a RAG system from the documentation files."""
    # Step 1: Load documents
    print(f"Loading documents from {docs_dir}...")
    documents = load_documents(docs_dir)
    print(f"Loaded {len(documents)} documents.")

    # Step 2: Create a vector database
    print("Creating vector database...")
    embeddings = OpenAIEmbeddings()  # Updated import and initialization
    vector_store = FAISS.from_documents(documents, embeddings)

    # Step 3: Define retriever and LLM
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    llm = OpenAI(model="text-davinci-003")  # Replace with your preferred LLM

    # Step 4: Create the RetrievalQA chain
    rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

    return rag_chain

def main(docs_dir):
    # Create the RAG system
    rag_system = create_rag_system(docs_dir)

    # Interact with the RAG system
    print("RAG system is ready! Type your questions below (type 'exit' to quit).")
    while True:
        query = input("Question: ")
        if query.lower() == "exit":
            print("Exiting RAG system. Goodbye!")
            break
        response = rag_system.run(query)
        print(f"Answer: {response}")


if __name__ == "__main__":
    # Parse arguments using docopt
    args = docopt(__doc__)
    docs_dir = args["<docs_dir>"]

    # Run the main function
    main(docs_dir)
