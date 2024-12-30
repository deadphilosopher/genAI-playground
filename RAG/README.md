Step-by-Step Tutorial: Building a RAG System with HTML Files

Step 1: Set Up the Environment

```bash
pip install beautifulsoup4 langchain transformers faiss sentence-transformers
```

Step 2: Process HTML Files

1. Load and Parse HTML Files using BeautifulSoup to extract text content from HTML files.

2. Extract only the meaningful text, ignoring scripts, styles, and other irrelevant parts.


```python
"""
RAG System for HTML Files

Usage:
  rag_system.py --html_dir=<html_dir> [--chunk_size=<chunk_size>] [--chunk_overlap=<chunk_overlap>]
  rag_system.py (-h | --help)

Options:
  --html_dir=<html_dir>       Directory containing HTML files.
  --chunk_size=<chunk_size>   Chunk size for text splitting [default: 500].
  --chunk_overlap=<chunk_overlap>   Overlap size for text splitting [default: 50].
  -h --help                   Show this help message.

"""

import os
from docopt import docopt
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA


def extract_text_from_html(file_path):
    """
    Extracts text content from an HTML file, removing script and style elements.

    Args:
        file_path (str): Path to the HTML file.

    Returns:
        str: Cleaned text extracted from the HTML file.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        soup = BeautifulSoup(file, "html.parser")
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()
        return soup.get_text()


def process_html_files(directory_path):
    """
    Processes all HTML files in a specified directory by extracting text from each.

    Args:
        directory_path (str): Path to the directory containing HTML files.

    Returns:
        list: A list of dictionaries, each containing the text and metadata of an HTML file.
    """
    documents = []
    for file_name in os.listdir(directory_path):
        if file_name.endswith(".html"):
            file_path = os.path.join(directory_path, file_name)
            text = extract_text_from_html(file_path)
            documents.append({"text": text, "metadata": {"source": file_name}})
    return documents


def split_text_into_chunks(documents, chunk_size, chunk_overlap):
    """
    Splits the text content of documents into smaller chunks.

    Args:
        documents (list): A list of dictionaries containing text and metadata.
        chunk_size (int): Maximum size of each text chunk.
        chunk_overlap (int): Overlap between consecutive chunks.

    Returns:
        list: A list of dictionaries, each containing a chunk of text and associated metadata.
    """
    text_splitter = CharacterTextSplitter(chunk_size=int(chunk_size), chunk_overlap=int(chunk_overlap))
    processed_documents = []
    for doc in documents:
        chunks = text_splitter.split_text(doc["text"])
        for chunk in chunks:
            processed_documents.append({"text": chunk, "metadata": doc["metadata"]})
    return processed_documents


def create_vector_store(processed_documents, embedding_model_name="all-MiniLM-L6-v2"):
    """
    Creates a FAISS vector store by embedding the text chunks.

    Args:
        processed_documents (list): A list of dictionaries containing text chunks and metadata.
        embedding_model_name (str): Name of the embedding model to use.

    Returns:
        FAISS: A FAISS vector store containing the embedded text chunks.
    """
    embedding_model = SentenceTransformer(embedding_model_name)
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

    texts = [doc["text"] for doc in processed_documents]
    metadata = [doc["metadata"] for doc in processed_documents]

    vector_store = FAISS.from_texts(texts, embeddings, metadatas=metadata)
    return vector_store


def create_rag_system(vector_store, model_name="gpt2"):
    """
    Creates a Retrieval-Augmented Generation (RAG) system by integrating a retriever and a language model.

    Args:
        vector_store (FAISS): The FAISS vector store to use as the retriever.
        model_name (str): Name of the language model to use for text generation.

    Returns:
        RetrievalQA: A RAG system capable of answering queries using the vector store and language model.
    """
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    generation_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
    llm = HuggingFacePipeline(pipeline=generation_pipeline)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain


if __name__ == "__main__":
    """
    Main script to process HTML files, set up the RAG system, and answer queries interactively.
    """
    # Parse command-line arguments
    args = docopt(__doc__)

    html_dir = args["--html_dir"]
    chunk_size = args["--chunk_size"]
    chunk_overlap = args["--chunk_overlap"]

    # Step 1-2: Extract and process HTML files
    documents = process_html_files(html_dir)

    # Step 3: Split text into chunks
    processed_documents = split_text_into_chunks(documents, chunk_size, chunk_overlap)

    # Step 4: Create vector store
    vector_store = create_vector_store(processed_documents)

    # Step 5: Create RAG system
    qa_chain = create_rag_system(vector_store)

    # Step 6: Query the RAG system
    print("RAG system ready. Please type your query below (or type 'exit' to quit):")
    while True:
        query = input("Query: ")
        if query.lower() == "exit":
            print("Exiting")
            break
        result = qa_chain.run(query)
        print("Answer:", result["result"])
        print("Source Documents:", result["source_documents"])
```
