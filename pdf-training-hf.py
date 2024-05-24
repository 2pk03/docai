# Copyright (c) Alexander Alten; 2pk03, 2024. All rights reserved.
# This code is licensed under the CC BY-NC-SA 4.0 license.

import subprocess
import sys
import pkg_resources

# Required libraries
required_libraries = ["langchain", "langchain-community", "transformers", "accelerate", "bitsandbytes", "sentence_transformers"]

# Check for Python version (requires at least 3.9)
if sys.version_info < (3, 9):
    print("Error: Python 3.9 or higher is required. Please update your Python installation.")
    sys.exit(1)

# Check for and install missing libraries
def install_or_upgrade_library(library_name):
    try:
        pkg_resources.get_distribution(library_name)
    except pkg_resources.DistributionNotFound:
        subprocess.check_call([sys.executable, "-m", "pip", "install", library_name])
    else:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", library_name])


for library in required_libraries:
    install_or_upgrade_library(library)

# --- PDF Q&A Functionality --- 
import argparse
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
from sentence_transformers import SentenceTransformer # Import the SentenceTransformer class
import torch

# Load the SentenceTransformer model for embeddings
embeddings = SentenceTransformer('all-mpnet-base-v2')

def load_and_process_pdf(file_path: str):
    """Loads a PDF, splits it, and creates a vector store for similarity search."""
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(pages)
    db = FAISS.from_documents(docs, embeddings)
    return db

def ask_and_get_answer(db, llm, query: str):
    """Retrieves relevant documents and generates an answer using the provided model."""
    docs = db.similarity_search(query)
    answer = llm("Answer the following question using only the provided context:\n\nContext:\n" + docs[0].page_content + "\n\nQuestion:\n" + query)
    return answer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ask questions about your PDF using a Hugging Face LLM.")
    parser.add_argument("model_id", type=str, help="The Hugging Face model ID (e.g., 'mistralai/Mistral-7B-Instruct-v0.1'). Find models at https://huggingface.co/models")
    parser.add_argument("pdf_file_path", type=str, help="The path to your PDF file.")
    args = parser.parse_args()

    print("\nRemember to choose a model suitable for instruction/question-answering tasks.\nYou can find a list of models at https://huggingface.co/models\n")
    # Load the model
    pipe = pipeline("text-generation", model=args.model_id, device_map="auto", model_kwargs={"torch_dtype":torch.float16})
    llm = HuggingFacePipeline(pipeline=pipe)
    # Load and process the PDF
    db = load_and_process_pdf(args.pdf_file_path)

    while True:
        # Ask for a question and get an answer
        query = input("Ask a question about the PDF (or type 'exit' to quit): ")
        if query.lower() == "exit":
            break
        answer = ask_and_get_answer(db, llm, query)
        print(answer)


