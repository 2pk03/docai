# Copyright (c) Alexander Alten; 2pk03, 2024. All rights reserved.
# This code is licensed under the CC BY-NC-SA 4.0 license.

import subprocess
import sys
import pkg_resources
import os
import argparse

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

# Check for and install HuggingFace Hub
try:
    import huggingface_hub
except ImportError:
    install_or_upgrade_library("huggingface_hub")

for library in required_libraries:
    install_or_upgrade_library(library)

# Scan for .gguf models in the current directory
gguf_models = [f for f in os.listdir('.') if f.endswith('.gguf')]

if gguf_models:
    print("Found the following .gguf models:")
    for i, model in enumerate(gguf_models):
        print(f"{i+1}. {model}")

    # Get user's choice of model
    while True:
        try:
            choice = int(input("Enter the number of the model you want to use: "))
            if 1 <= choice <= len(gguf_models):
                model_id = gguf_models[choice - 1]
                break
            else:
                print("Invalid choice. Please enter a number from the list.")
        except ValueError:
            print("Invalid input. Please enter a number.")
else:
    print("No .gguf models found in the current directory.")
    sys.exit(1)

# --- PDF Q&A Functionality --- 

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from transformers import pipeline
from sentence_transformers import SentenceTransformer # Use SentenceTransformer for embeddings
import torch

# Load the SentenceTransformer model for embeddings (you can choose a different model)
embeddings = SentenceTransformer('all-mpnet-base-v2')
pipe = pipeline("text-generation", model=model_id, device_map="auto", model_kwargs={"torch_dtype":torch.float16})
llm = HuggingFacePipeline(pipeline=pipe)

def load_and_process_pdf(file_path: str):
    """Loads a PDF, splits it, and creates a vector store for similarity search."""
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(pages)
    db = FAISS.from_documents(docs, embeddings)
    return db

def ask_and_get_answer(db, query: str):
    """Retrieves relevant documents and generates an answer using the model."""
    docs = db.similarity_search(query)
    answer = llm("Answer the following question using only the provided context:\n\nContext:\n" + docs[0].page_content + "\n\nQuestion:\n" + query)
    return answer

if __name__ == "__main__":
    pdf_file_path = input("Enter the path to your PDF file: ")

    # Load and process the PDF
    db = load_and_process_pdf(pdf_file_path)

    while True:
        # Ask for a question and get an answer
        query = input("Ask a question about the PDF (or type 'exit' to quit): ")
        if query.lower() == "exit":
            break
        answer = ask_and_get_answer(db, query)
        print(answer)

