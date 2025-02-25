import os
import tiktoken
from pathlib import Path
from PyPDF2 import PdfReader
from dotenv import load_dotenv 
from langchain_openai import AzureOpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS


load_dotenv()

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_ADA_DEPLOYMENT_VERSION = os.getenv("AZURE_OPENAI_ADA_DEPLOYMENT_VERSION")
AZURE_OPENAI_CHAT_DEPLOYMENT_VERSION= os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_VERSION")
AZURE_OPENAI_ADA_EMBEDDING_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_ADA_EMBEDDING_DEPLOYMENT_NAME")
AZURE_OPENAI_ADA_EMBEDDING_MODEL_NAME = os.getenv("AZURE_OPENAI_ADA_EMBEDDING_MODEL_NAME")
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")


PDF_STORAGE_PATH = "./documents/wix-upgrade-docs"
INDEX_NAME = 'wix-upgrade'

embeddings = AzureOpenAIEmbeddings(
    deployment=AZURE_OPENAI_ADA_EMBEDDING_DEPLOYMENT_NAME,
    model=AZURE_OPENAI_ADA_EMBEDDING_MODEL_NAME,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    openai_api_key=AZURE_OPENAI_API_KEY,
    openai_api_version=AZURE_OPENAI_ADA_DEPLOYMENT_VERSION
)

"""
Converts all PDF files in the specified storage path to a FAISS vector store.

This function reads PDF files from the given directory, extracts the text content from each 
page, and accumulates it into a single text string. The text is then split into chunks using
a RecursiveCharacterTextSplitter based on specified separators and overlap. Finally, the 
chunks are embedded into a vector store using FAISS and saved to a local directory with 
the specified index name. 

Parameters:
pdf_storage_path (str): The path to the directory containing PDF files.
index_name (str): The name to be used for the saved FAISS index.

Returns:
None: This function does not return a value, but it saves the FAISS index locally.
"""
def pdfs_to_faiss(pdf_storage_path: str, index_name: str):
    pdf_directory = Path(pdf_storage_path)
    text = ""
    text_splitter = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=1000, 
        chunk_overlap=250,
        length_function=len
        )
    
    for pdf_path in pdf_directory.glob("*.pdf"):
        print(f"Reading document {pdf_path}")
        pdf_reader = PdfReader(pdf_path, True)
        pdf_text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            pdf_text += page_text

        txt_path = pdf_path.with_name(pdf_path.stem + ".txt")
        with open(txt_path, "w", encoding="utf-8") as file:
            file.write(pdf_text)
        text += pdf_text + "\n\n" 

    print("splitting text into chunks...")
    chunks = text_splitter.split_text(text)
    
    print("embedding all documents...")
    vector_store =FAISS.from_texts(chunks, embeddings)
    vector_store.save_local("./vector_stores", index_name)

    print("finished processing all pdfs!")


"""
Processes all PDF files in the specified directory, extracts text from each PDF,
splits the text into documents with larger chunks, and then embeds these documents
into a FAISS vector store.

Args:
    pdf_storage_path (str): The path to the directory containing the PDF files.
    index_name (str): The name of the FAISS index to save the vector store with.

The function reads each PDF file, extracts its text content using a PyMuPDFLoader,
and splits the text from each PDF into documents with larger chunk sizes. These documents
are then combined into a single list of documents. The combined documents are embedded
into a FAISS vector store, which is subsequently saved locally under the specified index name.
"""
def pdfs_to_faiss2(pdf_storage_path: str, index_name: str):
    pdf_directory = Path(pdf_storage_path)
    docs = []
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " "],
        chunk_size=1000, 
        chunk_overlap=250,
        length_function=len
        )
    
    for pdf_path in pdf_directory.glob("*.pdf"):
        print(f"Reading document {pdf_path}")
        loader = PyMuPDFLoader(str(pdf_path))
        documents = loader.load()
        pdf_docs= text_splitter.split_documents(documents)
        docs += pdf_docs
    
    print("embedding all documents...")
    vector_store =FAISS.from_documents(docs, embeddings)
    vector_store.save_local("./vector_stores", index_name)

    print("finished processing all pdfs!")


"""
Processes all PDF files in the specified storage directory by reading, splitting, and embedding their contents.
The embedded documents are then saved to a local FAISS vector store.

Args:
    pdf_storage_path (str): The path to the directory containing the PDF files to be processed.
    faiss_dir_path (str): The path to the directory where the FAISS vector store files will be saved.

Returns:
    None: This function does not return a value, but it saves the processed FAISS vector store files locally.
"""
def pdfs_to_faiss_files(pdf_storage_path: str, faiss_dir_path: str):
    pdf_directory = Path(pdf_storage_path)
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " "],
        chunk_size=1000, 
        chunk_overlap=200,
        length_function=len
        )

    for pdf_path in pdf_directory.glob("*.pdf"):
        print(f"Reading document {pdf_path}")
        loader = PyMuPDFLoader(str(pdf_path))
        documents = loader.load()
        pdf_docs= text_splitter.split_documents(documents)

        print(f"embedding document {pdf_path}")
        vector_store =FAISS.from_documents(pdf_docs, embeddings)
        vector_store.save_local(faiss_dir_path, pdf_path.stem)
        print(f"finished processing pdf {pdf_path}!")

"""
Merges multiple FAISS files into a single index.

Args:
    faiss_dir_path (str): The path to the directory containing the FAISS files.
    index_name (str): The name of the index to be saved.
"""
def merge_faiss_files(faiss_dir_path: str, index_name: str):
    faiss_directory = Path(faiss_dir_path)
    
    is_first_faiss = True
    first_faiss = None
    for faiss_path in faiss_directory.glob("*.faiss"):
        print(f"Loading FAISS file {faiss_path}")
        if is_first_faiss:
            is_first_faiss = False
            first_faiss = FAISS.load_local(
                folder_path=faiss_dir_path, 
                embeddings=embeddings, 
                index_name=faiss_path.stem,
                allow_dangerous_deserialization=True)
        else:
            print(f"Merging with {faiss_path}")
            first_faiss.merge_from(FAISS.load_local(
                folder_path=faiss_dir_path, 
                embeddings=embeddings, 
                index_name=faiss_path.stem,
                allow_dangerous_deserialization=True))

    first_faiss.save_local("./vector_stores", index_name)
        

"""
Counts the number of tokens in a given text using the specified model's tokenizer.
If the model is not found, it defaults to using the 'cl100k_base' tokenizer.

Args:
    text (str): The input text to be tokenized.
    model (str): The name of the model for which the tokenizer is used.

Returns:
    int: The total number of tokens in the text.
"""
def count_tokens(text, model):
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(text))
    print(f"Total tokens: {num_tokens}")
    return num_tokens

if __name__ == "__main__":
    pdfs_to_faiss_files(PDF_STORAGE_PATH, "./faiss_files")
    merge_faiss_files("./faiss_files", INDEX_NAME)