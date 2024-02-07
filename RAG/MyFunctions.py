from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import Weaviate
from PyPDF2 import PdfReader
from langchain_community.embeddings import HuggingFaceEmbeddings
#from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)

from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv



load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


class MySpecialFunctions:
    def __init__(self):
        pass
    
    def get_file_text(self, files):
        text = ""
        for file in files:
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    text += content
            except Exception as e:
                print(f"Error reading file {file}: {e}")

        return text
    
    
    def get_pdf_text(self, pdf):
        text = ""
        for doc in pdf:
            reader = PdfReader(doc)
            for page in reader.pages:
                text += page.extract_text()

        return text
    
    
    def get_text_chunks(self, text):
        text_siplitter = RecursiveCharacterTextSplitter(
            chunk_size = 100, 
            chunk_overlap = 20,
            separators=['\n', '\n\n'],
            length_function = len)
        chunk = text_siplitter.split_text(text)
        return chunk
    

    def get_OpenAIEmbeddings(self):
        return OpenAIEmbeddings()

    
    def get_HuggingFaceEmbeddings(self):
        hf_embedder = HuggingFaceInstructEmbeddings(
            model_name="hkunlp/instructor-xl", 
            model_kwargs={"device": "cpu"}
        )
        return hf_embedder
    
    def get_open_embedding(self):
        # create the open-source embedding function
        open_embedder = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        return open_embedder
    

    def get_faiss_vectorstore(self, text_chunks, embeddings_method):
        vectorstore = FAISS.from_texts(texts= text_chunks, embedding= embeddings_method)
        #vectorstore = FAISS.from_documents(documents= text_chunks, embedding= embeddings_method)
        return vectorstore
    
    def get_chroma_vectorstore(self, text_chunks, embeddings_method):
        # vectoestore = Chroma.from_documents(
        #     documents = text_chunks,
        #     embedding = embeddings_method,
        # )

        vectoestore = Chroma.from_texts(
            text_chunks, 
            embeddings_method,
        )       
        return vectoestore
    
    def get_weaviate_vectorstore(self, text_chunks, embeddings_method):
        vectorstore = Weaviate.from_texts(
            text_chunks,
            embeddings_method,
            weaviate_url="http://127.0.0.1:8080"
        )
