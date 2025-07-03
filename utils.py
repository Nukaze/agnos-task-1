# utils.py
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_milvus import Milvus
from langchain.schema import Document
import streamlit as st
import os, re
import torch

def get_system_prompt():
    return """You are a highly knowledgeable, professional medical assistant AI using Retrieval-Augmented Generation (RAG).

When a user/patient asks a question or describes symptoms, perform the following:

1. **Retrieve** authoritative medical documents (e.g., clinical guidelines, PubMed abstracts, reputable health websites).
2. **Ground** your response in these documents—only include information explicitly supported by citations.
3. **Summarize** the user’s concern **in Thai** at the start, with key technical terms in English in parentheses.
4. **Explain** in clear and compassionate Thai; define any medical/technical terms in English.
5. **Indicate** when evidence is uncertain or insufficient.
6. **Ask follow‑up questions** if more context is needed (e.g., onset, duration, severity, history).
7. **Maintain** a respectful, empathetic, and professional tone.
8. **If unsure, serious, or evidence lacking**, advise:  
   “ควรติดต่อแพทย์หรือโทรหาฉุกเฉินทันที (call emergency services).”

**Formatting:**
- Include **citations** like (Source: [Title], [Year]) or numerical footnotes as possible.
- Briefly mention retrieval source (e.g., “ข้อมูลจาก PubMed, Agnos Health, Bangkok Hospital”).
- If no supporting documents found: say **“ไม่พบหลักฐานจากเอกสารที่ดึงมา”**.

**Process Example:**
User: “เจ็บคอและมีไข้มา 2 วัน”  
Assistant:
  1. สรุปความกังวล (ภาษาไทย)  
  2. แหล่งข้อมูลที่ดึงมา (เช่น PubMed, Agnos Health, Bangkok Hospital, abstracts)  
  3. อธิบายความเป็นไปได้ สาเหตุ และคำแนะนำ  
  4. ถามติดตามหรือให้คำแนะนำเพิ่มเติม

Always prioritize patient safety, clarity, and evidence-based care.
"""


def write_file(filename: str, content: str) -> None:
    try:
        with open(filename, 'a', encoding='utf-8') as f:
            f.write(str(content))
        print(f"File {filename} written successfully.")
    except Exception as e:
        print(f"Error writing to file {filename}: {e}")


def sanitize_filename(filename: str) -> str:
    # Replace invalid characters with underscores
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Check for reserved filenames
    reserved_names = {"CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9", "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9"}
    if sanitized.upper() in reserved_names:
        sanitized = f"_{sanitized}"
    return sanitized


class TextEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-m3"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.model = SentenceTransformerEmbeddings(
            model_name=model_name, 
            moedl_kwargs={"device": self.device},
            encode_kwargs={
                "normalize_embeddings": True, 
                "convert_to_tensor": True
            }
        )
        self.expected_embedding_dimension = self.model.get_sentence_embedding_dimension()
        print(f"Model {model_name} loaded successfully. with expected embedding dimension: {self.expected_embedding_dimension}")

    def embed_text(self, text: str) -> list[float]:
        try:
            return self.model.embed_query(text)
        except Exception as e:
            print(f"Error embedding text: {e}")
            return []
        
    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        try:
            return self.model.embed_documents(documents)
        except Exception as e:
            print(f"Error embedding documents: {e}")
            return []
        
        
class MilvusManager:
    def __init__(
            self,
            embedder: TextEmbedder = None, 
            db_path: str = "./milvus_demo.db", 
            collection_name: str = "medical_forums",
        ):
        self.embedder = embedder or TextEmbedder()
        self.db_path = db_path
        self.collection_name = collection_name
        
        self.vector_store = Milvus.from_documents(
            documents=[],
            embedding=self.embedder.model,
            connection_args={
                "uri": self.db_path,
                "collection_name": self.collection_name
            }
        )
        
        print(f"{MilvusManager.__name__} initialized with collection: {self.collection_name} at {self.db_path}")
    
    
    def add_documents(self, documents: list[str], metadatas: list[dict]) -> None:
        try:
            docs = [
                Document(page_content=doc, metadatas=metadatas[i] if metadatas else {}) 
                for i,doc in enumerate(documents)
            ]
            self.vector_store.add_documents(docs)
            print(f"Added {len(documents)} documents to Milvus collection {self.collection_name}.")
            
        except Exception as e:
            print(f"Error adding documents to Milvus: {e}")
            
            
    def retrieve(self, query: str, k: int = 5) -> list[Document]:
        try:
            retriever = self.vector_store.as_retriever(
                search_type="similarity", 
                search_kwargs={"k": k}
            )
            results = retriever.get_relevant_documents(query)
            return results
            
        except Exception as e:
            print(f"Error retrieving documents from Milvus: {e}")
            return []
        
        
    def similarity_search(self, query: str, k: int = 5, filter: dict = None) -> list[Document]:
        try:
            # similarity_score_threshold
            results = self.vector_store.similarity_search(query=query, k=k, filter=filter)
            return results
            
        except Exception as e:
            print(f"Error performing similarity search: {e}")
            return []
        
    
    def mmr_search(self, query: str, k: int = 5, filter: dict = None, lambda_mult: float = 0.5) -> list[Document]:
        try:
            results = self.vector_store.mmr_search(query=query, k=k, filter=filter, lambda_mult=lambda_mult)
            return results
            
        except Exception as e:
            print(f"Error performing MMR search: {e}")
            return []