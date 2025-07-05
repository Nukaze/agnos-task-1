# utils.py
import uuid
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import streamlit as st
import os, re
import torch
from pinecone import Pinecone, ServerlessSpec

def get_system_prompt():
    return """You are a highly knowledgeable, professional medical assistant AI (Agnos Assistant), Your character is women. using Retrieval-Augmented Generation (RAG) based on Thailand.

When a user/patient asks a question or describes symptoms, perform the following:

1. **Retrieve** authoritative medical documents (e.g., clinical guidelines, reputable health websites).
2. **Ground** your response in these documents—only include information explicitly supported by citations.
3. **Summarize** must response **in Thai**.
4. **Explain** in clear and compassionate Thai; if have any medical/technical terms in English.
5. **Indicate** when evidence is uncertain or insufficient.
6. **Ask follow‑up questions** if more context is needed (e.g., onset, duration, severity, history).
7. **Maintain** a respectful, empathetic, and professional tone.
8. **If unsure, serious, or evidence lacking**, advise:  
   "ควรติดต่อแพทย์หรือโทรหาฉุกเฉินทันที (call emergency services)."

**Formatting:**
- If the topic or technical is needed, Include **citations** like (Source forum url: [Title], [Year]) or numerical footnotes if the topic is needed.
- If the topic or technical is needed, Briefly mention retrieval source (e.g., "ข้อมูลจาก Agnos Health, Bangkok Hospital, etc,...").
- If the topic or technical is needed, reference and can't find any supporting documents found: say **"ไม่พบหลักฐานจากเอกสารที่ดึงมา"**.

**Process Example:**
User: "เจ็บคอและมีไข้มา 2 วัน"  
Assistant:
  1. **แสดงความห่วงใยเล็กน้อย**
  2. **สรุปความกังวล**
  3. แหล่งข้อมูลที่ดึงมาหากจำเป็น (เช่น Agnos Health, Bangkok Hospital, abstracts, ...)
  4. อธิบายความเป็นไปได้ สาเหตุ และคำแนะนำ 
  5. สรุปสั้นๆ
  6. ถามติดตามหรือให้คำแนะนำเพิ่มเติม

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
        # Better CUDA detection
        self.device = self._get_best_device()
        print(f"Using device: {self.device}")
        
        # Initialize model with proper device handling
        try:
            self.model = SentenceTransformerEmbeddings(
                model_name=model_name, 
                model_kwargs={
                    "device": self.device,
                    "trust_remote_code": True
                },
                encode_kwargs={
                    "normalize_embeddings": True, 
                    "convert_to_tensor": True,
                    "device": self.device
                }
            )
            print(f"Model {model_name} loaded successfully on {self.device}.")
        except Exception as e:
            print(f"Error loading model on {self.device}: {e}")
            # Fallback to CPU
            self.device = "cpu"
            print(f"Falling back to CPU...")
            self.model = SentenceTransformerEmbeddings(
                model_name=model_name, 
                model_kwargs={
                    "device": "cpu",
                    "trust_remote_code": True
                },
                encode_kwargs={
                    "normalize_embeddings": True, 
                    "convert_to_tensor": True,
                    "device": "cpu"
                }
            )
            print(f"Model {model_name} loaded successfully on CPU.")

    def _get_best_device(self):
        """Get the best available device for PyTorch"""
        try:
            import torch
            
            # Check if CUDA is available and working
            if torch.cuda.is_available():
                # Test CUDA functionality
                try:
                    test_tensor = torch.tensor([1.0], device="cuda")
                    print(f"CUDA is available and working. Found {torch.cuda.device_count()} GPU(s)")
                    print(f"GPU: {torch.cuda.get_device_name(0)}")
                    return "cuda"
                except Exception as e:
                    print(f"CUDA test failed: {e}")
                    return "cpu"
            else:
                print("CUDA is not available")
                return "cpu"
        except ImportError:
            print("PyTorch not available")
            return "cpu"

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
            use_remote: bool = False,
            remote_uri: str = None,
        ):
        self.embedder = embedder or TextEmbedder()
        self.db_path = db_path
        self.collection_name = collection_name
        self.use_remote = use_remote
        self.remote_uri = remote_uri

        if self.use_remote and self.remote_uri:
            self._init_remote()
        else:
            self._init_local()

    def _init_local(self):
        # Milvus Lite (local) client for low-level API (e.g. stats, dump, etc.)
        self.milvus_client = MilvusClient(self.db_path)
        # LangChain vector store for high-level RAG integration
        self.vector_store = Milvus.from_documents(
            documents=[],
            embedding=self.embedder.model,
            connection_args={
                "uri": self.db_path,
                "collection_name": self.collection_name
            }
        )
        print(f"MilvusManager (local) initialized with collection: {self.collection_name} at {self.db_path}")

    def _init_remote(self):
        self.vector_store = Milvus.from_documents(
            documents=[],
            embedding=self.embedder.model,
            connection_args={
                "uri": self.remote_uri,
                "collection_name": self.collection_name
            }
        )
        print(f"MilvusManager (remote) initialized with collection: {self.collection_name} at {self.remote_uri}")

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

class PineconeManager:
    def __init__(
            self,
            embedder: TextEmbedder = None,
            api_key: str = None,
            index_name: str = None,
            dimension: int = 1024,
            cloud: str = None,
            region: str = None,
        ):
        self.embedder = embedder
        self.api_key = api_key
        self.index_name = index_name
        self.cloud = cloud
        self.region = region
        self.vector_dimension = dimension
        
        if not all([self.api_key, self.index_name, self.region]):
            raise ValueError("Pinecone API key, index name, and region must be set via env or secrets.")
        self.pc = Pinecone(api_key=self.api_key)
        # Create index if needed
        if self.index_name not in self.pc.list_indexes().names():
            if not self.region:
                raise ValueError("Pinecone region must be specified via env, secrets, or argument.")
            self.pc.create_index(
                name=self.index_name,
                dimension=self.vector_dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud=self.cloud, region=self.region)
            )
        self.index = self.pc.Index(self.index_name)
        print(f"PineconeManager initialized with index: {self.index_name}")


    def add_documents(self, documents: list[str], metadatas: list[dict]) -> None:
        try:
            docs = []
            for i, doc in enumerate(documents):
                # Embed and check dimension
                embedding = self.embedder.embed_text(doc)
                if len(embedding) != self.vector_dimension:
                    print(f"Warning: Embedding dimension mismatch for doc {i}. Skipping.")
                    continue
                meta = metadatas[i] if metadatas else {}
                docs.append((str(uuid.uuid4()), embedding, meta))
            if docs:
                self.index.upsert(vectors=docs)
                print(f"Upserted {len(docs)} documents to Pinecone index {self.index_name}.")
            else:
                print("No valid documents to upsert.")
        except Exception as e:
            print(f"Error adding documents to Pinecone: {e}")


    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        try:
            embedding = self.embedder.embed_text(query)
            if len(embedding) != self.vector_dimension:
                print("Query embedding dimension mismatch.")
                return []
            results = self.index.query(vector=embedding, top_k=k, include_metadata=True)
            return results.matches
        except Exception as e:
            print(f"Error querying Pinecone: {e}")
            return []