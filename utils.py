# utils.py
import uuid
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import streamlit as st
import os, re
import torch
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

def get_system_prompt():
    return """You are a highly knowledgeable, professional medical assistant AI (AGNOS Assistant), Your character is women. using Retrieval-Augmented Generation (RAG) based on Thailand.

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
- Include **citations** like (Source: [Source Agnos forum url], [Title], [Year]) if the topic is needed.
- If the topic or technical is needed, Briefly mention retrieval source (e.g., ข้อมูลจากแหล่งข้อมูลอ้างอิง <Source>) or Source URL.
- If the topic or technical is needed, reference and can't find any supporting documents found: say **"ไม่พบเอกสารจากฐานข้อมูลที่ดึงมา"**.

**Process Example:**
User: "เจ็บคอและมีไข้มา 2 วัน"  
Assistant:
  1. **แสดงความห่วงใยเล็กน้อย**
  2. **สรุปความกังวล**
  3. แหล่งข้อมูลที่ดึงมาหากจำเป็น (เช่น Source)
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
        try:
            self.model = SentenceTransformer(model_name, device=self.device)
            print(f"Model {model_name} loaded successfully on {self.device}.")
        except Exception as e:
            print(f"Error loading embedding model {model_name}: {e}")
            self.model = None

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
            if not text or not text.strip():
                print("Warning: Empty text provided for embedding")
                return []
            result = self.model.encode(text, normalize_embeddings=True).tolist()
            if not result:
                print("Warning: Empty embedding result")
                return []
            return result
        except Exception as e:
            print(f"Error embedding text: {e}")
            # Try to provide a fallback embedding (zeros)
            try:
                dim = self.model.get_sentence_embedding_dimension() if self.model else 1024
                return [0.0] * dim
            except:
                return []
        
    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        try:
            if not documents:
                print("Warning: Empty documents list provided for embedding")
                return []
            valid_docs = [doc for doc in documents if doc and doc.strip()]
            if not valid_docs:
                print("Warning: No valid documents to embed")
                return []
            result = self.model.encode(valid_docs, normalize_embeddings=True)
            if result is None or len(result) == 0:
                print("Warning: Empty embedding results")
                return []
            return result.tolist() if hasattr(result, 'tolist') else result
        except Exception as e:
            print(f"Error embedding documents: {e}")
            try:
                dim = self.model.get_sentence_embedding_dimension() if self.model else 1024
                return [[0.0] * dim for _ in documents]
            except:
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
                # Store content in metadata so we can retrieve it later
                meta['content'] = doc
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
            results = self.index.query(vector=embedding, top_k=k, include_metadata=True, include_values=True)
            return results.matches
        except Exception as e:
            print(f"Error querying Pinecone: {e}")
            return []