# scrapper.py
import asyncio
import os
import glob

from crawl4ai import AsyncWebCrawler
from crawl4ai.async_configs import CrawlerRunConfig
from sentence_transformers import SentenceTransformer
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_milvus import Milvus
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

import utils

# Config
URLS = [
    "https://www.agnoshealth.com/forums",
    "https://www.agnoshealth.com/forums/search?query="
]
OUTPUT_DIR = "scraped_txt"
MILVUS_DB = "./milvus_agnos.db"
COLLECTION = "agnos_forums"
MODEL_NAME = "BAAI/bge-m3"

async def crawl_and_save():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    browser_cfg = {}  # default
    run_cfg = CrawlerRunConfig(
        only_text=True,
        excluded_tags=["script", "style", "nav", "footer"],
        exclude_external_links=True
    )

    async with AsyncWebCrawler(**browser_cfg) as crawler:
        for url in URLS:
            print(f"🔍 Crawling {url}...")
            result = await crawler.arun(url=url, config=run_cfg)
            sanitized_url = utils.sanitize_filename(url.split("/")[-1] or "root")
            filename = os.path.join(OUTPUT_DIR, sanitized_url)
            with open(f"{filename}.txt", "w", encoding="utf-8") as f:
                f.write(result.markdown or result.cleaned_html or result.text or "")
            print(f"✅ Saved to {filename}.txt")

def load_texts():
    texts = []
    for path in glob.glob(os.path.join(OUTPUT_DIR, "*.txt")):
        with open(path, "r", encoding="utf-8") as f:
            texts.append((os.path.basename(path), f.read()))
    return texts

def index_to_milvus(texts):
    # Embedding model
    embedder = SentenceTransformerEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": "cuda" if os.getenv("CUDA", None) else "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    # Vector store init or load
    vstore = Milvus.from_documents(
        documents=[],
        embedding=embedder,
        connection_args={"uri": MILVUS_DB},
        collection_name=COLLECTION,
        drop_old=True
    )

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    for fname, content in texts:
        fname = utils.sanitize_filename(fname)
        chunks = splitter.create_documents(
            [content], metadata=[{"source": fname}]
        )
        vstore.add_documents(chunks)
        print(f"✔️ Indexed {len(chunks)} chunks from {fname}")

    return vstore


def main():
    asyncio.run(crawl_and_save())
    texts = load_texts()
    print(f"Loaded {len(texts)} documents")
    vstore = index_to_milvus(texts)
    print("🎉 All done! Milvus contains:", vstore._collection.name)


if __name__ == "__main__":
    main()
