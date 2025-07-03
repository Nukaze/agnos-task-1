# scrapper.py
import asyncio
import os
import glob
from crawl4ai import AsyncWebCrawler
from crawl4ai.async_configs import CrawlerRunConfig
import utils
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Config
TARGET_URLS = [
    "https://www.agnoshealth.com/forums",
    "https://www.agnoshealth.com/forums/search?query="
]
OUTPUT_DIR = "scraped_txt"
COLLECTION = "agnos_forums"

async def crawl_and_save():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    browser_cfg = {}
    run_cfg = CrawlerRunConfig(
        only_text=True,
        excluded_tags=["script", "style", "nav", "footer"],
        exclude_external_links=True
    )
    async with AsyncWebCrawler(**browser_cfg) as crawler:
        for url in TARGET_URLS:
            print(f"🔍 Crawling {url}...")
            result = await crawler.arun(url=url, config=run_cfg)
            sanitized_url = utils.sanitize_filename(url.split("/")[-1] or "root")
            filename = os.path.join(OUTPUT_DIR, sanitized_url + ".txt")
            utils.write_file(
                filename, 
                result.markdown or result.cleaned_html or result.text or ""
            )
            print(f"✅ Saved to {filename}")

def load_texts():
    texts = []
    for path in glob.glob(os.path.join(OUTPUT_DIR, "*.txt")):
        with open(path, "r", encoding="utf-8") as f:
            texts.append((os.path.basename(path), f.read()))
    return texts

def index_to_milvus(texts):
    embedder = utils.TextEmbedder()
    milvus = utils.MilvusManager(
        use_remote=True,
        remote_uri="http://localhost:19530",
        embedder=embedder,
        collection_name=COLLECTION
    )
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    for fname, content in texts:
        fname = utils.sanitize_filename(fname)
        chunks = splitter.create_documents(
            [content], metadata=[{"source": fname}]
        )
        docs = [chunk for chunk in chunks]
        milvus.add_documents([doc.page_content for doc in docs], [doc.metadata for doc in docs])
        print(f"✔️ Indexed {len(chunks)} chunks from {fname}")
    return milvus

def main():
    asyncio.run(crawl_and_save())
    texts = load_texts()
    print(f"Loaded {len(texts)} documents")
    milvus = index_to_milvus(texts)
    print("🎉 All done! Milvus is ready for ollama frontend.")

if __name__ == "__main__":
    main()
