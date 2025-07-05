# scrapper.py
import asyncio
import os
import glob
import time
import json
from urllib.parse import urljoin, urlparse
from crawl4ai import AsyncWebCrawler
from crawl4ai.async_configs import CrawlerRunConfig, CacheMode
import utils
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
import os

# Config
BASE_URL = "https://www.agnoshealth.com/forums"
TARGET_URLS = [
    "https://www.agnoshealth.com/forums/%E0%B9%82%E0%B8%A3%E0%B8%84%E0%B8%99%E0%B8%AD%E0%B8%99%E0%B9%84%E0%B8%A1%E0%B9%88%E0%B8%AB%E0%B8%A5%E0%B8%B1%E0%B8%9A/137",
    "https://www.agnoshealth.com/forums/%E0%B8%81%E0%B8%A3%E0%B8%B0%E0%B9%80%E0%B8%9E%E0%B8%B2%E0%B8%B0%E0%B8%9B%E0%B8%B1%E0%B8%AA%E0%B8%AA%E0%B8%B2%E0%B8%A7%E0%B8%B0%E0%B8%AD%E0%B8%B1%E0%B8%81%E0%B9%80%E0%B8%AA%E0%B8%9A/2674",
    "https://www.agnoshealth.com/forums",
    "https://www.agnoshealth.com/forums/search?query="
    "",
]
OUTPUT_DIR = "data_scraped_txt"
COLLECTION = "agnos_forums"

# Anti-bot configuration based on Crawl4AI docs
ANTI_BOT_CONFIG = CrawlerRunConfig(
    # Anti-bot features
    magic=True,                    # Multiple stealth features
    simulate_user=True,            # Mimics mouse movements and random delays
    override_navigator=True,       # Fakes navigator properties
    
    # Content processing
    only_text=True,
    excluded_tags=["script", "style", "nav", "footer", "header", "icon", "icons"],
    exclude_external_links=True,
    word_count_threshold=10,
    
    # Page navigation
    delay_before_return_html=2.0,  # Wait 2s before capturing
    page_timeout=30000,            # 30s timeout
    
    # Cache control
    cache_mode=CacheMode.ENABLED,
    check_robots_txt=True,
    
    # Verbose logging
    verbose=True
)

async def discover_forum_urls(base_url: str, max_pages: int = 50) -> list[str]:
    """Discover all forum URLs using anti-bot protection"""
    discovered_urls = set()
    page_urls = [base_url]
    
    # Config for URL discovery (needs links)
    discovery_config = CrawlerRunConfig(
        magic=True,
        simulate_user=True,
        override_navigator=True,
        only_text=False,  # Need links
        excluded_tags=["script", "style", "icon"],
        exclude_external_links=False,  # Need to see all links
        delay_before_return_html=1.0,
        page_timeout=30000,
        cache_mode=CacheMode.ENABLED,
        check_robots_txt=True,
        verbose=True
    )
    
    async with AsyncWebCrawler() as crawler:
        for page_num in range(max_pages):
            if not page_urls:
                break
                
            current_url = page_urls.pop(0)
            print(f"🔍 Discovering URLs from page {page_num + 1}: {current_url}")
            
            try:
                # Add random delay to avoid rate limiting
                await asyncio.sleep(2 + (time.time() % 3))
                
                result = await crawler.arun(
                    url=current_url,
                    config=discovery_config
                )
                
                if not result.success:
                    print(f"❌ Failed to crawl {current_url}: {result.error_message}")
                    continue
                
                # Extract all links from the page
                if result.links:
                    for link in result.links:
                        full_url = urljoin(current_url, link)
                        # Filter for forum-related URLs
                        if "agnoshealth.com/forums" in full_url and full_url not in discovered_urls:
                            discovered_urls.add(full_url)
                            if len(discovered_urls) % 10 == 0:
                                print(f"📊 Discovered {len(discovered_urls)} URLs so far...")
                
                # Add pagination URLs if found
                if result.links:
                    for link in result.links:
                        if any(keyword in link.lower() for keyword in ['page', 'p=', '?page']):
                            full_url = urljoin(current_url, link)
                            if full_url not in page_urls and full_url not in discovered_urls:
                                page_urls.append(full_url)
                                
            except Exception as e:
                print(f"❌ Error discovering URLs from {current_url}: {e}")
                continue
    
    return list(discovered_urls)

async def scrape_forum_page(url: str, crawler: AsyncWebCrawler, session_id: str = None) -> dict:
    """Scrape a single forum page with anti-bot protection"""
    try:
        # Add random delay to avoid detection
        await asyncio.sleep(1 + (time.time() % 2))
        
        # Use session_id for persistent browsing
        config = ANTI_BOT_CONFIG
        if session_id:
            config.session_id = session_id
        
        result = await crawler.arun(
            url=url,
            config=config
        )
        
        if not result.success:
            print(f"❌ Failed to scrape {url}: {result.error_message}")
            return None
        
        # Extract title and content
        title = result.title or urlparse(url).path.split('/')[-1]
        content = result.text or result.markdown or result.cleaned_html or ""
        
        return {
            "url": url,
            "title": title,
            "content": content,
            "timestamp": time.time()
        }
        
    except Exception as e:
        print(f"❌ Error scraping {url}: {e}")
        return None

async def crawl_all_forums():
    """Main function to discover and scrape all forum URLs with anti-bot protection"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("🚀 Starting Agnos Health Forum Discovery with Anti-Bot Protection...")
    
    # Step 1: Discover all forum URLs
    discovered_urls = await discover_forum_urls(BASE_URL, max_pages=50)
    print(f"✅ Discovered {len(discovered_urls)} forum URLs")
    
    # Save discovered URLs for reference
    with open(os.path.join(OUTPUT_DIR, "discovered_urls.json"), "w", encoding="utf-8") as f:
        json.dump(discovered_urls, f, indent=2, ensure_ascii=False)
    
    # Step 2: Scrape each URL with anti-bot protection
    scraped_data = []
    session_id = f"agnos_session_{int(time.time())}"
    
    async with AsyncWebCrawler() as crawler:
        for i, url in enumerate(discovered_urls):
            print(f"📄 Scraping {i+1}/{len(discovered_urls)}: {url}")
            
            data = await scrape_forum_page(url, crawler, session_id)
            if data:
                scraped_data.append(data)
                
                # Save individual file
                filename = utils.sanitize_filename(f"{i+1:04d}_{urlparse(url).path.replace('/', '_')}.txt")
                filepath = os.path.join(OUTPUT_DIR, filename)
                
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(f"URL: {data['url']}\n")
                    f.write(f"Title: {data['title']}\n")
                    f.write(f"Timestamp: {data['timestamp']}\n")
                    f.write("-" * 80 + "\n")
                    f.write(data['content'])
                
                # Add delay between requests
                if (i + 1) % 10 == 0:
                    print(f"⏸️  Taking a break... ({i+1}/{len(discovered_urls)})")
                    await asyncio.sleep(5)
    
    print(f"✅ Scraped {len(scraped_data)} pages successfully")
    return scraped_data

def load_scraped_data():
    """Load all scraped data from files"""
    data = []
    for filepath in glob.glob(os.path.join(OUTPUT_DIR, "*.txt")):
        if filepath.endswith("discovered_urls.json"):
            continue
            
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
                
            # Parse the file content
            lines = content.split("\n")
            url = ""
            title = ""
            text_content = ""
            
            for line in lines:
                if line.startswith("URL: "):
                    url = line[5:]
                elif line.startswith("Title: "):
                    title = line[7:]
                elif line.startswith("Timestamp: "):
                    continue
                elif line.startswith("-" * 80):
                    continue
                else:
                    text_content += line + "\n"
            
            if url and text_content.strip():
                data.append((url, text_content.strip()))
                
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
    
    return data

def index_to_pinecone_with_metadata(texts):
    """Index scraped data to Pinecone with source URL metadata"""
    from utils import PineconeManager, TextEmbedder
    
    embedder = TextEmbedder()
    pinecone_manager = PineconeManager(
        api_key=st.secrets.get("PINECONE_API_KEY") or os.getenv("PINECONE_API_KEY"),
        cloud=st.secrets.get("PINECONE_ENVIRONMENT_CLOUD") or os.getenv("PINECONE_ENVIRONMENT_CLOUD") or "aws",
        region=st.secrets.get("PINECONE_ENVIRONMENT_REGION") or os.getenv("PINECONE_ENVIRONMENT_REGION"),
        index_name=st.secrets.get("PINECONE_INDEX_NAME"),
        embedder=embedder,
        dimension=1024,
    )
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    documents = []
    metadatas = []
    
    for url, content in texts:
        # Create chunks with source URL metadata
        chunks = splitter.create_documents(
            [content], 
            metadatas=[{"source_url": url, "source": "agnos_forums"}]
        )
        
        for chunk in chunks:
            documents.append(chunk.page_content)
            metadatas.append(chunk.metadata)
    
    # Upsert to Pinecone
    pinecone_manager.add_documents(documents, metadatas)
    print(f"✔️ Indexed {len(documents)} chunks to Pinecone with source URLs.")
    return pinecone_manager

async def main():
    """Main execution function"""
    print("🚀 Starting Agnos Health Forum Scraper with Anti-Bot Protection...")
    
    # Step 1: Crawl and discover all forum URLs
    await crawl_all_forums()
    
    # Step 2: Load scraped data
    texts = load_scraped_data()
    print(f"📊 Loaded {len(texts)} scraped documents")
    
    # Step 3: Index to Pinecone with metadata
    pinecone_manager = index_to_pinecone_with_metadata(texts)
    print("🎉 All done! Pinecone is ready with source URL metadata.")

if __name__ == "__main__":
    asyncio.run(main())
