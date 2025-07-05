#!/usr/bin/env python3
"""
Batch Forum Data Processor
Processes multiple forum files and upserts to Pinecone in batches
"""

import os
import json
import re
import glob
from datetime import datetime
import utils
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st

# Configuration
INPUT_DIR = "data_scraped_txt"
OUTPUT_DIR = "processed_data"
BATCH_SIZE = 20  # Process documents in batches

prepare_batch_upsert_source = []

def extract_simple_records(content: str) -> list:
    """Extract records by splitting on double newlines and processing each section"""
    records = []
    
    # Split content by double newlines to get individual posts
    sections = content.split('\n\n')
    
    for i, section in enumerate(sections):
        if not section.strip() or len(section.strip()) < 50:
            continue
        
        # Clean the section
        clean_section = section.strip()
        
        # Extract basic information
        record = {
            "content": clean_section,
            "section_index": i,
            "word_count": len(clean_section.split()),
            "has_url": "https://" in clean_section,
            "has_date": bool(re.search(r'\d{1,2}/\d{1,2}/\d{4}', clean_section))
        }
        
        # Try to extract URL
        url_match = re.search(r'https://[^\s]+', clean_section)
        if url_match:
            record["forum_url"] = url_match.group(0)
        else:
            record["forum_url"] = ""
        
        # Try to extract date
        date_match = re.search(r'(\d{1,2}/\d{1,2}/\d{4})', clean_section)
        if date_match:
            record["forum_posted_timestamp"] = date_match.group(1)
        else:
            record["forum_posted_timestamp"] = ""
        
        # Try to extract headline (look for disease names in parentheses)
        headline_match = re.search(r'\(([^)]+)\)', clean_section)
        if headline_match:
            record["forum_headline"] = headline_match.group(1)
        else:
            # Look for Thai disease names
            thai_disease_match = re.search(r'([ก-๙]+(?:\s+[ก-๙]+)*)', clean_section[:200])
            if thai_disease_match:
                record["forum_headline"] = thai_disease_match.group(1)
            else:
                record["forum_headline"] = "ไม่ระบุ"
        
        # Count replies (look for reply indicators)
        reply_indicators = ["ตอบ", "reply", "comment", "แพทย์ตอบ"]
        reply_count = sum(1 for indicator in reply_indicators if indicator in clean_section.lower())
        record["forum_reply_count"] = reply_count
        
        records.append(record)
    
    return records

def process_file(filepath: str) -> list:
    """Process a single forum file"""
    print(f"📄 Processing file: {os.path.basename(filepath)}")
    
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        
        records = extract_simple_records(content)
        print(f"✅ Extracted {len(records)} records from {os.path.basename(filepath)}")
        return records
        
    except Exception as e:
        print(f"❌ Error processing {filepath}: {e}")
        return []

def upsert_batch(documents: list, metadatas: list, pinecone_manager):
    """Upsert a batch of documents to Pinecone"""
    if not documents:
        return 0
    
    try:
        pinecone_manager.add_documents(documents, metadatas)
        print(f"✅ Upserted batch of {len(documents)} documents")
        return len(documents)
    except Exception as e:
        print(f"❌ Error upserting batch: {e}")
        return 0

def get_clean_source_forum_url(data):
    prefix_part = "https://www.agnoshealth.com/forums/"
    
    if prefix_part not in data:
        return "prefix source forum url not found"
    try:
        prefix_index = data.index(prefix_part)
        data = data[prefix_index:]
        
        suffix_part = ")![thumbs-up]"
        suffix_index = data.index(suffix_part)
        stripped_source_url = data[:suffix_index]

        return stripped_source_url
    except Exception as e:
        return "error when clean source forum url"

def batch_process_forums():
    """Process all forum files in batches"""
    print("🚀 Starting batch forum processing...")
    
    # Find all forum files
    forum_files = glob.glob(os.path.join(INPUT_DIR, "*.txt"))
    print(f"📁 Found {len(forum_files)} forum files")
    
    if not forum_files:
        print(f"❌ No forum files found in {INPUT_DIR}")
        return
    
    # Initialize Pinecone
    try:
        embedder = utils.TextEmbedder()
        pinecone_manager = utils.PineconeManager(
            api_key=st.secrets.get("PINECONE_API_KEY") or os.getenv("PINECONE_API_KEY"),
            cloud=st.secrets.get("PINECONE_ENVIRONMENT_CLOUD") or os.getenv("PINECONE_ENVIRONMENT_CLOUD") or "aws",
            region=st.secrets.get("PINECONE_ENVIRONMENT_REGION") or os.getenv("PINECONE_ENVIRONMENT_REGION"),
            # index_name=st.secrets.get("PINECONE_INDEX_NAME") or os.getenv("PINECONE_INDEX_NAME") or "agnos-forums",
            index_name="agnos-forum-firecrawl",
            embedder=embedder,
            dimension=1024,
        )
        print("✅ Pinecone manager initialized")
    except Exception as e:
        print(f"❌ Error initializing Pinecone: {e}")
        return
    
    # Initialize text splitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    
    # Process all files
    all_records = []
    for filepath in forum_files:
        records = process_file(filepath)
        all_records.extend(records)
    
    print(f"📊 Total records extracted: {len(all_records)}")
    
    # Process records in batches
    documents = []
    metadatas = []
    total_upserted = 0
    batch_count = 0
    
    for i, record in enumerate(all_records):
        # Skip records with too little content
        if record["word_count"] < 20:
            continue
        
        # Skip records without forum URL
        if not record["forum_url"]:
            print(f"⚠️ Skipping record {i} - no forum URL found")
            continue
        
        # Debug: Print the URL being processed
        print(f"🔍 Processing URL: {record['forum_url'][:100]}...")
        
        clean_source = get_clean_source_forum_url(record["forum_url"])
        
        print(f"✅ Cleaned URL: {clean_source}")
        
        # Skip if cleaning failed
        if not clean_source:
            print(f"⚠️ Skipping record {i} - URL cleaning failed")
            continue
        
        if clean_source in prepare_batch_upsert_source:
            # if clean_source already in upserted source
            continue
        
        # Create metadata
        metadata = {
            "source": clean_source,
            "info": record["forum_url"],
            # "forum_headline": record["forum_headline"],
            "forum_posted_timestamp": record["forum_posted_timestamp"],
            "forum_reply_count": record["forum_reply_count"],
            "scraped_at": datetime.now().isoformat(),
            "record_index": i,
        }
        
        # Split content into chunks
        try:
            chunks = splitter.create_documents([record["content"]], metadatas=[metadata])
            
            for chunk in chunks:
                documents.append(chunk.page_content)
                metadatas.append(chunk.metadata)
            
            prepare_batch_upsert_source.append(clean_source)
            
            
            # Upsert batch if we have enough documents
            if len(documents) >= BATCH_SIZE:
                batch_count += 1
                print(f"🔄 Processing batch {batch_count} ({len(documents)} documents)")
                upserted = upsert_batch(documents, metadatas, pinecone_manager)
                
                total_upserted += upserted
                documents = []
                metadatas = []
                
        except Exception as e:
            print(f"❌ Error processing record {i}: {e}")
            continue
    
    # Upsert remaining documents
    if documents:
        batch_count += 1
        print(f"🔄 Processing final batch {batch_count} ({len(documents)} documents)")
        upserted = upsert_batch(documents, metadatas, pinecone_manager)
        total_upserted += upserted
    
    # Save summary
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary_file = os.path.join(OUTPUT_DIR, "batch_processing_summary.json")
    
    summary = {
        "total_files": len(forum_files),
        "total_records": len(all_records),
        "total_documents_upserted": total_upserted,
        "batches_processed": batch_count,
        "processed_at": datetime.now().isoformat(),
        "input_directory": INPUT_DIR,
        "sample_records": all_records[:] if all_records else []
    }
    
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"📊 Summary saved to {summary_file}")
    print(f"🎉 Batch processing completed! Upserted {total_upserted} documents in {batch_count} batches")

def main():
    """Main execution function"""
    print("🔍 Batch Forum Data Processor")
    print("=" * 50)
    
    # Check if input directory exists
    if not os.path.exists(INPUT_DIR):
        print(f"❌ Input directory not found: {INPUT_DIR}")
        return
    
    # Process forums in batches
    batch_process_forums()

if __name__ == "__main__":
    main() 