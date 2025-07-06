#!/usr/bin/env python3
"""
Test TextEmbedder fix for meta tensor issue
"""

import utils
import time
import traceback

def test_embedder_fix():
    print("🧪 Testing TextEmbedder fix for meta tensor issue...")
    
    try:
        # Test 1: Initialize TextEmbedder
        print("\n1. Initializing TextEmbedder...")
        start_time = time.time()
        
        try:
            embedder = utils.TextEmbedder()
            init_time = time.time() - start_time
            print(f"✅ TextEmbedder initialized in {init_time:.2f} seconds")
        except Exception as init_error:
            print(f"❌ Failed to initialize TextEmbedder: {init_error}")
            traceback.print_exc()
            return False
        
        # Test 2: Test single text embedding
        print("\n2. Testing single text embedding...")
        test_text = "สวัสดีค่ะ อยากจะปรึกษาเรื่องอาการซึมเศร้า"
        start_time = time.time()
        
        try:
            embedding = embedder.embed_text(test_text)
            embed_time = time.time() - start_time
            
            if embedding:
                print(f"✅ Single embedding successful in {embed_time:.2f} seconds")
                print(f"   Embedding dimension: {len(embedding)}")
                print(f"   First 5 values: {embedding[:5]}")
            else:
                print("❌ Single embedding returned empty result")
                return False
        except Exception as embed_error:
            print(f"❌ Single embedding failed: {embed_error}")
            traceback.print_exc()
            return False
        
        # Test 3: Test multiple documents embedding
        print("\n3. Testing multiple documents embedding...")
        test_docs = [
            "อาการซึมเศร้าเป็นอย่างไร",
            "วิธีรักษาโรคซึมเศร้า",
            "สาเหตุของโรคซึมเศร้า"
        ]
        start_time = time.time()
        
        try:
            embeddings = embedder.embed_documents(test_docs)
            embed_time = time.time() - start_time
            
            if embeddings and len(embeddings) == len(test_docs):
                print(f"✅ Multiple embeddings successful in {embed_time:.2f} seconds")
                print(f"   Number of embeddings: {len(embeddings)}")
                print(f"   Each embedding dimension: {len(embeddings[0])}")
            else:
                print("❌ Multiple embeddings failed or returned wrong number")
                return False
        except Exception as multi_error:
            print(f"❌ Multiple embeddings failed: {multi_error}")
            traceback.print_exc()
            return False
        
        print("\n🎉 All tests passed! TextEmbedder fix is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_embedder_fix()
    if success:
        print("\n✅ TextEmbedder fix is successful!")
    else:
        print("\n❌ TextEmbedder still needs fixing!") 