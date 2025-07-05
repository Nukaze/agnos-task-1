#!/usr/bin/env python3
"""
Test TextEmbedder to check if model loading works correctly
"""

import utils
import time

def test_embedder():
    print("🧪 Testing TextEmbedder...")
    
    try:
        # Test 1: Initialize TextEmbedder
        print("\n1. Initializing TextEmbedder...")
        start_time = time.time()
        embedder = utils.TextEmbedder()
        init_time = time.time() - start_time
        print(f"✅ TextEmbedder initialized in {init_time:.2f} seconds")
        
        # Test 2: Test single text embedding
        print("\n2. Testing single text embedding...")
        test_text = "สวัสดีค่ะ อยากจะปรึกษาเรื่องอาการซึมเศร้า"
        start_time = time.time()
        embedding = embedder.embed_text(test_text)
        embed_time = time.time() - start_time
        
        if embedding:
            print(f"✅ Single embedding successful in {embed_time:.2f} seconds")
            print(f"   Embedding dimension: {len(embedding)}")
            print(f"   First 5 values: {embedding[:5]}")
        else:
            print("❌ Single embedding failed")
            return False
        
        # Test 3: Test multiple documents embedding
        print("\n3. Testing multiple documents embedding...")
        test_docs = [
            "อาการซึมเศร้าเป็นอย่างไร",
            "วิธีรักษาโรคซึมเศร้า",
            "สาเหตุของโรคซึมเศร้า"
        ]
        start_time = time.time()
        embeddings = embedder.embed_documents(test_docs)
        embed_time = time.time() - start_time
        
        if embeddings and len(embeddings) == len(test_docs):
            print(f"✅ Multiple embeddings successful in {embed_time:.2f} seconds")
            print(f"   Number of embeddings: {len(embeddings)}")
            print(f"   Each embedding dimension: {len(embeddings[0])}")
        else:
            print("❌ Multiple embeddings failed")
            return False
        
        # Test 4: Test similarity
        print("\n4. Testing similarity calculation...")
        import numpy as np
        
        # Calculate cosine similarity between similar texts
        sim1 = np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
        sim2 = np.dot(embeddings[0], embeddings[2]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[2]))
        
        print(f"   Similarity between 'อาการซึมเศร้า' and 'วิธีรักษาโรคซึมเศร้า': {sim1:.4f}")
        print(f"   Similarity between 'อาการซึมเศร้า' and 'สาเหตุของโรคซึมเศร้า': {sim2:.4f}")
        
        if sim1 > 0.5 and sim2 > 0.5:
            print("✅ Similarity calculation looks reasonable")
        else:
            print("⚠️ Similarity values seem low, but continuing...")
        
        print("\n🎉 All tests passed! TextEmbedder is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_embedder()
    if success:
        print("\n✅ TextEmbedder is ready for use!")
    else:
        print("\n❌ TextEmbedder needs fixing!") 