#!/usr/bin/env python3
"""
Improved Quick Test with Better Retrieval
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
import time

def improved_quick_test():
    print("⚡ IMPROVED MAHABHARATA RAG TEST")
    print("=" * 40)
    
    try:
        from src.rag_system import MahabharataRAG
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return
    
    # Load existing chunks
    try:
        with open("data/processed/chunks.json", 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        print(f"✅ Loaded {len(chunks)} chunks")
    except Exception as e:
        print(f"❌ Error loading chunks: {e}")
        return
    
    # Initialize RAG with better settings
    try:
        rag = MahabharataRAG()
        print("✅ RAG system initialized")
    except Exception as e:
        print(f"❌ RAG initialization failed: {e}")
        return
    
    # Add documents (use more chunks for better coverage)
    rag.add_documents(chunks[:30])  # Increased from 20 to 30
    
    # Test multiple questions with different retrieval strategies
    test_questions = [
        "Who is the author of Mahabharata?",
        "Who are the Pandava brothers?",
        "What is the Adi Parva about?",
        "Tell me about Krishna in Mahabharata"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'─' * 50}")
        print(f"TEST {i}: {question}")
        print(f"{'─' * 50}")
        
        start_time = time.time()
        
        # Try with more sources for better answers
        result = rag.query(question, k=5)  # Increased from 3 to 5 sources
        
        query_time = time.time() - start_time
        
        print(f"🤖 ANSWER: {result['answer']}")
        print(f"📊 Confidence: {result['confidence']:.3f}")
        print(f"⏱️  Time: {query_time:.2f}s")
        print(f"📚 Sources used: {result['sources_count']}")
        
        # Show top source info
        if result['sources']:
            best_source = result['sources'][0]
            print(f"🔝 Best source: {best_source['metadata']['parva']} Section {best_source['metadata']['section_id']}")
            print(f"   Similarity: {best_source['similarity_score']:.3f}")

if __name__ == "__main__":
    improved_quick_test()