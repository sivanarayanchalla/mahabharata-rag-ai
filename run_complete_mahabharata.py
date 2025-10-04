#!/usr/bin/env python3
"""
Complete Mahabharata RAG Runner - Uses all 18 parvas
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.multi_file_processor import process_complete_mahabharata
from src.rag_system import MahabharataRAG
import json
import time

def main():
    print("=" * 60)
    print("🏛️  COMPLETE MAHABHARATA RAG SYSTEM")
    print("=" * 60)
    
    # Configuration
    COMPLETE_DATA_FILE = "data/processed/complete_mahabharata.json"
    MAX_CHUNKS = 400  # Use more chunks for better coverage
    
    # Step 1: Process all files (if not already done)
    if not os.path.exists(COMPLETE_DATA_FILE):
        print("\n📚 STEP 1: Processing Complete Mahabharata...")
        print("This may take a few minutes...")
        
        chunks = process_complete_mahabharata(max_chunks_per_file=25)
        if not chunks:
            print("❌ Failed to process Mahabharata files")
            return
    else:
        print("\n📚 STEP 1: Loading pre-processed complete Mahabharata...")
        try:
            with open(COMPLETE_DATA_FILE, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            print(f"✅ Loaded {len(chunks)} chunks from complete Mahabharata")
        except Exception as e:
            print(f"❌ Error loading complete data: {e}")
            return
    
    # Step 2: Initialize RAG with complete data
    print("\n🤖 STEP 2: Initializing Complete RAG System...")
    rag = MahabharataRAG(use_complete_data=True)
    
    # Step 3: Add all documents
    print("\n📖 STEP 3: Building Complete Knowledge Base...")
    if len(chunks) > MAX_CHUNKS:
        print(f"🔬 Using {MAX_CHUNKS} chunks for performance")
        chunks = chunks[:MAX_CHUNKS]
    
    rag.add_documents(chunks)
    
    # Step 4: Comprehensive testing
    print("\n🧪 STEP 4: Comprehensive Testing...")
    
    test_queries = [
        {
            "question": "Who are the five Pandava brothers and their qualities?",
            "category": "Characters"
        },
        {
            "question": "What is the Bhagavad Gita and what does it teach?",
            "category": "Philosophy"
        },
        {
            "question": "Describe the Kurukshetra war and its causes",
            "category": "Events"
        },
        {
            "question": "What is Dharma according to the Mahabharata?",
            "category": "Concepts"
        },
        {
            "question": "Tell me about Krishna's role as Arjuna's charioteer",
            "category": "Key Scenes"
        }
    ]
    
    for i, test in enumerate(test_queries, 1):
        print(f"\n{'─' * 60}")
        print(f"TEST {i} [{test['category']}]: {test['question']}")
        print(f"{'─' * 60}")
        
        start_time = time.time()
        result = rag.query(test['question'], k=6)  # More sources for better answers
        query_time = time.time() - start_time
        
        print(f"🤖 ANSWER: {result['answer']}")
        print(f"📊 Confidence: {result['confidence']:.3f}")
        print(f"⏱️  Time: {query_time:.2f}s")
        print(f"📚 Sources: {result['sources_count']}")
        
        # Show source distribution
        if result['sources']:
            sources_by_parva = {}
            for source in result['sources']:
                parva = source['metadata']['full_parva']
                sources_by_parva[parva] = sources_by_parva.get(parva, 0) + 1
            
            print(f"📖 Sources from: {', '.join(sources_by_parva.keys())}")
    
    print("\n" + "=" * 60)
    print("🎉 COMPLETE MAHABHARATA RAG READY!")
    print("=" * 60)
    
    # Interactive mode with enhanced capabilities
    print("\n💬 INTERACTIVE MODE: Ask about any Mahabharata topic!")
    print("   You can now ask about:")
    print("   • Characters from all 18 parvas")
    print("   • Philosophical teachings") 
    print("   • Historical events")
    print("   • Moral dilemmas")
    print("   • And much more...")
    
    while True:
        user_question = input("\n❓ Your question (or 'quit' to exit): ").strip()
        if user_question.lower() in ['quit', 'exit', 'q']:
            break
        if user_question:
            print("⏳ Searching through all 18 parvas...")
            start_time = time.time()
            result = rag.query(user_question, k=6)
            response_time = time.time() - start_time
            
            print(f"\n🤖 {result['answer']}")
            print(f"\n📊 Confidence: {result['confidence']:.3f} | ⏱️  {response_time:.2f}s | 📚 {result['sources_count']} sources")
            
            # Show which parvas were used
            if result['sources']:
                unique_parvas = set(s['metadata']['full_parva'] for s in result['sources'])
                print(f"📖 Information from: {', '.join(unique_parvas)}")

if __name__ == "__main__":
    main()