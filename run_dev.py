#!/usr/bin/env python3
"""
Robust Mahabharata RAG Runner - Handles Ollama issues
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_processor import MahabharataDataProcessor
from src.rag_system import MahabharataRAG
import json
import time
import subprocess

def is_ollama_running():
    """Check if Ollama is running"""
    try:
        import ollama
        models = ollama.list()
        return True
    except:
        return False

def start_ollama_background():
    """Start Ollama in background"""
    print("🚀 Starting Ollama in background...")
    try:
        subprocess.Popen(['ollama', 'serve'], 
                        stdout=subprocess.DEVNULL, 
                        stderr=subprocess.DEVNULL)
        time.sleep(8)  # Wait for startup
        return True
    except:
        return False

def main():
    print("=" * 60)
    print("🛡️  ROBUST MAHABHARATA RAG RUNNER")
    print("=" * 60)
    
    # Check if Ollama is running
    if not is_ollama_running():
        print("❌ Ollama is not running")
        print("💡 Attempting to start Ollama automatically...")
        if start_ollama_background():
            print("✅ Ollama started successfully")
        else:
            print("❌ Failed to start Ollama automatically")
            print("💡 Please manually run: ollama serve")
            return
    
    # Configuration
    INPUT_FILE = "data/raw/maha01.txt"
    OUTPUT_FILE = "data/processed/chunks.json"
    MAX_CHUNKS = 30  # Reduced for faster testing
    
    # Step 1: Process Data
    print("\n📖 STEP 1: Processing/Loading Data...")
    
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            print(f"✅ Loaded {len(chunks)} pre-processed chunks")
        except Exception as e:
            print(f"❌ Error loading processed data: {e}")
            return
    else:
        if not os.path.exists(INPUT_FILE):
            print(f"❌ Input file not found: {INPUT_FILE}")
            return
        
        processor = MahabharataDataProcessor()
        chunks = processor.process_file(INPUT_FILE, OUTPUT_FILE, MAX_CHUNKS)
        if not chunks:
            print("❌ No chunks processed. Exiting.")
            return
    
    # Step 2: Initialize RAG System
    print("\n🤖 STEP 2: Initializing RAG System...")
    try:
        rag = MahabharataRAG()
    except Exception as e:
        print(f"❌ Failed to initialize RAG: {e}")
        return
    
    # Step 3: Add documents
    print("\n📚 STEP 3: Building Knowledge Base...")
    rag.add_documents(chunks)
    
    # Step 4: Quick test
    print("\n🧪 STEP 4: Quick Test...")
    test_questions = [
        "Who are the Pandavas?",
        "What is the Mahabharata about?"
    ]
    
    for question in test_questions:
        print(f"\n❓ {question}")
        try:
            result = rag.query(question)
            print(f"🤖 {result['answer'][:200]}...")
            print(f"📊 Confidence: {result['confidence']:.3f} | ⏱️  {result['timing']['total']:.2f}s")
        except Exception as e:
            print(f"❌ Query failed: {e}")
    
    print("\n" + "=" * 60)
    print("✅ ROBUST RUN COMPLETED!")
    print("=" * 60)

if __name__ == "__main__":
    main()