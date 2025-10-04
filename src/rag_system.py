import ollama
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import time
import json

class MahabharataRAG:
    def __init__(self, model_name: str = "llama2:7b-chat", use_complete_data: bool = False):
        self.llm_model = model_name
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize ChromaDB
        self.client = chromadb.EphemeralClient()
        collection_name = "complete_mahabharata" if use_complete_data else "mahabharata_dev"
        self.collection = self.client.create_collection(
            name=collection_name,
            metadata={"description": "Complete Mahabharata text" if use_complete_data else "Mahabharata development data"}
        )
        
        self._verify_llm_model()
        print(f"✅ Mahabharata RAG initialized with {model_name}")
        if use_complete_data:
            print("📚 Using complete Mahabharata (all 18 parvas)")
        else:
            print("🔬 Using development dataset (Adi Parva)")
    
    def _verify_llm_model(self):
        """Verify that the LLM model is available"""
        try:
            import ollama
            # Simple direct test
            response = ollama.list()
            
            # Handle different response formats
            if isinstance(response, dict) and 'models' in response:
                models = response['models']
            elif isinstance(response, list):
                models = response
            else:
                models = []
            
            available_models = [model['name'] for model in models]
            
            if self.llm_model in available_models:
                print(f"✅ Model {self.llm_model} is available")
                return True
            else:
                print(f"⚠️  Model {self.llm_model} not found. Available: {available_models}")
                return True  # Still return True to continue
                
        except Exception as e:
            print(f"⚠️  Could not verify models: {str(e)}")
            print("💡 Continuing anyway - Ollama might be starting up...")
            return True  # Continue anyway
    
    def add_documents(self, chunks: List[Dict]):
        """Add processed chunks to vector database"""
        if not chunks:
            print("❌ No chunks to add")
            return
        
        documents = []
        embeddings = []
        metadatas = []
        ids = []
        
        for chunk in chunks:
            documents.append(chunk['content'])
            embeddings.append(self.embedder.encode(chunk['content']).tolist())
            metadatas.append({
                'section_id': chunk['section_id'],
                'parva': chunk['parva'],
                'full_parva': chunk.get('full_parva', 'UNKNOWN'),
                'source_file': chunk.get('source_file', 'unknown'),
                'chunk_id': chunk['chunk_id'],
                'word_count': chunk['word_count']
            })
            ids.append(chunk.get('global_chunk_id', chunk['chunk_id']))
        
        # Add to collection
        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"📚 Added {len(documents)} documents to vector store")
    
    def retrieve_context(self, query: str, k: int = 3) -> List[Dict]:
        """Retrieve relevant context using semantic search"""
        query_embedding = self.embedder.encode(query).tolist()
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=['documents', 'metadatas', 'distances']
        )
        
        contexts = []
        for doc, metadata, distance in zip(
            results['documents'][0],
            results['metadatas'][0], 
            results['distances'][0]
        ):
            # Ensure similarity score is between 0 and 1
            similarity_score = max(0.0, 1 - distance)
            contexts.append({
                'content': doc,
                'metadata': metadata,
                'similarity_score': similarity_score
            })
        
        return contexts
    
    def build_prompt(self, query: str, contexts: List[Dict]) -> str:
        """Build a prompt for Llama 2"""
        if not contexts:
            return f"Question: {query}\nAnswer:"
        
        # Build context string
        context_str = "\n\n".join([
            f"SOURCE {i+1} (From {ctx['metadata']['full_parva']}, Section {ctx['metadata']['section_id']}):\n{ctx['content']}"
            for i, ctx in enumerate(contexts)
        ])
        
        prompt = f"""<s>[INST] <<SYS>>
        You are a precise scholar of the Mahabharata. Answer the question using ONLY the provided context excerpts.

        CRITICAL RULES:
        1. Only use information explicitly stated in the provided context
        2. If the context doesn't contain the answer, say "The provided text doesn't contain information about this"
        3. Do not add any external knowledge or make assumptions
        4. Be specific and cite which source your information comes from
        5. If information conflicts between sources, mention this
        <</SYS>>

        CONTEXT FROM MAHABHARATA:
        {context_str}

        QUESTION: {query}

        Provide a concise, accurate answer based strictly on the context above. Cite sources like [Source 1], [Source 2] etc. [/INST]"""
                
        return prompt
    
    def query(self, question: str, k: int = 3) -> Dict[str, Any]:
        """Main query method"""
        start_time = time.time()
        
        # Retrieve context
        contexts = self.retrieve_context(question, k)
        retrieval_time = time.time() - start_time
        
        if not contexts:
            return {
                'answer': "I couldn't find relevant information in the available text.",
                'sources': [],
                'confidence': 0.0,
                'timing': {
                    'retrieval': retrieval_time,
                    'total': retrieval_time
                }
            }
        
        # Build prompt and generate response
        prompt = self.build_prompt(question, contexts)
        
        try:
            gen_start = time.time()
            response = ollama.generate(
                model=self.llm_model,
                prompt=prompt,
                options={
                    'temperature': 0.1,
                    'top_k': 20,
                    'num_predict': 300  # Limit response length
                }
            )
            generation_time = time.time() - gen_start
            
            answer = response['response']
            
        except Exception as e:
            answer = f"Error generating response: {str(e)}"
            generation_time = 0
        
        total_time = time.time() - start_time
        
        # Calculate confidence (ensure it's between 0 and 1)
        confidence_scores = [max(0.0, ctx['similarity_score']) for ctx in contexts]
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if contexts else 0.0
        
        return {
            'answer': answer,
            'sources': contexts,
            'confidence': avg_confidence,
            'timing': {
                'retrieval': retrieval_time,
                'generation': generation_time,
                'total': total_time
            },
            'sources_count': len(contexts)
        }
    
    def get_knowledge_stats(self):
        """Get statistics about the loaded knowledge base"""
        try:
            # Get collection info
            count = self.collection.count()
            return {
                'total_chunks': count,
                'model': self.llm_model,
                'embedding_model': 'all-MiniLM-L6-v2'
            }
        except:
            return {'total_chunks': 0, 'model': self.llm_model}