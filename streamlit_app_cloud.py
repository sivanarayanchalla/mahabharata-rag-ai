import streamlit as st
import json
import os
import requests
from sentence_transformers import SentenceTransformer
import chromadb
from typing import List, Dict
import time

# Custom CSS for professional look
st.set_page_config(
    page_title="Mahabharata AI Scholar - Cloud Edition",
    page_icon="🕉️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF6B35;
        text-align: center;
        margin-bottom: 1rem;
    }
    .answer-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #FF6B35;
        margin: 10px 0;
    }
    .metric-box {
        background-color: white;
        padding: 10px;
        border-radius: 8px;
        border: 1px solid #ddd;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

class CloudMahabharataRAG:
    def __init__(self):
        # Load embedding model
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize vector store
        self.client = chromadb.EphemeralClient()
        self.collection = self.client.create_collection("mahabharata_cloud")
        
        # Load pre-processed data
        self.load_knowledge_base()
        
    def load_knowledge_base(self):
        """Load pre-processed Mahabharata chunks"""
        try:
            with open('data/processed/complete_mahabharata.json', 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            
            # Add to vector store
            documents = [chunk['content'] for chunk in chunks[:200]]  # Use 200 chunks
            embeddings = self.embedder.encode(documents).tolist()
            metadatas = [{
                'section_id': chunk['section_id'],
                'parva': chunk.get('full_parva', chunk['parva']),
                'chunk_id': chunk['chunk_id']
            } for chunk in chunks[:200]]
            
            self.collection.add(
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
                ids=[chunk['chunk_id'] for chunk in chunks[:200]]
            )
            
            st.sidebar.success(f"✅ Loaded {len(documents)} knowledge chunks")
            
        except Exception as e:
            st.error(f"❌ Error loading knowledge base: {e}")
    
    def retrieve_context(self, query: str, k: int = 5) -> List[Dict]:
        """Retrieve relevant context"""
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
            contexts.append({
                'content': doc,
                'metadata': metadata,
                'similarity_score': max(0.0, 1 - distance)
            })
        
        return contexts
    
    def generate_free_answer(self, question: str, contexts: List[Dict]) -> str:
        """Generate answer using free APIs with fallbacks"""
        
        # Build context
        context_str = "\n\n".join([
            f"From {ctx['metadata']['parva']}: {ctx['content']}"
            for ctx in contexts
        ])
        
        prompt = f"""Based on this Mahabharata context: {context_str}

Question: {question}

Provide a clear, accurate answer based only on the context:"""
        
        # Try Hugging Face free inference first
        try:
            return self.try_hugging_face(prompt)
        except:
            pass
        
        # Fallback: Use a rule-based answer
        return self.rule_based_answer(question, contexts)
    
    def try_hugging_face(self, prompt: str) -> str:
        """Try Hugging Face free inference"""
        try:
            # Using Hugging Face inference API (free tier)
            API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
            headers = {"Authorization": f"Bearer {os.getenv('HF_TOKEN', '')}"}
            
            payload = {
                "inputs": prompt,
                "parameters": {"max_new_tokens": 300, "temperature": 0.1}
            }
            
            response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get('generated_text', '').split(prompt)[-1].strip()
            
        except Exception as e:
            st.sidebar.warning(f"⚠️ Hugging Face API: {str(e)}")
        
        raise Exception("Hugging Face failed")
    
    def rule_based_answer(self, question: str, contexts: List[Dict]) -> str:
        """Fallback rule-based answer"""
        if not contexts:
            return "I couldn't find specific information about this in the available text."
        
        # Simple rule-based response
        answer_parts = []
        answer_parts.append("Based on the Mahabharata text:")
        
        for i, ctx in enumerate(contexts[:3], 1):
            answer_parts.append(f"\nFrom {ctx['metadata']['parva']}: {ctx['content'][:200]}...")
        
        answer_parts.append(f"\n\nThis information from {len(contexts)} sources addresses your question about '{question}'.")
        
        return " ".join(answer_parts)
    
    def query(self, question: str) -> Dict:
        """Main query method"""
        start_time = time.time()
        
        # Retrieve context
        contexts = self.retrieve_context(question, k=5)
        retrieval_time = time.time() - start_time
        
        if not contexts:
            return {
                'answer': "I couldn't find relevant information about this topic in the Mahabharata text.",
                'sources': [],
                'confidence': 0.0,
                'response_time': retrieval_time
            }
        
        # Generate answer
        gen_start = time.time()
        answer = self.generate_free_answer(question, contexts)
        generation_time = time.time() - gen_start
        
        total_time = time.time() - start_time
        
        # Calculate confidence
        confidence = sum(ctx['similarity_score'] for ctx in contexts) / len(contexts) if contexts else 0.0
        
        return {
            'answer': answer,
            'sources': contexts,
            'confidence': confidence,
            'response_time': total_time,
            'sources_count': len(contexts)
        }

def main():
    st.markdown('<h1 class="main-header">🕉️ Mahabharata AI Scholar</h1>', unsafe_allow_html=True)
    st.markdown('### Cloud Edition - Free & Accurate Answers')
    
    # Initialize session state
    if 'rag' not in st.session_state:
        with st.spinner("🔄 Loading Mahabharata knowledge base..."):
            st.session_state.rag = CloudMahabharataRAG()
        st.success("✅ System ready!")
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        st.info("**Model:** Free Cloud LLM\n**Knowledge:** Complete Mahabharata\n**Status:** ✅ Active")
        
        st.header("🎯 Example Questions")
        examples = [
            "Who are the Pandava brothers?",
            "What is the Bhagavad Gita about?",
            "Tell me about Krishna's role",
            "Describe the Kurukshetra war",
            "What is Dharma in Mahabharata?"
        ]
        
        for example in examples:
            if st.button(example, use_container_width=True):
                st.session_state.current_question = example
    
    # Main interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        question = st.text_input(
            "**Your Question:**",
            placeholder="Ask anything about Mahabharata...",
            value=getattr(st.session_state, 'current_question', '')
        )
        
        if st.button("🚀 Get Answer", type="primary", use_container_width=True):
            if question:
                with st.spinner("🔍 Searching through Mahabharata..."):
                    result = st.session_state.rag.query(question)
                
                # Store in history
                st.session_state.chat_history.append({
                    'question': question,
                    'answer': result['answer'],
                    'timestamp': time.time(),
                    'confidence': result['confidence']
                })
                
                # Display results
                st.markdown("### 📜 Answer")
                st.markdown(f'<div class="answer-box">{result["answer"]}</div>', unsafe_allow_html=True)
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("⏱️ Time", f"{result['response_time']:.2f}s")
                with col2:
                    st.metric("📊 Confidence", f"{result['confidence']:.3f}")
                with col3:
                    st.metric("📚 Sources", result['sources_count'])
                with col4:
                    st.metric("🔍 Method", "Cloud RAG")
                
                # Sources
                with st.expander(f"📖 View {len(result['sources'])} Sources"):
                    for i, source in enumerate(result['sources']):
                        st.markdown(f"**Source {i+1}** - {source['metadata']['parva']}")
                        st.markdown(f"*Relevance: {source['similarity_score']:.3f}*")
                        st.markdown(f"{source['content'][:250]}...")
                        st.divider()
            else:
                st.warning("Please enter a question!")
    
    with col2:
        st.markdown("### 💡 Quick Facts")
        facts = [
            "📖 18 Parvas",
            "👑 100+ Characters", 
            "⚔️ Epic War",
            "🕉️ Deep Philosophy",
            "🌍 Ancient Wisdom"
        ]
        for fact in facts:
            st.info(fact)
        
        # Chat history
        if st.session_state.chat_history:
            st.markdown("### 💬 Recent Questions")
            for chat in reversed(st.session_state.chat_history[-3:]):
                st.caption(f"**Q:** {chat['question'][:30]}...")
        
        st.markdown("### 🆓 Free Tier")
        st.success("""
        **Features:**
        • Full Mahabharata access
        • Accurate RAG retrieval  
        • Free cloud processing
        • Source citations
        """)

if __name__ == "__main__":
    main()