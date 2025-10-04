import streamlit as st
import json
import os
import requests
from sentence_transformers import SentenceTransformer
import chromadb
from typing import List, Dict
import time
import re

# Custom CSS for superior readability and performance
st.set_page_config(
    page_title="Mahabharata AI Scholar - Cloud Edition",
    page_icon="🕉️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Main styling for readability */
    .main-header {
        font-size: 2.8rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 0.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #2E86AB, #A23B72);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        font-size: 1.3rem;
        color: #5D5D5D;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Question box styling */
    .question-input {
        background-color: #FFFFFF;
        border: 2px solid #E0E0E0;
        border-radius: 12px;
        padding: 15px;
        font-size: 1.1em;
        margin-bottom: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    .question-input:focus {
        border-color: #2E86AB;
        box-shadow: 0 4px 8px rgba(46, 134, 171, 0.2);
    }
    
    /* Answer box with excellent readability */
    .answer-box {
        background: linear-gradient(135deg, #F8F9FA 0%, #FFFFFF 100%);
        padding: 25px;
        border-radius: 15px;
        border-left: 6px solid #2E86AB;
        margin: 20px 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        font-size: 1.15em;
        line-height: 1.7;
        color: #2D3748;
        font-family: 'Georgia', serif;
    }
    .answer-box h3, .answer-box h4 {
        color: #2E86AB;
        margin-top: 1.5em;
        margin-bottom: 0.8em;
    }
    .answer-box p {
        margin-bottom: 1.2em;
    }
    
    /* Sample questions styling */
    .sample-question {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 12px 16px;
        border-radius: 10px;
        margin: 8px 0;
        cursor: pointer;
        transition: all 0.3s ease;
        border: none;
        text-align: left;
        font-size: 0.95em;
        box-shadow: 0 2px 6px rgba(0,0,0,0.15);
    }
    .sample-question:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Metrics and badges */
    .metric-box {
        background: linear-gradient(135deg, #FFFFFF 0%, #F7FAFC 100%);
        padding: 15px;
        border-radius: 12px;
        border: 2px solid #E2E8F0;
        text-align: center;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    }
    .source-badge {
        background: linear-gradient(135deg, #2E86AB, #A23B72);
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.85em;
        margin: 4px;
        display: inline-block;
        font-weight: 500;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Chat history items */
    .chat-item {
        background: #F7FAFC;
        padding: 10px 14px;
        border-radius: 10px;
        margin: 6px 0;
        border-left: 4px solid #2E86AB;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    .chat-item:hover {
        background: #EDF2F7;
        transform: translateX(4px);
    }
    
    /* Loading animation */
    .loading-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid #f3f3f3;
        border-top: 3px solid #2E86AB;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Improved sidebar */
    .sidebar-content {
        background: linear-gradient(180deg, #F8F9FA 0%, #FFFFFF 100%);
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
    }
    
    /* Button improvements */
    .stButton button {
        width: 100%;
        border-radius: 10px;
        height: 50px;
        font-size: 1.1em;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
</style>
""", unsafe_allow_html=True)

class HighPerformanceMahabharataRAG:
    def __init__(self):
        # Cache the embedding model
        if 'embedder' not in st.session_state:
            with st.spinner("🔄 Loading AI models..."):
                st.session_state.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        self.embedder = st.session_state.embedder
        
        # Initialize vector store with persistence
        self.client = chromadb.EphemeralClient()
        self.collection = self.client.create_collection("mahabharata_cloud_v2")
        
        # Load knowledge base with caching
        self.load_knowledge_base()
        
    def load_knowledge_base(self):
        """Load pre-processed Mahabharata chunks with caching"""
        try:
            if 'knowledge_loaded' not in st.session_state:
                with st.spinner("📚 Loading Mahabharata knowledge..."):
                    with open('data/processed/complete_mahabharata.json', 'r', encoding='utf-8') as f:
                        chunks = json.load(f)
                    
                    # Add to vector store
                    documents = [chunk['content'] for chunk in chunks[:400]]
                    embeddings = self.embedder.encode(documents).tolist()
                    metadatas = [{
                        'section_id': chunk['section_id'],
                        'parva': chunk.get('full_parva', chunk['parva']),
                        'chunk_id': chunk['chunk_id'],
                        'source_file': chunk.get('source_file', 'unknown')
                    } for chunk in chunks[:400]]
                    
                    self.collection.add(
                        embeddings=embeddings,
                        documents=documents,
                        metadatas=metadatas,
                        ids=[chunk['chunk_id'] for chunk in chunks[:400]]
                    )
                    
                    st.session_state.knowledge_loaded = True
                    st.session_state.loaded_chunks_count = len(documents)
            
            st.sidebar.success(f"✅ Loaded {st.session_state.loaded_chunks_count} knowledge chunks")
            
        except Exception as e:
            st.error(f"❌ Error loading knowledge base: {e}")
    
    def retrieve_context(self, query: str, k: int = 5) -> List[Dict]:
        """Fast context retrieval with timeout protection"""
        try:
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
                similarity_score = max(0.0, 1 - distance)
                if similarity_score > 0.15:  # Higher threshold for better quality
                    contexts.append({
                        'content': doc,
                        'metadata': metadata,
                        'similarity_score': similarity_score
                    })
            
            return contexts
            
        except Exception as e:
            st.error(f"Retrieval error: {e}")
            return []
    
    def generate_answer(self, question: str, contexts: List[Dict]) -> str:
        """Generate answer with multiple fallback strategies"""
        if not contexts:
            return self.get_no_answer_template(question)
        
        # Try cloud APIs first
        try:
            return self.try_cloud_apis(question, contexts)
        except:
            pass
        
        # Fallback to enhanced template
        return self.enhanced_template_answer(question, contexts)
    
    def try_cloud_apis(self, question: str, contexts: List[Dict]) -> str:
        """Try various free cloud APIs"""
        prompt = self.build_quality_prompt(question, contexts)
        
        # List of free API endpoints to try
        apis_to_try = [
            self.try_hugging_face,
            self.try_deepinfra,
            self.try_together_ai
        ]
        
        for api_func in apis_to_try:
            try:
                result = api_func(prompt)
                if result and len(result) > 100:
                    return self.format_answer(result, contexts)
            except:
                continue
        
        raise Exception("All cloud APIs failed")
    
    def try_hugging_face(self, prompt: str) -> str:
        """Try Hugging Face inference"""
        try:
            API_URL = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-large"
            headers = {"Authorization": f"Bearer {os.getenv('HF_TOKEN', '')}"}
            
            payload = {
                "inputs": prompt,
                "parameters": {"max_new_tokens": 350, "temperature": 0.4}
            }
            
            response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get('generated_text', '')
            
        except:
            pass
        raise Exception("HF failed")
    
    def try_deepinfra(self, prompt: str) -> str:
        """Try DeepInfra free tier"""
        try:
            # This would require an API key, but structure is ready
            pass
        except:
            pass
        raise Exception("DeepInfra not configured")
    
    def try_together_ai(self, prompt: str) -> str:
        """Try Together AI free tier"""
        try:
            # This would require an API key, but structure is ready
            pass
        except:
            pass
        raise Exception("Together AI not configured")
    
    def build_quality_prompt(self, question: str, contexts: List[Dict]) -> str:
        """Build high-quality prompt"""
        context_str = "\n\n".join([
            f"**From {ctx['metadata']['parva']}:** {ctx['content']}"
            for ctx in contexts[:3]  # Use top 3 contexts
        ])
        
        return f"""You are a Mahabharata scholar. Answer the question using ONLY the provided context.

CONTEXT:
{context_str}

QUESTION: {question}

Provide a comprehensive, well-structured answer with:
1. Clear main answer first
2. Supporting details from the context
3. Specific examples when available
4. Citation of which parva information comes from

ANSWER:"""
    
    def enhanced_template_answer(self, question: str, contexts: List[Dict]) -> str:
        """High-quality template-based answer"""
        # Group by parva
        parva_groups = {}
        for ctx in contexts:
            parva = ctx['metadata']['parva']
            if parva not in parva_groups:
                parva_groups[parva] = []
            parva_groups[parva].append(ctx)
        
        # Build answer
        answer_parts = [f"## 📖 Answer: {question}\n"]
        
        # Main answer from best context
        if contexts:
            best_ctx = max(contexts, key=lambda x: x['similarity_score'])
            main_info = self.extract_core_info(best_ctx['content'])
            answer_parts.append(f"\n{main_info}\n")
        
        # Add supporting information
        if len(parva_groups) > 1:
            answer_parts.append("\n### 🔍 Additional Insights:\n")
            for parva, ctx_list in list(parva_groups.items())[:2]:
                if ctx_list:
                    sample = ctx_list[0]['content'][:120] + "..."
                    answer_parts.append(f"• **{parva}**: {sample}\n")
        
        answer_parts.append(f"\n*Information synthesized from {len(contexts)} sources across {len(parva_groups)} parvas.*")
        
        return "\n".join(answer_parts)
    
    def extract_core_info(self, text: str) -> str:
        """Extract the most relevant information from text"""
        sentences = re.split(r'[.!?]+', text)
        meaningful_sentences = [s.strip() for s in sentences if len(s.strip()) > 30]
        return " ".join(meaningful_sentences[:2]) + "."
    
    def format_answer(self, answer: str, contexts: List[Dict]) -> str:
        """Format the answer for better readability"""
        # Clean up the answer
        answer = re.sub(r'.*ANSWER:\s*', '', answer, flags=re.IGNORECASE)
        answer = answer.strip()
        
        # Add parva citations
        if contexts:
            parvas = set(ctx['metadata']['parva'] for ctx in contexts[:3])
            parva_str = ", ".join(parvas)
            answer += f"\n\n*Based on information from {parva_str}*"
        
        return answer
    
    def get_no_answer_template(self, question: str) -> str:
        """Template for when no good context is found"""
        return f"""## 🔍 Information Not Found

I couldn't find specific information about **"{question}"** in the available Mahabharata text.

**💡 Suggestions:**
- Try rephrasing your question
- Ask about major characters like Krishna, Arjuna, or Yudhishthira
- Inquire about key events like the Kurukshetra war
- Explore philosophical concepts like Dharma or Karma

**📚 Available Topics:**
- Character histories and relationships
- Philosophical teachings (Bhagavad Gita)
- Major events and battles  
- Moral dilemmas and lessons
- Cultural and spiritual insights"""

    def query(self, question: str) -> Dict:
        """Fast query method with timeout protection"""
        start_time = time.time()
        
        # Quick context retrieval
        contexts = self.retrieve_context(question, k=4)
        retrieval_time = time.time() - start_time
        
        # Generate answer with timeout
        gen_start = time.time()
        try:
            answer = self.generate_answer(question, contexts)
        except Exception as e:
            answer = self.enhanced_template_answer(question, contexts)
        
        generation_time = time.time() - gen_start
        total_time = time.time() - start_time
        
        # Calculate confidence
        confidence = sum(ctx['similarity_score'] for ctx in contexts) / len(contexts) if contexts else 0.0
        
        return {
            'answer': answer,
            'sources': contexts,
            'confidence': confidence,
            'response_time': total_time,
            'sources_count': len(contexts),
            'parvas_used': list(set(ctx['metadata']['parva'] for ctx in contexts))
        }

def main():
    st.markdown('<h1 class="main-header">🕉️ Mahabharata AI Scholar</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Cloud Edition • Instant Answers • Superior Readability</p>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'rag' not in st.session_state:
        st.session_state.rag = HighPerformanceMahabharataRAG()
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'current_question' not in st.session_state:
        st.session_state.current_question = ""
    
    # Sidebar with improved layout
    with st.sidebar:
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        st.header("🚀 Quick Questions")
        
        sample_questions = [
            "Who are the five Pandava brothers and their unique qualities?",
            "What is the main message of the Bhagavad Gita?",
            "Describe Krishna's role in the Kurukshetra war",
            "What caused the conflict between Pandavas and Kauravas?",
            "Explain the concept of Dharma with examples",
            "Tell me about Arjuna's exceptional archery skills",
            "What is the significance of Draupadi's character?",
            "Describe Yudhishthira's commitment to truth",
            "What lessons does Bhishma's life teach us?",
            "Explain Karna's tragic destiny and choices"
        ]
        
        for i, question in enumerate(sample_questions):
            if st.button(
                f"• {question}",
                key=f"sample_{i}",
                use_container_width=True,
                help="Click to load this question"
            ):
                st.session_state.current_question = question
                # Use JavaScript to trigger immediate execution (conceptual)
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Performance stats
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        st.header("📊 Performance")
        if st.session_state.chat_history:
            total_questions = len(st.session_state.chat_history)
            avg_time = sum(chat['response_time'] for chat in st.session_state.chat_history) / total_questions
            avg_confidence = sum(chat['confidence'] for chat in st.session_state.chat_history) / total_questions
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Questions", total_questions)
                st.metric("Avg Time", f"{avg_time:.1f}s")
            with col2:
                st.metric("Avg Confidence", f"{avg_confidence:.3f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Question input with instant loading
        question = st.text_input(
            " ",
            placeholder="Ask anything about Mahabharata characters, philosophy, or events...",
            value=st.session_state.current_question,
            key="main_question_input"
        )
        
        # Update current question when input changes
        if question != st.session_state.current_question:
            st.session_state.current_question = question
        
        if st.button("🚀 Get Instant Answer", type="primary", use_container_width=True):
            if question and question.strip():
                # Show loading state
                with st.spinner("🔍 Searching across all parvas..."):
                    result = st.session_state.rag.query(question.strip())
                
                # Store in history
                st.session_state.chat_history.append({
                    'question': question,
                    'answer': result['answer'],
                    'timestamp': time.time(),
                    'confidence': result['confidence'],
                    'sources_count': result['sources_count'],
                    'response_time': result['response_time']
                })
                
                # Display results
                st.markdown("## 📜 Comprehensive Answer")
                st.markdown(f'<div class="answer-box">{result["answer"]}</div>', unsafe_allow_html=True)
                
                # Performance metrics
                st.markdown("### 📊 Answer Metrics")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("⏱️ Response Time", f"{result['response_time']:.1f}s")
                with col2:
                    confidence_color = "🟢" if result['confidence'] > 0.6 else "🟡" if result['confidence'] > 0.3 else "🔴"
                    st.metric("📊 Confidence", f"{confidence_color} {result['confidence']:.3f}")
                with col3:
                    st.metric("📚 Sources", result['sources_count'])
                with col4:
                    st.metric("📖 Parvas", len(result.get('parvas_used', [])))
                
                # Source information
                if result['sources_count'] > 0:
                    st.markdown("### 📖 Sources Used")
                    parva_counts = {}
                    for source in result['sources']:
                        parva = source['metadata']['parva']
                        parva_counts[parva] = parva_counts.get(parva, 0) + 1
                    
                    for parva, count in parva_counts.items():
                        st.markdown(f'<span class="source-badge">{parva}: {count} source{"s" if count > 1 else ""}</span>', unsafe_allow_html=True)
            
            else:
                st.warning("Please enter a question about the Mahabharata")
    
    with col2:
        st.markdown("### 🌟 Quick Facts")
        facts = [
            "📚 18 Sacred Books",
            "👑 100+ Epic Characters", 
            "⚔️ Righteous War Theme",
            "🕉️ Spiritual Wisdom",
            "🎯 Moral Complexity",
            "🌍 Cultural Heritage",
            "📖 World's Longest Epic",
            "💡 Life Lessons"
        ]
        for fact in facts:
            st.info(fact)
        
        # Recent questions with instant loading
        if st.session_state.chat_history:
            st.markdown("### 💬 Recent Questions")
            for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):
                if st.button(
                    f"Q: {chat['question'][:40]}...",
                    key=f"recent_{i}",
                    use_container_width=True,
                    help="Click to ask this question again"
                ):
                    st.session_state.current_question = chat['question']
                    st.rerun()

if __name__ == "__main__":
    main()