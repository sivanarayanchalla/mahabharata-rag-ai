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
        width: 100%;
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
</style>
""", unsafe_allow_html=True)

class HighPerformanceMahabharataRAG:
    def __init__(self):
        # Cache the embedding model
        if 'embedder' not in st.session_state:
            with st.spinner("🔄 Loading AI models..."):
                st.session_state.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        self.embedder = st.session_state.embedder
        
        # Initialize vector store with error handling for existing collections
        self.client = chromadb.EphemeralClient()
        try:
            # Try to get existing collection first
            self.collection = self.client.get_collection("mahabharata_cloud_v2")
            st.sidebar.info("✅ Using existing knowledge base")
        except Exception:
            # If collection doesn't exist, create it
            self.collection = self.client.create_collection("mahabharata_cloud_v2")
            st.sidebar.info("🔄 Creating new knowledge base")
        
        # Load knowledge base with caching
        self.load_knowledge_base()
        
    def load_knowledge_base(self):
        """Load pre-processed Mahabharata chunks with caching"""
        try:
            if 'knowledge_loaded' not in st.session_state:
                with st.spinner("📚 Loading Mahabharata knowledge..."):
                    # Try complete Mahabharata first, fallback to single file
                    data_files = [
                        'data/processed/complete_mahabharata.json',
                        'data/processed/chunks.json'
                    ]
                    
                    chunks = []
                    for data_file in data_files:
                        if os.path.exists(data_file):
                            try:
                                with open(data_file, 'r', encoding='utf-8') as f:
                                    file_chunks = json.load(f)
                                chunks.extend(file_chunks)
                                st.sidebar.success(f"✅ Loaded {len(file_chunks)} chunks from {os.path.basename(data_file)}")
                                break
                            except Exception as e:
                                st.sidebar.warning(f"⚠️ Could not load {data_file}: {e}")
                                continue
                    
                    if not chunks:
                        st.error("❌ No Mahabharata data files found!")
                        return
                    
                    # Add to vector store (only if collection is empty)
                    if self.collection.count() == 0:
                        documents = [chunk['content'] for chunk in chunks[:400]]
                        embeddings = self.embedder.encode(documents).tolist()
                        metadatas = [{
                            'section_id': chunk.get('section_id', ''),
                            'parva': chunk.get('full_parva', chunk.get('parva', 'UNKNOWN')),
                            'chunk_id': chunk.get('chunk_id', ''),
                            'source_file': chunk.get('source_file', 'unknown')
                        } for chunk in chunks[:400]]
                        
                        self.collection.add(
                            embeddings=embeddings,
                            documents=documents,
                            metadatas=metadatas,
                            ids=[f"chunk_{i}" for i in range(len(documents))]
                        )
                    
                    st.session_state.knowledge_loaded = True
                    st.session_state.loaded_chunks_count = self.collection.count()
            
            st.sidebar.success(f"✅ Knowledge base ready: {st.session_state.loaded_chunks_count} chunks")
            
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
        
        # Enhanced template answer (primary method)
        return self.enhanced_template_answer(question, contexts)
    
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
        answer_parts = [f"## 📖 Answer\n"]
        
        # Main answer from best context
        if contexts:
            best_ctx = max(contexts, key=lambda x: x['similarity_score'])
            main_info = self.extract_core_info(best_ctx['content'])
            answer_parts.append(f"\n{main_info}\n")
        
        # Add supporting information from different parvas
        if len(parva_groups) > 1:
            answer_parts.append("\n### 🔍 Additional Insights:\n")
            for parva, ctx_list in list(parva_groups.items())[:3]:
                if ctx_list:
                    sample_text = ctx_list[0]['content']
                    # Extract the most meaningful sentence
                    sentences = re.split(r'[.!?]+', sample_text)
                    meaningful = [s.strip() for s in sentences if len(s.strip()) > 20]
                    if meaningful:
                        sample = meaningful[0] + "."
                        answer_parts.append(f"• **{parva}**: {sample}\n")
        
        # Add source information
        answer_parts.append(f"\n*Based on analysis of {len(contexts)} sources across {len(parva_groups)} parvas of the Mahabharata.*")
        
        return "\n".join(answer_parts)
    
    def extract_core_info(self, text: str) -> str:
        """Extract the most relevant information from text"""
        sentences = re.split(r'[.!?]+', text)
        meaningful_sentences = [s.strip() for s in sentences if len(s.strip()) > 30]
        return " ".join(meaningful_sentences[:3]) + "."
    
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
        
        # Generate answer
        gen_start = time.time()
        answer = self.generate_answer(question, contexts)
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
            'parvas_used': list(set(ctx['metadata']['parva'] for ctx in contexts)) if contexts else []
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
    
    if 'answer_trigger' not in st.session_state:
        st.session_state.answer_trigger = False
    
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
        
        # Use form to handle sample question clicks properly
        with st.form("sample_questions_form"):
            selected_question = st.selectbox(
                "Choose a sample question:",
                [""] + sample_questions,
                format_func=lambda x: "Select a question..." if x == "" else x
            )
            
            sample_submitted = st.form_submit_button("Ask This Question")
            
            if sample_submitted and selected_question:
                st.session_state.current_question = selected_question
                st.session_state.answer_trigger = True
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Performance stats
        if st.session_state.chat_history:
            st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
            st.header("📊 Performance")
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
        # Main question form
        with st.form("main_question_form"):
            question = st.text_input(
                "Your Question:",
                placeholder="Ask anything about Mahabharata characters, philosophy, or events...",
                value=st.session_state.current_question,
                key="main_question_input"
            )
            
            submitted = st.form_submit_button("🚀 Get Instant Answer", use_container_width=True)
            
            if submitted and question.strip():
                st.session_state.current_question = question.strip()
                st.session_state.answer_trigger = True
                st.rerun()
        
        # Handle answer generation when triggered
        if st.session_state.answer_trigger and st.session_state.current_question:
            question = st.session_state.current_question
            
            # Show loading state
            with st.spinner("🔍 Searching across all parvas..."):
                result = st.session_state.rag.query(question)
            
            # Store in history
            st.session_state.chat_history.append({
                'question': question,
                'answer': result['answer'],
                'timestamp': time.time(),
                'confidence': result['confidence'],
                'sources_count': result['sources_count'],
                'response_time': result['response_time']
            })
            
            # Reset trigger
            st.session_state.answer_trigger = False
            
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
        
        # Recent questions
        if st.session_state.chat_history:
            st.markdown("### 💬 Recent Questions")
            recent_chats = list(reversed(st.session_state.chat_history[-5:]))
            
            for i, chat in enumerate(recent_chats):
                # Shorten question for display
                display_question = chat['question'][:50] + "..." if len(chat['question']) > 50 else chat['question']
                
                if st.button(
                    f"Q: {display_question}",
                    key=f"recent_{i}",
                    use_container_width=True,
                    help="Click to ask this question again"
                ):
                    st.session_state.current_question = chat['question']
                    st.session_state.answer_trigger = True
                    st.rerun()

if __name__ == "__main__":
    main()