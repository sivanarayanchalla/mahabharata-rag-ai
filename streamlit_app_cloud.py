import streamlit as st
import json
import os
import requests
from sentence_transformers import SentenceTransformer
import chromadb
from typing import List, Dict
import time
import re
import glob

# Custom CSS for superior readability and performance
st.set_page_config(
    page_title="Mahabharata AI Scholar - Complete Edition",
    page_icon="🕉️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
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
    .sidebar-content {
        background: linear-gradient(180deg, #F8F9FA 0%, #FFFFFF 100%);
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

class RobustMahabharataRAG:
    def __init__(self):
        # Initialize with error handling
        try:
            # Cache the embedding model
            if 'embedder' not in st.session_state:
                with st.spinner("🔄 Loading AI models..."):
                    st.session_state.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            
            self.embedder = st.session_state.embedder
            
            # Initialize ChromaDB with persistence
            self.client = chromadb.PersistentClient(path="./chroma_db")
            
            # Try to get existing collection or create new one
            try:
                self.collection = self.client.get_collection("mahabharata_complete")
                st.sidebar.info("✅ Using existing knowledge base")
            except Exception:
                self.collection = self.client.create_collection("mahabharata_complete")
                st.sidebar.info("🔄 Creating new knowledge base")
            
            # Load knowledge base
            self.load_knowledge_base()
            
        except Exception as e:
            st.error(f"❌ Failed to initialize RAG system: {e}")
            raise
    
    def load_knowledge_base(self):
        """Load all available Mahabharata data with comprehensive fallbacks"""
        try:
            # Check if knowledge is already loaded
            if self.collection.count() > 0:
                st.session_state.knowledge_loaded = True
                st.session_state.loaded_chunks_count = self.collection.count()
                st.sidebar.success(f"✅ Knowledge base ready: {self.collection.count()} chunks")
                return
            
            with st.spinner("📚 Loading Mahabharata knowledge base..."):
                all_chunks = []
                
                # Try to load complete processed data first
                complete_data_path = "data/processed/complete_mahabharata.json"
                if os.path.exists(complete_data_path):
                    try:
                        with open(complete_data_path, 'r', encoding='utf-8') as f:
                            complete_chunks = json.load(f)
                        all_chunks.extend(complete_chunks)
                        st.sidebar.success(f"✅ Loaded {len(complete_chunks)} chunks from complete Mahabharata")
                    except Exception as e:
                        st.sidebar.warning(f"⚠️ Could not load complete data: {e}")
                
                # If no complete data, try individual files
                if not all_chunks:
                    individual_files = glob.glob("data/processed/*_chunks.json")
                    for file_path in individual_files:
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                chunks = json.load(f)
                            all_chunks.extend(chunks)
                            st.sidebar.info(f"📖 Loaded {len(chunks)} from {os.path.basename(file_path)}")
                        except Exception as e:
                            st.sidebar.warning(f"⚠️ Could not load {file_path}: {e}")
                
                # Last resort: process raw files on the fly
                if not all_chunks:
                    all_chunks = self.process_raw_files()
                
                if not all_chunks:
                    st.error("❌ No Mahabharata data found! Please check your data files.")
                    return
                
                # Add chunks to vector store
                self.add_chunks_to_store(all_chunks)
                
                st.session_state.knowledge_loaded = True
                st.session_state.loaded_chunks_count = len(all_chunks)
                st.sidebar.success(f"✅ Knowledge base loaded: {len(all_chunks)} chunks")
                
        except Exception as e:
            st.error(f"❌ Error loading knowledge base: {e}")
    
    def process_raw_files(self):
        """Process raw Mahabharata files as last resort"""
        try:
            st.sidebar.warning("🔄 Processing raw files...")
            
            raw_files = glob.glob("data/raw/maha*.txt")
            if not raw_files:
                return []
            
            all_chunks = []
            
            for file_path in raw_files[:3]:  # Process first 3 files for speed
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Simple chunking
                    paragraphs = re.split(r'\n\s*\n', content)
                    chunks = []
                    
                    for i, para in enumerate(paragraphs[:50]):  # Limit chunks
                        if len(para.strip()) > 100:
                            chunks.append({
                                'content': para.strip(),
                                'chunk_id': f"{os.path.basename(file_path)}_{i}",
                                'section_id': 'AUTO',
                                'parva': 'AUTO_PROCESSED',
                                'word_count': len(para.split())
                            })
                    
                    all_chunks.extend(chunks)
                    st.sidebar.info(f"📄 Processed {len(chunks)} from {os.path.basename(file_path)}")
                    
                except Exception as e:
                    st.sidebar.warning(f"⚠️ Error processing {file_path}: {e}")
            
            return all_chunks
            
        except Exception as e:
            st.error(f"❌ Error processing raw files: {e}")
            return []
    
    def add_chunks_to_store(self, chunks: List[Dict]):
        """Add chunks to vector store with proper batching"""
        try:
            # Use more chunks for better coverage
            chunks_to_add = chunks[:600]  # Increased from 400 to 600
            
            documents = [chunk['content'] for chunk in chunks_to_add]
            embeddings = self.embedder.encode(documents).tolist()
            metadatas = [{
                'section_id': chunk.get('section_id', ''),
                'parva': chunk.get('full_parva', chunk.get('parva', 'UNKNOWN')),
                'chunk_id': chunk.get('chunk_id', ''),
                'source_file': chunk.get('source_file', 'unknown'),
                'word_count': chunk.get('word_count', 0)
            } for chunk in chunks_to_add]
            
            ids = [chunk.get('global_chunk_id', f"chunk_{i}") for i, chunk in enumerate(chunks_to_add)]
            
            self.collection.add(
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            st.sidebar.success(f"📚 Added {len(documents)} chunks to knowledge base")
            
        except Exception as e:
            st.error(f"❌ Error adding chunks to store: {e}")
    
    def retrieve_context(self, query: str, k: int = 6) -> List[Dict]:  # Increased k from 5 to 6
        """Improved context retrieval with better matching"""
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
                # Lower threshold to catch more relevant results
                if similarity_score > 0.1:  # Reduced from 0.15 to 0.1
                    contexts.append({
                        'content': doc,
                        'metadata': metadata,
                        'similarity_score': similarity_score
                    })
            
            return contexts
            
        except Exception as e:
            st.error(f"🔍 Retrieval error: {e}")
            return []
    
    def generate_enhanced_answer(self, question: str, contexts: List[Dict]) -> str:
        """Generate comprehensive answers with better context usage"""
        if not contexts:
            return self.get_enhanced_no_answer_template(question)
        
        # Build detailed answer using all contexts
        answer_parts = []
        
        # Main answer section
        answer_parts.append("## 📖 Comprehensive Answer\n")
        
        # Extract and combine information from all contexts
        combined_info = self.combine_context_information(contexts)
        answer_parts.append(combined_info)
        
        # Add specific examples if available
        specific_examples = self.extract_specific_examples(contexts, question)
        if specific_examples:
            answer_parts.append("\n### 🔍 Specific Details:\n")
            answer_parts.append(specific_examples)
        
        # Add source information
        source_info = self.get_source_information(contexts)
        answer_parts.append(f"\n{source_info}")
        
        return "\n".join(answer_parts)
    
    def combine_context_information(self, contexts: List[Dict]) -> str:
        """Combine information from multiple contexts"""
        # Sort by similarity score
        sorted_contexts = sorted(contexts, key=lambda x: x['similarity_score'], reverse=True)
        
        combined_texts = []
        used_content = set()
        
        for ctx in sorted_contexts:
            content = ctx['content']
            # Avoid duplicate content
            content_hash = hash(content[:100])
            if content_hash not in used_content:
                combined_texts.append(content)
                used_content.add(content_hash)
        
        # Take top 3 most relevant contexts
        top_contexts = combined_texts[:3]
        
        # Simple combination - in production, you'd use an LLM here
        if top_contexts:
            main_info = self.extract_core_info(top_contexts[0])
            supporting_info = " ".join([self.extract_key_points(ctx) for ctx in top_contexts[1:]])
            
            return f"{main_info}\n\n{supporting_info}"
        else:
            return "Based on the available text, here's what I found..."
    
    def extract_specific_examples(self, contexts: List[Dict], question: str) -> str:
        """Extract specific examples relevant to the question"""
        examples = []
        question_lower = question.lower()
        
        for ctx in contexts[:2]:  # Use top 2 contexts for examples
            content = ctx['content']
            
            # Look for specific patterns based on question type
            if any(keyword in question_lower for keyword in ['who is', 'who are', 'describe']):
                # Extract character descriptions
                sentences = re.split(r'[.!?]+', content)
                descriptive_sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
                if descriptive_sentences:
                    examples.append(f"• {descriptive_sentences[0]}.")
            
            elif any(keyword in question_lower for keyword in ['what is', 'explain', 'meaning']):
                # Extract explanatory sentences
                sentences = re.split(r'[.!?]+', content)
                explanatory = [s.strip() for s in sentences if any(word in s.lower() for word in ['is', 'means', 'defined', 'called'])]
                if explanatory:
                    examples.append(f"• {explanatory[0]}.")
        
        return "\n".join(examples) if examples else ""
    
    def extract_core_info(self, text: str) -> str:
        """Extract the most relevant information from text"""
        sentences = re.split(r'[.!?]+', text)
        meaningful_sentences = [s.strip() for s in sentences if len(s.strip()) > 25]
        return " ".join(meaningful_sentences[:2]) + "."
    
    def extract_key_points(self, text: str) -> str:
        """Extract key points from text"""
        sentences = re.split(r'[.!?]+', text)
        key_sentences = [s.strip() for s in sentences if len(s.strip()) > 15]
        return " ".join(key_sentences[:1]) + "."
    
    def get_source_information(self, contexts: List[Dict]) -> str:
        """Get information about sources used"""
        parva_counts = {}
        for ctx in contexts:
            parva = ctx['metadata']['parva']
            parva_counts[parva] = parva_counts.get(parva, 0) + 1
        
        parva_list = [f"{parva} ({count})" for parva, count in parva_counts.items()]
        return f"*Information synthesized from {len(contexts)} sources across {len(parva_counts)} parvas: {', '.join(parva_list)}*"
    
    def get_enhanced_no_answer_template(self, question: str) -> str:
        """Enhanced template for when no good context is found"""
        return f"""## 🔍 Information Not Found

I searched through the available Mahabharata text but couldn't find specific information about **"{question}"**.

**🤔 This could be because:**
- The information might be in parts of the text not currently loaded
- Your question might need rephrasing for better matching
- The concept might be known by a different name in the text

**💡 Try these approaches:**
- Rephrase your question (e.g., "Tell me about Krishna's role" instead of "Who is Krishna")
- Ask about specific events or relationships
- Use the character's full name or common epithets
- Break complex questions into simpler ones

**🎯 Sample questions that work well:**
- "Describe Krishna's role in the Kurukshetra war"
- "What is the relationship between Krishna and Arjuna?"
- "Tell me about the Bhagavad Gita teachings"
- "Who are the main Pandava brothers?"""

    def query(self, question: str) -> Dict:
        """Comprehensive query method with enhanced retrieval"""
        start_time = time.time()
        
        # Enhanced context retrieval with query expansion
        contexts = self.retrieve_context(question, k=6)
        
        # If no contexts found, try with expanded query
        if not contexts:
            expanded_queries = self.expand_query(question)
            for expanded_query in expanded_queries:
                contexts = self.retrieve_context(expanded_query, k=4)
                if contexts:
                    break
        
        retrieval_time = time.time() - start_time
        
        # Generate comprehensive answer
        gen_start = time.time()
        answer = self.generate_enhanced_answer(question, contexts)
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
    
    def expand_query(self, question: str) -> List[str]:
        """Expand query with variations for better retrieval"""
        question_lower = question.lower()
        expansions = [question]
        
        # Add common variations
        if 'krishna' in question_lower:
            expansions.extend([
                f"{question} in Mahabharata",
                "Krishna role Kurukshetra war",
                "Krishna and Arjuna relationship",
                "Lord Krishna Mahabharata",
                "Krishna Bhagavad Gita"
            ])
        elif 'arjuna' in question_lower:
            expansions.extend([
                f"{question} Pandava",
                "Arjuna archery skills",
                "Arjuna and Krishna",
                "Pandava brothers"
            ])
        elif 'pandava' in question_lower:
            expansions.extend([
                "five Pandava brothers",
                "Yudhishthira Bhima Arjuna Nakula Sahadeva",
                "Pandava family"
            ])
        
        return expansions

def main():
    st.markdown('<h1 class="main-header">🕉️ Mahabharata AI Scholar</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Complete Edition • Enhanced Retrieval • Better Answers</p>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'rag' not in st.session_state:
        try:
            st.session_state.rag = RobustMahabharataRAG()
        except Exception as e:
            st.error(f"❌ Failed to initialize: {e}")
            st.info("💡 Please check that Ollama is running and data files are available")
            return
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'current_question' not in st.session_state:
        st.session_state.current_question = ""
    
    if 'answer_trigger' not in st.session_state:
        st.session_state.answer_trigger = False
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        st.header("🚀 Quick Questions")
        
        sample_questions = [
            "Who is Krishna and what is his role?",
            "Tell me about the Pandava brothers",
            "What is the Bhagavad Gita about?",
            "Describe the Kurukshetra war",
            "Who is Arjuna and what are his skills?",
            "Explain the concept of Dharma",
            "What is Yudhishthira known for?",
            "Tell me about Draupadi's character",
            "Who are the main Kauravas?",
            "What is Bhishma's role in the story?"
        ]
        
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
            
            submitted = st.form_submit_button("🚀 Get Comprehensive Answer", use_container_width=True)
            
            if submitted and question.strip():
                st.session_state.current_question = question.strip()
                st.session_state.answer_trigger = True
                st.rerun()
        
        # Handle answer generation when triggered
        if st.session_state.answer_trigger and st.session_state.current_question:
            question = st.session_state.current_question
            
            # Show loading state
            with st.spinner("🔍 Enhanced search across all parvas..."):
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
            "⚔️ Kurukshetra War",
            "🕉️ Bhagavad Gita",
            "📖 Longest Epic Poem",
            "🌍 Ancient Indian Wisdom",
            "💡 Philosophical Depth",
            "🎭 Complex Relationships"
        ]
        for fact in facts:
            st.info(fact)
        
        # Recent questions
        if st.session_state.chat_history:
            st.markdown("### 💬 Recent Questions")
            recent_chats = list(reversed(st.session_state.chat_history[-5:]))
            
            for i, chat in enumerate(recent_chats):
                display_question = chat['question'][:45] + "..." if len(chat['question']) > 45 else chat['question']
                
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