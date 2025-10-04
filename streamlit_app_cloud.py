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
    .character-highlight {
        background: linear-gradient(120deg, #FFEAA7 0%, #FFEAA7 100%);
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #FDCB6E;
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
            
            # Load ALL available knowledge base files
            self.load_complete_knowledge_base()
            
        except Exception as e:
            st.error(f"❌ Failed to initialize RAG system: {e}")
            raise
    
    def load_complete_knowledge_base(self):
        """Load ALL available Mahabharata chunk files"""
        try:
            # Check if knowledge is already loaded
            if self.collection.count() > 100:  # Reasonable threshold
                st.session_state.knowledge_loaded = True
                st.session_state.loaded_chunks_count = self.collection.count()
                st.sidebar.success(f"✅ Knowledge base ready: {self.collection.count()} chunks")
                return
            
            with st.spinner("📚 Loading COMPLETE Mahabharata knowledge base..."):
                all_chunks = []
                loaded_files = []
                
                # Load ALL individual chunk files (maha01_chunks.json to maha18_chunks.json)
                for i in range(1, 19):
                    file_name = f"maha{i:02d}_chunks.json"
                    file_path = f"data/processed/{file_name}"
                    
                    if os.path.exists(file_path):
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                chunks = json.load(f)
                            all_chunks.extend(chunks)
                            loaded_files.append(file_name)
                            st.sidebar.info(f"📖 Loaded {len(chunks)} from {file_name}")
                        except Exception as e:
                            st.sidebar.warning(f"⚠️ Could not load {file_name}: {e}")
                    else:
                        st.sidebar.warning(f"📭 File not found: {file_name}")
                
                # Also try to load complete combined file if it exists
                complete_data_path = "data/processed/complete_mahabharata.json"
                if os.path.exists(complete_data_path) and len(all_chunks) < 100:
                    try:
                        with open(complete_data_path, 'r', encoding='utf-8') as f:
                            complete_chunks = json.load(f)
                        all_chunks.extend(complete_chunks)
                        loaded_files.append("complete_mahabharata.json")
                        st.sidebar.success(f"✅ Added {len(complete_chunks)} from complete file")
                    except Exception as e:
                        st.sidebar.warning(f"⚠️ Could not load complete data: {e}")
                
                # If still no chunks, process raw files
                if not all_chunks:
                    all_chunks = self.process_raw_files()
                    if all_chunks:
                        loaded_files.append("raw_files")
                
                if not all_chunks:
                    st.error("❌ No Mahabharata data found! Please check your data files.")
                    st.info("💡 Make sure you have processed chunk files in data/processed/ folder")
                    return
                
                # Add ALL chunks to vector store (no limiting for now)
                self.add_all_chunks_to_store(all_chunks)
                
                st.session_state.knowledge_loaded = True
                st.session_state.loaded_chunks_count = len(all_chunks)
                st.sidebar.success(f"✅ COMPLETE knowledge base loaded: {len(all_chunks)} chunks from {len(loaded_files)} files")
                
        except Exception as e:
            st.error(f"❌ Error loading knowledge base: {e}")
    
    def process_raw_files(self):
        """Process raw Mahabharata files as last resort"""
        try:
            st.sidebar.warning("🔄 Processing raw files as fallback...")
            
            raw_files = glob.glob("data/raw/maha*.txt")
            if not raw_files:
                st.sidebar.error("❌ No raw files found in data/raw/")
                return []
            
            all_chunks = []
            
            for file_path in raw_files[:5]:  # Process first 5 files for speed
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Simple chunking
                    paragraphs = re.split(r'\n\s*\n', content)
                    chunks = []
                    
                    for i, para in enumerate(paragraphs[:100]):  # Increased limit
                        if len(para.strip()) > 50:  # Lower threshold
                            chunks.append({
                                'content': para.strip(),
                                'chunk_id': f"{os.path.basename(file_path)}_{i}",
                                'section_id': 'AUTO',
                                'parva': f"AUTO_{os.path.basename(file_path)}",
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
    
    def add_all_chunks_to_store(self, chunks: List[Dict]):
        """Add ALL chunks to vector store without limiting"""
        try:
            # Use ALL chunks for maximum coverage
            documents = [chunk['content'] for chunk in chunks]
            embeddings = self.embedder.encode(documents).tolist()
            metadatas = [{
                'section_id': chunk.get('section_id', ''),
                'parva': chunk.get('full_parva', chunk.get('parva', 'UNKNOWN')),
                'chunk_id': chunk.get('chunk_id', ''),
                'source_file': chunk.get('source_file', 'unknown'),
                'word_count': chunk.get('word_count', 0)
            } for chunk in chunks]
            
            # Create unique IDs
            ids = [f"chunk_{i}" for i in range(len(chunks))]
            
            # Clear existing collection and add all chunks
            try:
                self.client.delete_collection("mahabharata_complete")
            except:
                pass
            
            self.collection = self.client.create_collection("mahabharata_complete")
            
            # Add in batches if too many chunks
            batch_size = 500
            total_batches = (len(documents) - 1) // batch_size + 1
            
            for i in range(0, len(documents), batch_size):
                batch_end = min(i + batch_size, len(documents))
                self.collection.add(
                    embeddings=embeddings[i:batch_end],
                    documents=documents[i:batch_end],
                    metadatas=metadatas[i:batch_end],
                    ids=ids[i:batch_end]
                )
                progress = (i + batch_size) / len(documents)
                st.sidebar.info(f"📦 Added batch {(i//batch_size) + 1}/{total_batches} ({progress:.1%})")
            
            st.sidebar.success(f"📚 Added {len(documents)} TOTAL chunks to knowledge base")
            
        except Exception as e:
            st.error(f"❌ Error adding chunks to store: {e}")
    
    def retrieve_context(self, query: str, k: int = 8) -> List[Dict]:
        """Enhanced context retrieval with better matching"""
        try:
            query_embedding = self.embedder.encode(query).tolist()
            
            # Get more results initially, then filter
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(k * 2, 20),  # Get more, filter later
                include=['documents', 'metadatas', 'distances']
            )
            
            contexts = []
            for doc, metadata, distance in zip(
                results['documents'][0],
                results['metadatas'][0], 
                results['distances'][0]
            ):
                similarity_score = max(0.0, 1 - distance)
                # Much lower threshold to catch more relevant results
                if similarity_score > 0.05:  # Reduced threshold significantly
                    contexts.append({
                        'content': doc,
                        'metadata': metadata,
                        'similarity_score': similarity_score
                    })
            
            # Sort by similarity and take top k
            contexts.sort(key=lambda x: x['similarity_score'], reverse=True)
            return contexts[:k]
            
        except Exception as e:
            st.error(f"🔍 Retrieval error: {e}")
            return []
    
    def generate_enhanced_answer(self, question: str, contexts: List[Dict]) -> str:
        """Generate comprehensive answers with character-specific enhancements"""
        if not contexts:
            return self.get_character_specific_fallback(question)
        
        # Build detailed answer using all contexts
        answer_parts = []
        
        # Character-specific enhancements
        if self.is_character_question(question):
            answer_parts.append(self.generate_character_profile(question, contexts))
        else:
            answer_parts.append(self.generate_general_answer(question, contexts))
        
        return "\n".join(answer_parts)
    
    def is_character_question(self, question: str) -> bool:
        """Check if question is about a character"""
        character_keywords = [
            'draupadi', 'krishna', 'arjuna', 'yudhishthira', 'bhima', 'nakula', 'sahadeva',
            'duryodhana', 'karna', 'bhishma', 'drona', 'vidura', 'kunti', 'gandhari',
            'shakuni', 'dushasana', 'abhima', 'subhadra', 'ulupi', 'chitrangada'
        ]
        question_lower = question.lower()
        return any(char in question_lower for char in character_keywords)
    
    def generate_character_profile(self, question: str, contexts: List[Dict]) -> str:
        """Generate character-specific profile"""
        character_name = self.extract_character_name(question)
        
        answer_parts = []
        answer_parts.append(f"## 👑 Character Profile: {character_name.title()}")
        
        # Extract character information
        character_info = self.extract_character_information(character_name, contexts)
        
        if character_info:
            answer_parts.append(f"\n{character_info}")
        else:
            answer_parts.append(f"\n**Based on the available text, here's what I found about {character_name.title()}:**")
            
            # Try to extract any relevant information
            relevant_sentences = self.extract_relevant_sentences(character_name, contexts)
            if relevant_sentences:
                answer_parts.append("\n### 📖 Key Information:")
                for sentence in relevant_sentences[:5]:  # Limit to 5 sentences
                    answer_parts.append(f"• {sentence}")
            else:
                answer_parts.append(f"\n*Specific detailed information about {character_name.title()} was not found in the currently loaded text sections.*")
        
        # Add source information
        source_info = self.get_source_information(contexts)
        answer_parts.append(f"\n{source_info}")
        
        return "\n".join(answer_parts)
    
    def extract_character_name(self, question: str) -> str:
        """Extract character name from question"""
        question_lower = question.lower()
        
        character_mapping = {
            'draupadi': 'Draupadi',
            'krishna': 'Krishna',
            'arjuna': 'Arjuna', 
            'yudhishthira': 'Yudhishthira',
            'bhima': 'Bhima',
            'nakula': 'Nakula',
            'sahadeva': 'Sahadeva',
            'duryodhana': 'Duryodhana',
            'karna': 'Karna',
            'bhishma': 'Bhishma',
            'drona': 'Drona',
            'vidura': 'Vidura',
            'kunti': 'Kunti',
            'gandhari': 'Gandhari',
            'shakuni': 'Shakuni',
            'dushasana': 'Dushasana'
        }
        
        for key, name in character_mapping.items():
            if key in question_lower:
                return name
        
        # Default: extract first word after "about"
        if 'about' in question_lower:
            parts = question_lower.split('about')
            if len(parts) > 1:
                return parts[1].strip().title()
        
        return "this character"
    
    def extract_character_information(self, character_name: str, contexts: List[Dict]) -> str:
        """Extract comprehensive character information"""
        character_sentences = []
        
        for ctx in contexts:
            content = ctx['content']
            # Look for sentences mentioning the character
            sentences = re.split(r'[.!?]+', content)
            
            for sentence in sentences:
                sentence_lower = sentence.lower()
                if character_name.lower() in sentence_lower and len(sentence.strip()) > 15:
                    # Clean up the sentence
                    clean_sentence = sentence.strip()
                    if clean_sentence not in character_sentences:
                        character_sentences.append(clean_sentence)
        
        if character_sentences:
            # Group related information
            return " ".join(character_sentences[:10])  # Increased limit to 10 sentences
        else:
            return ""
    
    def extract_relevant_sentences(self, character_name: str, contexts: List[Dict]) -> List[str]:
        """Extract any sentences that might be relevant to the character"""
        relevant_sentences = []
        character_lower = character_name.lower()
        
        for ctx in contexts:
            content = ctx['content']
            sentences = re.split(r'[.!?]+', content)
            
            for sentence in sentences:
                sentence_clean = sentence.strip()
                if (len(sentence_clean) > 10 and 
                    any(keyword in sentence_clean.lower() for keyword in 
                        ['wife', 'husband', 'son', 'daughter', 'brother', 'sister', 
                         'king', 'queen', 'warrior', 'prince', 'princess', 'married',
                         'father', 'mother', 'family', 'role', 'character'])):
                    relevant_sentences.append(sentence_clean)
        
        return relevant_sentences[:8]  # Limit to 8 sentences
    
    def generate_general_answer(self, question: str, contexts: List[Dict]) -> str:
        """Generate answer for non-character questions"""
        answer_parts = []
        answer_parts.append("## 📖 Comprehensive Answer")
        
        # Use the most relevant context as main answer
        if contexts:
            main_context = contexts[0]
            main_content = main_context['content']
            
            # Extract key sentences
            sentences = re.split(r'[.!?]+', main_content)
            key_sentences = [s.strip() for s in sentences if len(s.strip()) > 15]
            
            if key_sentences:
                answer_parts.append("\n" + " ".join(key_sentences[:4]) + ".")  # Increased to 4 sentences
            
            # Add supporting information from other contexts
            if len(contexts) > 1:
                answer_parts.append("\n### 🔍 Additional Context:")
                supporting_info = []
                for ctx in contexts[1:5]:  # Use next 4 contexts
                    sentences = re.split(r'[.!?]+', ctx['content'])
                    meaningful = [s.strip() for s in sentences if len(s.strip()) > 10]  # Lower threshold
                    if meaningful:
                        supporting_info.append(meaningful[0] + ".")
                
                if supporting_info:
                    answer_parts.extend(supporting_info)
        
        # Add source information
        source_info = self.get_source_information(contexts)
        answer_parts.append(f"\n{source_info}")
        
        return "\n".join(answer_parts)
    
    def get_character_specific_fallback(self, question: str) -> str:
        """Character-specific fallback when no information is found"""
        character_name = self.extract_character_name(question)
        
        fallback = f"""## 👑 Character: {character_name.title()}

**🔍 Information Limited in Current Search**

I searched through the available Mahabharata text but found limited specific information about **{character_name.title()}** in the currently loaded sections.

**💡 Character-Specific Suggestions:**

• **Try alternative names or titles**: {self.get_character_alternatives(character_name)}
• **Ask about specific relationships**: "What is {character_name.title()}'s relationship with [other character]?"
• **Inquire about key events**: "What role did {character_name.title()} play in [specific event]?"
• **Request general background**: "Tell me about {character_name.title()}'s family and background"

**📚 Common Mahabharata Characters with Good Coverage:**
- Krishna, Arjuna, Yudhishthira, Bhima
- Draupadi, Kunti, Gandhari  
- Duryodhana, Karna, Bhishma
- Drona, Vidura, Shakuni

*Note: Character information is distributed across different parvas. The system may not have loaded all relevant sections yet.*"""

        return fallback
    
    def get_character_alternatives(self, character_name: str) -> str:
        """Get alternative names for characters"""
        alternatives = {
            'draupadi': 'Panchali, Yajnaseni, Krishnaa',
            'krishna': 'Govinda, Madhava, Vasudeva, Gopala',
            'arjuna': 'Partha, Dhananjaya, Gudakesha, Kirīṭī',
            'yudhishthira': 'Dharmaraja, Ajatashatru',
            'bhima': 'Vrikodara, Bhimasena',
            'karna': 'Radheya, Vasusena, Angaraja',
            'bhishma': 'Ganga putra, Devavrata',
            'drona': 'Dronacharya, Guru Drona'
        }
        return alternatives.get(character_name.lower(), 'various epithets and titles')
    
    def get_source_information(self, contexts: List[Dict]) -> str:
        """Get information about sources used"""
        if not contexts:
            return "*No specific sources were found for this query.*"
        
        parva_counts = {}
        for ctx in contexts:
            parva = ctx['metadata']['parva']
            parva_counts[parva] = parva_counts.get(parva, 0) + 1
        
        parva_list = [f"{parva} ({count})" for parva, count in parva_counts.items()]
        return f"*Information synthesized from {len(contexts)} sources across {len(parva_counts)} parvas: {', '.join(parva_list)}*"

    def query(self, question: str) -> Dict:
        """Comprehensive query method with enhanced retrieval"""
        start_time = time.time()
        
        # Enhanced context retrieval with query expansion
        contexts = self.retrieve_context(question, k=8)
        
        # If no contexts found, try with expanded query
        if not contexts:
            expanded_queries = self.expand_query(question)
            for expanded_query in expanded_queries:
                contexts = self.retrieve_context(expanded_query, k=6)
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
        
        # Add character-specific expansions
        character_expansions = {
            'draupadi': ['Draupadi Panchali', 'Yajnaseni', 'Krishnaa', 'Pandava wife', 'Draupadi character', 'Draupadi story'],
            'arjuna': ['Arjuna Partha', 'Dhananjaya', 'Pandava archer', 'Arjuna skills', 'Arjuna and Krishna'],
            'krishna': ['Krishna Govinda', 'Vasudeva', 'Lord Krishna', 'Krishna role', 'Krishna Bhagavad Gita'],
            'yudhishthira': ['Yudhishthira Dharmaraja', 'eldest Pandava', 'Yudhishthira justice', 'Yudhishthira Dharma'],
            'bhima': ['Bhima Vrikodara', 'strong Pandava', 'Bhima strength', 'Bhimasena'],
            'karna': ['Karna Radheya', 'suryaputra', 'Karna generosity', 'Karna story'],
            'bhishma': ['Bhishma pitamaha', 'Ganga putra', 'Bhishma vow', 'Devavrata'],
            'drona': ['Dronacharya', 'Guru Drona', 'Drona teacher', 'Dronacharya story']
        }
        
        for char, variants in character_expansions.items():
            if char in question_lower:
                expansions.extend(variants)
                break
        
        return expansions

def main():
    st.markdown('<h1 class="main-header">🕉️ Mahabharata AI Scholar</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Complete Edition • All 18 Parvas • Enhanced Character Answers</p>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'rag' not in st.session_state:
        try:
            st.session_state.rag = RobustMahabharataRAG()
        except Exception as e:
            st.error(f"❌ Failed to initialize: {e}")
            st.info("💡 Please check that your data files are available in data/processed/ folder")
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
            "Who is Draupadi and what is her story?",
            "Tell me about Draupadi's marriage", 
            "What is Draupadi's role in the Mahabharata?",
            "Describe Draupadi's relationship with the Pandavas",
            "Who is Krishna and what is his role?",
            "Tell me about the Pandava brothers",
            "What is the Bhagavad Gita about?",
            "Describe the Kurukshetra war",
            "Who is Arjuna and what are his skills?",
            "Explain the concept of Dharma",
            "What is Yudhishthira known for?",
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
        
        # System Info
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        st.header("⚙️ System Info")
        if hasattr(st.session_state, 'loaded_chunks_count'):
            st.metric("Knowledge Chunks", st.session_state.loaded_chunks_count)
        st.metric("Vector DB", "ChromaDB")
        st.metric("Embeddings", "Sentence Transformers")
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
                display_question = chat['question'][:40] + "..." if len(chat['question']) > 40 else chat['question']
                
                if st.button(
                    f"Q: {display_question}",
                    key=f"recent_{i}",
                    use_container_width=True,
                    help="Click to ask this question again"
                ):
                    st.session_state.current_question = chat['question']
                    st.session_state.answer_trigger = True
                    st.rerun()
        
        # Character Highlights
        st.markdown("### 👑 Popular Characters")
        characters = [
            "Draupadi - Panchali",
            "Krishna - Govinda", 
            "Arjuna - Partha",
            "Yudhishthira - Dharmaraja",
            "Bhima - Vrikodara",
            "Karna - Radheya"
        ]
        for char in characters:
            st.caption(f"• {char}")

if __name__ == "__main__":
    main()