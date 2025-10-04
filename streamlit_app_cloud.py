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
    .alias-tag {
        background: #E8F4FD;
        color: #2E86AB;
        padding: 4px 10px;
        border-radius: 15px;
        font-size: 0.8em;
        margin: 2px;
        display: inline-block;
        border: 1px solid #2E86AB;
    }
</style>
""", unsafe_allow_html=True)

class EnhancedMahabharataRAG:
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
                st.sidebar.success("✅ Using existing knowledge base")
            except Exception:
                self.collection = self.client.create_collection("mahabharata_complete")
                st.sidebar.info("🔄 Creating new knowledge base")
            
            # Load ALL available knowledge base files
            self.load_all_chunks_efficiently()
            
        except Exception as e:
            st.error(f"❌ Failed to initialize RAG system: {e}")
            raise
    
    def load_all_chunks_efficiently(self):
        """Load ALL available Mahabharata chunk files efficiently"""
        try:
            # Check if knowledge is already loaded
            if self.collection.count() > 50:
                st.session_state.knowledge_loaded = True
                st.session_state.loaded_chunks_count = self.collection.count()
                st.sidebar.success(f"✅ Knowledge base ready: {self.collection.count()} chunks")
                return
            
            with st.spinner("📚 Loading COMPLETE Mahabharata knowledge base..."):
                all_chunks = []
                loaded_files = []
                
                # Load ALL individual chunk files (maha01_chunks.json to maha18_chunks.json)
                chunk_files = glob.glob("data/processed/*_chunks.json")
                chunk_files.extend(glob.glob("data/processed/maha*.json"))
                
                # Also include the complete file if it exists
                complete_file = "data/processed/complete_mahabharata.json"
                if os.path.exists(complete_file):
                    chunk_files.append(complete_file)
                
                # Remove duplicates and sort
                chunk_files = sorted(list(set(chunk_files)))
                
                if not chunk_files:
                    st.error("❌ No chunk files found in data/processed/")
                    st.info("💡 Please process your Mahabharata files first")
                    return
                
                total_files = len(chunk_files)
                progress_bar = st.sidebar.progress(0)
                
                for i, file_path in enumerate(chunk_files):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            chunks = json.load(f)
                        
                        if not isinstance(chunks, list):
                            st.sidebar.warning(f"⚠️ Invalid format in {os.path.basename(file_path)}")
                            continue
                            
                        all_chunks.extend(chunks)
                        loaded_files.append(os.path.basename(file_path))
                        st.sidebar.info(f"📖 Loaded {len(chunks)} from {os.path.basename(file_path)}")
                        
                        # Update progress
                        progress_bar.progress((i + 1) / total_files)
                        
                    except Exception as e:
                        st.sidebar.warning(f"⚠️ Could not load {os.path.basename(file_path)}: {str(e)}")
                
                progress_bar.empty()
                
                if not all_chunks:
                    st.error("❌ No chunks loaded from any file!")
                    return
                
                # Add ALL chunks to vector store
                self.add_all_chunks_to_store(all_chunks)
                
                st.session_state.knowledge_loaded = True
                st.session_state.loaded_chunks_count = len(all_chunks)
                st.sidebar.success(f"✅ COMPLETE knowledge base loaded: {len(all_chunks)} chunks from {len(loaded_files)} files")
                
        except Exception as e:
            st.error(f"❌ Error loading knowledge base: {e}")
    
    def add_all_chunks_to_store(self, chunks: List[Dict]):
        """Add ALL chunks to vector store efficiently"""
        try:
            # Use ALL chunks for maximum coverage
            documents = []
            embeddings = []
            metadatas = []
            ids = []
            
            for i, chunk in enumerate(chunks):
                content = chunk.get('content', '')
                if not content or len(content.strip()) < 10:
                    continue
                    
                documents.append(content)
                metadatas.append({
                    'section_id': chunk.get('section_id', ''),
                    'parva': chunk.get('full_parva', chunk.get('parva', 'UNKNOWN')),
                    'chunk_id': chunk.get('chunk_id', ''),
                    'source_file': chunk.get('source_file', 'unknown'),
                    'word_count': chunk.get('word_count', 0),
                    'global_index': i
                })
                ids.append(f"chunk_{i}")
            
            # Generate embeddings in batches
            batch_size = 100
            total_batches = (len(documents) - 1) // batch_size + 1
            
            st.sidebar.info(f"🔧 Generating embeddings for {len(documents)} chunks...")
            progress_bar = st.sidebar.progress(0)
            
            for i in range(0, len(documents), batch_size):
                batch_end = min(i + batch_size, len(documents))
                batch_docs = documents[i:batch_end]
                
                # Generate embeddings for this batch
                batch_embeddings = self.embedder.encode(batch_docs).tolist()
                embeddings.extend(batch_embeddings)
                
                # Update progress
                progress = (i + batch_size) / len(documents)
                progress_bar.progress(progress)
            
            progress_bar.empty()
            
            # Clear existing collection and add all chunks
            try:
                self.client.delete_collection("mahabharata_complete")
            except:
                pass
            
            self.collection = self.client.create_collection("mahabharata_complete")
            
            # Add in final batch
            self.collection.add(
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            st.sidebar.success(f"📚 Added {len(documents)} chunks to knowledge base")
            
        except Exception as e:
            st.error(f"❌ Error adding chunks to store: {e}")
    
    def retrieve_context(self, query: str, k: int = 12) -> List[Dict]:
        """Enhanced context retrieval with better matching"""
        try:
            query_embedding = self.embedder.encode(query).tolist()
            
            # Get more results to ensure good coverage
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(k * 3, 50),  # Get many results, filter later
                include=['documents', 'metadatas', 'distances']
            )
            
            contexts = []
            for doc, metadata, distance in zip(
                results['documents'][0],
                results['metadatas'][0], 
                results['distances'][0]
            ):
                similarity_score = max(0.0, 1 - distance)
                # Very low threshold to catch all potentially relevant results
                if similarity_score > 0.01:  # Very low threshold
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
    
    def generate_comprehensive_answer(self, question: str, contexts: List[Dict]) -> str:
        """Generate comprehensive answers using ALL available context"""
        if not contexts:
            return self.get_intelligent_fallback(question)
        
        # Extract key information from all contexts
        character_info = self.extract_comprehensive_character_info(question, contexts)
        relationships = self.extract_relationships(question, contexts)
        events = self.extract_relevant_events(question, contexts)
        aliases = self.extract_aliases(question, contexts)
        
        # Build comprehensive answer
        answer_parts = []
        
        # Character-specific answer structure
        if self.is_character_question(question):
            character_name = self.extract_character_name(question)
            answer_parts.append(f"## 👑 {character_name}")
            
            # Aliases and titles
            if aliases:
                answer_parts.append(f"\n**Also known as:** {', '.join(aliases)}")
            
            # Main character information
            if character_info:
                answer_parts.append(f"\n{character_info}")
            else:
                answer_parts.append(f"\n**Character Overview:**\n\nBased on the available texts, {character_name} is a significant figure in the Mahabharata epic.")
            
            # Relationships
            if relationships:
                answer_parts.append(f"\n**Key Relationships:**\n\n{relationships}")
            
            # Events
            if events:
                answer_parts.append(f"\n**Notable Events:**\n\n{events}")
                
        else:
            # General question answer structure
            answer_parts.append("## 📖 Comprehensive Answer")
            
            # Use all relevant contexts
            main_content = self.synthesize_main_content(contexts)
            if main_content:
                answer_parts.append(f"\n{main_content}")
            
            # Additional insights
            additional_info = self.extract_additional_insights(contexts)
            if additional_info:
                answer_parts.append(f"\n**Additional Context:**\n\n{additional_info}")
        
        # Source information
        source_info = self.get_detailed_source_info(contexts)
        answer_parts.append(f"\n{source_info}")
        
        return "\n".join(answer_parts)
    
    def extract_comprehensive_character_info(self, question: str, contexts: List[Dict]) -> str:
        """Extract comprehensive character information from all contexts"""
        character_name = self.extract_character_name(question)
        character_keywords = self.get_character_keywords(character_name)
        
        relevant_sentences = []
        
        for ctx in contexts:
            content = ctx['content']
            sentences = re.split(r'[.!?]+', content)
            
            for sentence in sentences:
                sentence_clean = sentence.strip()
                if (len(sentence_clean) > 20 and 
                    any(keyword in sentence_clean.lower() for keyword in character_keywords)):
                    
                    # Clean and format the sentence
                    formatted_sentence = self.clean_sentence(sentence_clean)
                    if formatted_sentence and formatted_sentence not in relevant_sentences:
                        relevant_sentences.append(formatted_sentence)
        
        # Group and organize the information
        if relevant_sentences:
            return self.organize_character_info(relevant_sentences, character_name)
        else:
            return ""
    
    def get_character_keywords(self, character_name: str) -> List[str]:
        """Get comprehensive keywords for character search"""
        base_name = character_name.lower()
        keywords = [base_name]
        
        # Add common variations and related terms
        character_variations = {
            'draupadi': ['draupadi', 'panchali', 'yajnaseni', 'krishnaa', 'drupada\'s daughter', 'pandavas\' wife'],
            'krishna': ['krishna', 'govinda', 'madhava', 'vasudeva', 'hari', 'narayana', 'vishnu'],
            'arjuna': ['arjuna', 'partha', 'dhananjaya', 'gudakesha', 'savyasachi', 'phalguna'],
            'yudhishthira': ['yudhishthira', 'dharmaraja', 'ajatashatru', 'pandava king'],
            'bhima': ['bhima', 'bhimasena', 'vrikodara', 'wolf-bellied'],
            'karna': ['karna', 'radheya', 'vasusena', 'suryaputra', 'anga raja'],
            'bhishma': ['bhishma', 'devavrata', 'gangaputra', 'pitamaha'],
            'drona': ['drona', 'dronacharya', 'guru drona'],
            'duryodhana': ['duryodhana', 'suyodhana', 'kaurava prince'],
            'kunti': ['kunti', 'pritha', 'pandu\'s wife'],
            'gandhari': ['gandhari', 'dhritarashtra\'s wife']
        }
        
        return character_variations.get(base_name, [base_name])
    
    def clean_sentence(self, sentence: str) -> str:
        """Clean and format sentence for better readability"""
        # Remove extra whitespace
        sentence = re.sub(r'\s+', ' ', sentence.strip())
        
        # Capitalize first letter
        if sentence and sentence[0].islower():
            sentence = sentence[0].upper() + sentence[1:]
            
        return sentence
    
    def organize_character_info(self, sentences: List[str], character_name: str) -> str:
        """Organize character information into coherent paragraphs"""
        # Group sentences by content type
        background_sentences = []
        role_sentences = []
        event_sentences = []
        relationship_sentences = []
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            if any(word in sentence_lower for word in ['born', 'birth', 'son of', 'daughter of', 'prince', 'princess']):
                background_sentences.append(sentence)
            elif any(word in sentence_lower for word in ['wife', 'husband', 'married', 'mother', 'father', 'brother', 'sister']):
                relationship_sentences.append(sentence)
            elif any(word in sentence_lower for word in ['war', 'battle', 'fight', 'killed', 'died', 'event']):
                event_sentences.append(sentence)
            else:
                role_sentences.append(sentence)
        
        # Build organized response
        parts = []
        
        if background_sentences:
            parts.append("**Background:** " + " ".join(background_sentences[:3]))
        
        if role_sentences:
            parts.append("**Role and Significance:** " + " ".join(role_sentences[:4]))
        
        if relationship_sentences:
            parts.append("**Relationships:** " + " ".join(relationship_sentences[:3]))
        
        if event_sentences:
            parts.append("**Key Events:** " + " ".join(event_sentences[:3]))
        
        return "\n\n".join(parts)
    
    def extract_relationships(self, question: str, contexts: List[Dict]) -> str:
        """Extract relationship information"""
        character_name = self.extract_character_name(question)
        relationships = []
        
        relationship_keywords = ['wife', 'husband', 'son', 'daughter', 'father', 'mother', 'brother', 'sister', 'friend', 'enemy']
        
        for ctx in contexts:
            content = ctx['content']
            sentences = re.split(r'[.!?]+', content)
            
            for sentence in sentences:
                sentence_lower = sentence.lower()
                if (character_name.lower() in sentence_lower and 
                    any(keyword in sentence_lower for keyword in relationship_keywords) and
                    len(sentence.strip()) > 15):
                    
                    clean_sentence = self.clean_sentence(sentence.strip())
                    if clean_sentence not in relationships:
                        relationships.append(clean_sentence)
        
        return " • ".join(relationships[:5]) if relationships else ""
    
    def extract_relevant_events(self, question: str, contexts: List[Dict]) -> str:
        """Extract relevant events involving the character"""
        character_name = self.extract_character_name(question)
        events = []
        
        event_keywords = ['war', 'battle', 'fight', 'sacrifice', 'ceremony', 'game', 'exile', 'kingdom']
        
        for ctx in contexts:
            content = ctx['content']
            sentences = re.split(r'[.!?]+', content)
            
            for sentence in sentences:
                sentence_lower = sentence.lower()
                if (character_name.lower() in sentence_lower and 
                    any(keyword in sentence_lower for keyword in event_keywords) and
                    len(sentence.strip()) > 20):
                    
                    clean_sentence = self.clean_sentence(sentence.strip())
                    if clean_sentence not in events:
                        events.append(clean_sentence)
        
        return " • ".join(events[:4]) if events else ""
    
    def extract_aliases(self, question: str, contexts: List[Dict]) -> List[str]:
        """Extract aliases and alternative names for characters"""
        character_name = self.extract_character_name(question)
        
        # Predefined aliases for major characters
        predefined_aliases = {
            'draupadi': ['Panchali', 'Yajnaseni', 'Krishnaa', 'Drupadi'],
            'krishna': ['Govinda', 'Madhava', 'Vasudeva', 'Hari', 'Narayana'],
            'arjuna': ['Partha', 'Dhananjaya', 'Gudakesha', 'Savyasachi', 'Phalguna'],
            'yudhishthira': ['Dharmaraja', 'Ajatashatru'],
            'bhima': ['Bhimasena', 'Vrikodara'],
            'karna': ['Radheya', 'Vasusena', 'Suryaputra'],
            'bhishma': ['Devavrata', 'Gangaputra', 'Pitamaha'],
            'drona': ['Dronacharya'],
            'duryodhana': ['Suyodhana']
        }
        
        aliases = predefined_aliases.get(character_name.lower(), [])
        
        # Also extract from context
        for ctx in contexts:
            content = ctx['content']
            # Look for common alias patterns
            if character_name.lower() in content.lower():
                # Simple pattern matching for aliases
                patterns = [
                    r'also known as ([^.,;]+)',
                    r'called ([^.,;]+)',
                    r'named ([^.,;]+)',
                    r'known as ([^.,;]+)'
                ]
                
                for pattern in patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    for match in matches:
                        alias = match.strip()
                        if (alias.lower() != character_name.lower() and 
                            len(alias) > 2 and 
                            alias not in aliases):
                            aliases.append(alias)
        
        return list(set(aliases))[:8]  # Limit to 8 aliases
    
    def synthesize_main_content(self, contexts: List[Dict]) -> str:
        """Synthesize main content from multiple contexts"""
        if not contexts:
            return ""
        
        # Use the most relevant contexts
        main_sentences = []
        
        for ctx in contexts[:6]:  # Use top 6 contexts
            content = ctx['content']
            sentences = re.split(r'[.!?]+', content)
            
            # Take most meaningful sentences
            meaningful_sentences = [s.strip() for s in sentences if len(s.strip()) > 25]
            if meaningful_sentences:
                main_sentences.extend(meaningful_sentences[:2])
        
        # Remove duplicates and join
        unique_sentences = []
        for sentence in main_sentences:
            if sentence not in unique_sentences:
                unique_sentences.append(sentence)
        
        return " ".join(unique_sentences[:8])  # Limit to 8 sentences
    
    def extract_additional_insights(self, contexts: List[Dict]) -> str:
        """Extract additional insights from remaining contexts"""
        if len(contexts) <= 6:
            return ""
        
        insights = []
        
        for ctx in contexts[6:12]:  # Use next 6 contexts
            content = ctx['content']
            sentences = re.split(r'[.!?]+', content)
            
            meaningful_sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
            if meaningful_sentences:
                insights.extend(meaningful_sentences[:1])
        
        return " ".join(insights[:4]) if insights else ""
    
    def get_detailed_source_info(self, contexts: List[Dict]) -> str:
        """Get detailed information about sources used"""
        if not contexts:
            return "*No specific sources were found for this query.*"
        
        parva_counts = {}
        total_chunks = len(contexts)
        
        for ctx in contexts:
            parva = ctx['metadata']['parva']
            parva_counts[parva] = parva_counts.get(parva, 0) + 1
        
        # Sort parvas by count
        sorted_parvas = sorted(parva_counts.items(), key=lambda x: x[1], reverse=True)
        parva_list = [f"{parva} ({count})" for parva, count in sorted_parvas[:5]]  # Show top 5
        
        source_info = f"*Information synthesized from {total_chunks} sources across {len(parva_counts)} different sections of the Mahabharata.*"
        
        if parva_list:
            source_info += f" *Main sources: {', '.join(parva_list)}*"
        
        return source_info
    
    def is_character_question(self, question: str) -> bool:
        """Check if question is about a character"""
        character_keywords = [
            'draupadi', 'krishna', 'arjuna', 'yudhishthira', 'bhima', 'nakula', 'sahadeva',
            'duryodhana', 'karna', 'bhishma', 'drona', 'vidura', 'kunti', 'gandhari',
            'shakuni', 'dushasana', 'subhadra', 'ulupi', 'chitrangada', 'abhimanyu'
        ]
        question_lower = question.lower()
        return any(char in question_lower for char in character_keywords)
    
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
            'dushasana': 'Dushasana',
            'subhadra': 'Subhadra',
            'ulupi': 'Ulupi',
            'chitrangada': 'Chitrangada',
            'abhimanyu': 'Abhimanyu'
        }
        
        for key, name in character_mapping.items():
            if key in question_lower:
                return name
        
        # Default: extract first word after "about" or the main subject
        if 'about' in question_lower:
            parts = question_lower.split('about')
            if len(parts) > 1:
                first_word = parts[1].strip().split()[0]
                return first_word.title()
        
        return "this character"
    
    def get_intelligent_fallback(self, question: str) -> str:
        """Provide intelligent fallback when no information is found"""
        character_name = self.extract_character_name(question)
        
        fallback = f"""## 🔍 Comprehensive Search Results

**Search Status:** I conducted an extensive search through the available Mahabharata texts but found limited specific information about **{character_name.title()}** in the currently loaded sections.

**📚 Available Knowledge Base:**
- Multiple Mahabharata parvas and sections
- Various character references and events
- Philosophical discussions and narratives

**💡 Suggestions for Better Results:**
• Try asking about more well-documented characters like Krishna, Arjuna, or Draupadi
• Inquire about specific events like the Kurukshetra War or Bhagavad Gita
• Ask about philosophical concepts like Dharma or Karma
• Request information about the Pandava or Kaurava families

**🎯 Popular Questions with Good Coverage:**
- "Tell me about Krishna's role in the Mahabharata"
- "Who are the Pandava brothers and their qualities?"
- "What is the story of Draupadi's marriage?"
- "Explain the concept of Dharma in the epic"

*Note: The system searches across all available text chunks. Some characters or specific details might be mentioned in sections not currently loaded.*"""

        return fallback

    def query(self, question: str) -> Dict:
        """Comprehensive query method with enhanced retrieval"""
        start_time = time.time()
        
        # Enhanced context retrieval
        contexts = self.retrieve_context(question, k=15)  # Increased to 15 for better coverage
        
        retrieval_time = time.time() - start_time
        
        # Generate comprehensive answer
        gen_start = time.time()
        answer = self.generate_comprehensive_answer(question, contexts)
        generation_time = time.time() - gen_start
        
        total_time = time.time() - start_time
        
        # Calculate confidence based on best matches
        confidence = 0.0
        if contexts:
            top_scores = [ctx['similarity_score'] for ctx in contexts[:5]]
            confidence = sum(top_scores) / len(top_scores)
        
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
    st.markdown('<p class="sub-header">Complete Edition • All Available Texts • Enhanced Comprehensive Answers</p>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'rag' not in st.session_state:
        try:
            st.session_state.rag = EnhancedMahabharataRAG()
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
            "Who is Draupadi and what is her significance?",
            "Tell me about Krishna's role in Mahabharata", 
            "Who are the Pandava brothers and their qualities?",
            "What is the Bhagavad Gita about?",
            "Describe the Kurukshetra war",
            "Who is Arjuna and what are his skills?",
            "Explain the concept of Dharma in Mahabharata",
            "What is Yudhishthira known for?",
            "Who is Karna and what is his story?",
            "Tell me about Bhishma's role",
            "What are Draupadi's other names?",
            "Describe the relationship between Krishna and Arjuna"
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
        st.metric("Search Scope", "All Available Texts")
        st.metric("Answer Quality", "Comprehensive")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Main question form
        with st.form("main_question_form"):
            question = st.text_input(
                "Your Question:",
                placeholder="Ask anything about Mahabharata characters, philosophy, events, or relationships...",
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
            with st.spinner("🔍 Conducting comprehensive search across all available texts..."):
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
            st.markdown("### 📊 Search Metrics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("⏱️ Response Time", f"{result['response_time']:.1f}s")
            with col2:
                confidence_color = "🟢" if result['confidence'] > 0.6 else "🟡" if result['confidence'] > 0.3 else "🔴"
                st.metric("📊 Confidence", f"{confidence_color} {result['confidence']:.3f}")
            with col3:
                st.metric("📚 Sources", result['sources_count'])
            with col4:
                st.metric("📖 Text Sections", len(result.get('parvas_used', [])))
            
            # Source information
            if result['sources_count'] > 0:
                st.markdown("### 📖 Sources Used")
                parva_counts = {}
                for source in result['sources']:
                    parva = source['metadata']['parva']
                    parva_counts[parva] = parva_counts.get(parva, 0) + 1
                
                # Show top sources
                sorted_parvas = sorted(parva_counts.items(), key=lambda x: x[1], reverse=True)[:6]
                for parva, count in sorted_parvas:
                    st.markdown(f'<span class="source-badge">{parva}: {count} source{"s" if count > 1 else ""}</span>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🌟 Quick Facts")
        facts = [
            "📚 Multiple Parvas",
            "👑 50+ Characters Covered", 
            "⚔️ Major Events Included",
            "🕉️ Philosophical Depth",
            "📖 Comprehensive Answers",
            "🌍 Ancient Wisdom",
            "💡 Enhanced Search",
            "🎭 Rich Relationships"
        ]
        for fact in facts:
            st.info(fact)
        
        # Recent questions
        if st.session_state.chat_history:
            st.markdown("### 💬 Recent Questions")
            recent_chats = list(reversed(st.session_state.chat_history[-5:]))
            
            for i, chat in enumerate(recent_chats):
                display_question = chat['question'][:35] + "..." if len(chat['question']) > 35 else chat['question']
                
                if st.button(
                    f"Q: {display_question}",
                    key=f"recent_{i}",
                    use_container_width=True,
                    help="Click to ask this question again"
                ):
                    st.session_state.current_question = chat['question']
                    st.session_state.answer_trigger = True
                    st.rerun()
        
        # Character Highlights with aliases
        st.markdown("### 👑 Popular Characters")
        characters_with_aliases = [
            "Draupadi (Panchali, Yajnaseni)",
            "Krishna (Govinda, Madhava)", 
            "Arjuna (Partha, Dhananjaya)",
            "Yudhishthira (Dharmaraja)",
            "Bhima (Vrikodara)",
            "Karna (Radheya, Suryaputra)"
        ]
        for char in characters_with_aliases:
            st.caption(f"• {char}")

if __name__ == "__main__":
    main()