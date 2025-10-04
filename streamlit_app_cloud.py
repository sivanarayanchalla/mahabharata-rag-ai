import streamlit as st
import json
import os
import requests
from sentence_transformers import SentenceTransformer
import chromadb
from typing import List, Dict
import time
import re

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
        font-size: 2.8rem;
        color: #FF6B35;
        text-align: center;
        margin-bottom: 0.5rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.3rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .answer-box {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 25px;
        border-radius: 15px;
        border-left: 6px solid #FF6B35;
        margin: 15px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        font-size: 1.1em;
        line-height: 1.6;
    }
    .source-badge {
        background-color: #FF6B35;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85em;
        margin: 5px;
        display: inline-block;
    }
    .metric-box {
        background: white;
        padding: 15px;
        border-radius: 10px;
        border: 2px solid #e9ecef;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .question-box {
        background: #e8f4fd;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #007bff;
    }
    .highlight {
        background-color: #fff3cd;
        padding: 2px 4px;
        border-radius: 3px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

class EnhancedCloudMahabharataRAG:
    def __init__(self):
        # Load embedding model
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize vector store
        self.client = chromadb.EphemeralClient()
        self.collection = self.client.create_collection("mahabharata_cloud_enhanced")
        
        # Load pre-processed data
        self.load_knowledge_base()
        
    def load_knowledge_base(self):
        """Load pre-processed Mahabharata chunks"""
        try:
            with open('data/processed/complete_mahabharata.json', 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            
            # Add to vector store
            documents = [chunk['content'] for chunk in chunks[:300]]  # Increased to 300 chunks
            embeddings = self.embedder.encode(documents).tolist()
            metadatas = [{
                'section_id': chunk['section_id'],
                'parva': chunk.get('full_parva', chunk['parva']),
                'chunk_id': chunk['chunk_id'],
                'source_file': chunk.get('source_file', 'unknown')
            } for chunk in chunks[:300]]
            
            self.collection.add(
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
                ids=[chunk['chunk_id'] for chunk in chunks[:300]]
            )
            
            st.sidebar.success(f"✅ Loaded {len(documents)} knowledge chunks")
            
        except Exception as e:
            st.error(f"❌ Error loading knowledge base: {e}")
    
    def retrieve_context(self, query: str, k: int = 6) -> List[Dict]:
        """Retrieve relevant context with better filtering"""
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
            # Filter out low similarity results
            if similarity_score > 0.1:  # Minimum similarity threshold
                contexts.append({
                    'content': doc,
                    'metadata': metadata,
                    'similarity_score': similarity_score
                })
        
        return contexts
    
    def build_enhanced_prompt(self, question: str, contexts: List[Dict]) -> str:
        """Build a much better prompt for high-quality answers"""
        
        # Build structured context
        context_parts = []
        for i, ctx in enumerate(contexts, 1):
            context_parts.append(f"""SOURCE {i} [From {ctx['metadata']['parva']}]:
{ctx['content']}""")
        
        context_str = "\n\n" + "="*50 + "\n".join(context_parts) + "\n" + "="*50
        
        prompt = f"""You are an expert scholar of the Mahabharata. Based EXCLUSIVELY on the provided context, provide a comprehensive, well-structured answer.

CONTEXT FROM MAHABHARATA:{context_str}

QUESTION: {question}

INSTRUCTIONS:
1. Provide a clear, detailed answer using ONLY the information from the context
2. Structure your answer with proper paragraphs and logical flow
3. Highlight key names, concepts, and events using **bold** or emphasis
4. If relevant, mention which parva(s) the information comes from
5. If the context doesn't contain enough information, acknowledge this limitation
6. Make the answer engaging and informative

ANSWER:"""
        
        return prompt
    
    def generate_enhanced_answer(self, question: str, contexts: List[Dict]) -> str:
        """Generate high-quality answer using multiple fallback strategies"""
        
        if not contexts:
            return "I couldn't find specific information about this topic in the available Mahabharata text. Please try rephrasing your question or ask about a different aspect of the epic."
        
        prompt = self.build_enhanced_prompt(question, contexts)
        
        # Strategy 1: Try Hugging Face
        try:
            answer = self.try_hugging_face_enhanced(prompt)
            if answer and len(answer) > 50:  # Valid answer check
                return self.post_process_answer(answer)
        except Exception as e:
            st.sidebar.warning(f"⚠️ Cloud API: {str(e)}")
        
        # Strategy 2: Enhanced rule-based answer
        return self.enhanced_rule_based_answer(question, contexts)
    
    def try_hugging_face_enhanced(self, prompt: str) -> str:
        """Try Hugging Face with better parameters"""
        try:
            # Using a smaller, faster model for free tier
            API_URL = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-large"
            headers = {"Authorization": f"Bearer {os.getenv('HF_TOKEN', '')}"}
            
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 400,
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "do_sample": True,
                    "return_full_text": False
                }
            }
            
            response = requests.post(API_URL, headers=headers, json=payload, timeout=45)
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get('generated_text', '').strip()
            
        except Exception:
            raise Exception("Hugging Face unavailable")
    
    def enhanced_rule_based_answer(self, question: str, contexts: List[Dict]) -> str:
        """Much improved rule-based answer generator"""
        
        # Group contexts by parva
        contexts_by_parva = {}
        for ctx in contexts:
            parva = ctx['metadata']['parva']
            if parva not in contexts_by_parva:
                contexts_by_parva[parva] = []
            contexts_by_parva[parva].append(ctx)
        
        # Build comprehensive answer
        answer_parts = []
        answer_parts.append("## 📖 Based on the Mahabharata Text\n\n")
        
        # Add main answer from best source
        if contexts:
            best_context = max(contexts, key=lambda x: x['similarity_score'])
            main_content = self.extract_key_information(best_context['content'], question)
            answer_parts.append(f"{main_content}\n\n")
        
        # Add supporting information from other parvas
        if len(contexts_by_parva) > 1:
            answer_parts.append("### Additional Context:\n\n")
            for parva, parva_contexts in list(contexts_by_parva.items())[:3]:
                if len(parva_contexts) > 0:
                    sample_content = parva_contexts[0]['content'][:150] + "..."
                    answer_parts.append(f"• **{parva}**: {sample_content}\n")
        
        answer_parts.append(f"\n*This information is synthesized from {len(contexts)} sources across {len(contexts_by_parva)} different parvas of the Mahabharata.*")
        
        return "\n".join(answer_parts)
    
    def extract_key_information(self, text: str, question: str) -> str:
        """Extract and format key information from text"""
        # Simple extraction of key sentences
        sentences = re.split(r'[.!?]+', text)
        relevant_sentences = []
        
        question_keywords = set(question.lower().split())
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20:  # Meaningful sentence
                sentence_lower = sentence.lower()
                # Check if sentence contains keywords or seems relevant
                keyword_matches = sum(1 for keyword in question_keywords if keyword in sentence_lower)
                if keyword_matches > 0 or any(name in sentence_lower for name in ['pandava', 'kaurava', 'krishna', 'arjuna', 'yudhishthira', 'bhima']):
                    relevant_sentences.append(sentence)
        
        # Limit to 3 most relevant sentences
        relevant_sentences = relevant_sentences[:3]
        
        if relevant_sentences:
            return " ".join(relevant_sentences) + "."
        else:
            return text[:300] + "..."
    
    def post_process_answer(self, answer: str) -> str:
        """Clean up and enhance the generated answer"""
        # Remove any prompt remnants
        answer = re.sub(r'.*ANSWER:\s*', '', answer, flags=re.IGNORECASE)
        
        # Ensure proper formatting
        answer = answer.strip()
        
        # Add emphasis to key terms
        key_terms = ['Pandava', 'Kaurava', 'Krishna', 'Arjuna', 'Dharma', 'Karma', 'Bhagavad Gita', 
                    'Kurukshetra', 'Draupadi', 'Bhishma', 'Drona', 'Karna']
        
        for term in key_terms:
            answer = re.sub(r'\b' + re.escape(term) + r'\b', f'**{term}**', answer)
        
        return answer
    
    def query(self, question: str) -> Dict:
        """Enhanced main query method"""
        start_time = time.time()
        
        # Retrieve context with more sources
        contexts = self.retrieve_context(question, k=6)
        retrieval_time = time.time() - start_time
        
        if not contexts:
            return {
                'answer': "## 🔍 No Specific Information Found\n\nI couldn't find detailed information about this specific topic in the available Mahabharata text. \n\n**Suggestions:**\n- Try rephrasing your question\n- Ask about major characters or events\n- Be more specific about what you're looking for",
                'sources': [],
                'confidence': 0.0,
                'response_time': retrieval_time,
                'sources_count': 0
            }
        
        # Generate enhanced answer
        gen_start = time.time()
        answer = self.generate_enhanced_answer(question, contexts)
        generation_time = time.time() - gen_start
        
        total_time = time.time() - start_time
        
        # Calculate confidence based on similarity scores
        confidence_scores = [ctx['similarity_score'] for ctx in contexts]
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if contexts else 0.0
        
        return {
            'answer': answer,
            'sources': contexts,
            'confidence': avg_confidence,
            'response_time': total_time,
            'sources_count': len(contexts),
            'parvas_used': list(set(ctx['metadata']['parva'] for ctx in contexts))
        }

def main():
    st.markdown('<h1 class="main-header">🕉️ Mahabharata AI Scholar</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Cloud Edition - Enhanced Answers & Professional Insights</p>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'rag' not in st.session_state:
        with st.spinner("🔄 Loading Enhanced Mahabharata Knowledge Base..."):
            st.session_state.rag = EnhancedCloudMahabharataRAG()
        st.success("✅ Enhanced System Ready!")
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ System Settings")
        st.info("""
        **Model:** Enhanced Cloud RAG
        **Knowledge:** Complete Mahabharata (18 Parvas)
        **Features:** Multi-source synthesis
        **Status:** ✅ Active
        """)
        
        st.header("🎯 Smart Questions")
        examples = [
            "Who are the Pandava brothers and their unique qualities?",
            "Explain the philosophical teachings of Bhagavad Gita",
            "Describe Krishna's role in the Mahabharata war",
            "What caused the Kurukshetra war between cousins?",
            "Explain the concept of Dharma with examples",
            "Tell me about Arjuna's skills and achievements",
            "What is the significance of Draupadi's character?"
        ]
        
        for example in examples:
            if st.button(f"• {example}", use_container_width=True, key=example):
                st.session_state.current_question = example
        
        st.header("📊 Performance")
        if st.session_state.chat_history:
            avg_confidence = sum(chat['confidence'] for chat in st.session_state.chat_history) / len(st.session_state.chat_history)
            st.metric("Average Confidence", f"{avg_confidence:.3f}")
            st.metric("Questions Asked", len(st.session_state.chat_history))
    
    # Main interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        question = st.text_input(
            "**Your Question About Mahabharata:**",
            placeholder="Ask about characters, philosophy, events, or teachings...",
            value=getattr(st.session_state, 'current_question', ''),
            key="question_input"
        )
        
        if st.button("🚀 Get Enhanced Answer", type="primary", use_container_width=True):
            if question:
                with st.spinner("🔍 Analyzing across all 18 parvas..."):
                    result = st.session_state.rag.query(question)
                
                # Store in history
                st.session_state.chat_history.append({
                    'question': question,
                    'answer': result['answer'],
                    'timestamp': time.time(),
                    'confidence': result['confidence'],
                    'sources_count': result['sources_count']
                })
                
                # Display enhanced results
                st.markdown("## 📜 Comprehensive Answer")
                st.markdown(f'<div class="answer-box">{result["answer"]}</div>', unsafe_allow_html=True)
                
                # Enhanced Metrics
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("⏱️ Response Time", f"{result['response_time']:.2f}s")
                with col2:
                    st.metric("📊 Confidence", f"{result['confidence']:.3f}")
                with col3:
                    st.metric("📚 Sources", result['sources_count'])
                with col4:
                    st.metric("📖 Parvas", len(result.get('parvas_used', [])))
                with col5:
                    st.metric("🎯 Method", "Enhanced RAG")
                
                # Source breakdown
                if result['sources']:
                    st.markdown("### 📖 Source Breakdown")
                    parva_counts = {}
                    for source in result['sources']:
                        parva = source['metadata']['parva']
                        parva_counts[parva] = parva_counts.get(parva, 0) + 1
                    
                    for parva, count in parva_counts.items():
                        st.markdown(f'<span class="source-badge">{parva}: {count} sources</span>', unsafe_allow_html=True)
                    
                    # Detailed sources
                    with st.expander(f"🔍 View Detailed Sources ({len(result['sources'])})"):
                        for i, source in enumerate(result['sources']):
                            st.markdown(f"**Source {i+1}** - {source['metadata']['parva']} (Relevance: {source['similarity_score']:.3f})")
                            st.markdown(f"*{source['content'][:200]}...*")
                            st.divider()
            else:
                st.warning("Please enter a question about the Mahabharata!")
    
    with col2:
        st.markdown("### 🌟 Quick Facts")
        facts = [
            "📚 18 Sacred Parvas",
            "👑 100+ Epic Characters", 
            "⚔️ Dharma Yuddha (Righteous War)",
            "🕉️ Bhagavad Gita Wisdom",
            "🎯 Complex Moral Dilemmas",
            "🌍 Ancient Indian Heritage",
            "📖 World's Longest Epic",
            "💡 Philosophical Treasury"
        ]
        for fact in facts:
            st.info(fact)
        
        # Enhanced chat history
        if st.session_state.chat_history:
            st.markdown("### 💭 Recent Questions")
            for chat in reversed(st.session_state.chat_history[-4:]):
                confidence_color = "🟢" if chat['confidence'] > 0.5 else "🟡" if chat['confidence'] > 0.3 else "🔴"
                st.caption(f"{confidence_color} **{chat['question'][:25]}...**")
        
        st.markdown("### 🆓 Free Features")
        st.success("""
        **Enhanced Capabilities:**
        • Multi-parva synthesis
        • Professional formatting  
        • Source verification
        • Confidence scoring
        • Historical tracking
        • Zero cost access
        """)

if __name__ == "__main__":
    main()