import streamlit as st
import sys
import os
import json
import time

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set page config
st.set_page_config(
    page_title="Mahabharata AI Scholar - Complete Edition",
    page_icon="🕉️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium CSS with fancy colors optimized for readability
st.markdown("""
<style>
    /* Main background with subtle gradient */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Header with premium gradient */
    .main-header {
        font-size: 3.5rem;
        background: linear-gradient(45deg, #FF6B35, #F7931E, #667eea);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradientShift 8s ease infinite;
        text-align: center;
        margin-bottom: 0.5rem;
        font-weight: 800;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .sub-header {
        font-size: 1.3rem;
        color: #4A5568;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 300;
        opacity: 0.9;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Premium question container */
    .question-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        padding: 30px;
        border-radius: 20px;
        margin: 20px 0;
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .question-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #FF6B35, #F7931E, #667eea);
    }
    
    /* Luxury answer box with perfect readability */
    .answer-box {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 35px;
        border-radius: 20px;
        margin: 25px 0;
        box-shadow: 0 25px 50px rgba(0, 0, 0, 0.25);
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: #ffffff;
        position: relative;
        overflow: hidden;
        font-family: 'Georgia', serif;
        line-height: 1.7;
        font-size: 1.05rem;
    }
    
    .answer-box::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 6px;
        background: linear-gradient(90deg, #FF6B35, #F7931E, #667eea, #764ba2);
        background-size: 400% 400%;
        animation: gradientShift 6s ease infinite;
    }
    
    /* Elegant source boxes */
    .source-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 25px;
        border-radius: 16px;
        margin: 15px 0;
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.25);
        color: #ffffff;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        position: relative;
        overflow: hidden;
    }
    
    .source-box::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
        transition: left 0.6s;
    }
    
    .source-box:hover::before {
        left: 100%;
    }
    
    .source-box:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 25px 50px rgba(102, 126, 234, 0.3);
    }
    
    /* Premium metric cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        padding: 25px 15px;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.3);
        margin: 8px;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.15);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #2D3748;
        margin-bottom: 8px;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #4A5568;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Luxury sidebar */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #2D3748 0%, #4A5568 100%);
        color: white;
    }
    
    /* Premium buttons */
    .stButton button {
        background: linear-gradient(45deg, #FF6B35, #F7931E);
        color: white;
        border: none;
        padding: 15px 30px;
        border-radius: 50px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        box-shadow: 0 8px 25px rgba(255, 107, 53, 0.3);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .stButton button:hover {
        transform: translateY(-3px) scale(1.05);
        box-shadow: 0 15px 35px rgba(255, 107, 53, 0.4);
        background: linear-gradient(45deg, #F7931E, #FF6B35);
    }
    
    /* Elegant example questions */
    .example-question {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 16px 20px;
        border-radius: 12px;
        margin: 8px 0;
        width: 100%;
        text-align: left;
        font-size: 0.95rem;
        transition: all 0.3s ease;
        cursor: pointer;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .example-question:hover {
        transform: translateX(8px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Luxury fact cards */
    .fact-card {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 18px;
        border-radius: 14px;
        margin: 10px 0;
        box-shadow: 0 8px 25px rgba(252, 182, 159, 0.3);
        border-left: 5px solid #FF6B35;
        color: #2D3748;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .fact-card:hover {
        transform: translateX(5px);
        box-shadow: 0 12px 30px rgba(252, 182, 159, 0.4);
    }
    
    /* Premium parva badges */
    .parva-badge {
        background: linear-gradient(45deg, #FF6B35, #F7931E);
        color: white;
        padding: 6px 16px;
        border-radius: 25px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 4px;
        display: inline-block;
        box-shadow: 0 4px 15px rgba(255, 107, 53, 0.3);
        transition: all 0.3s ease;
    }
    
    .parva-badge:hover {
        transform: scale(1.1);
        box-shadow: 0 6px 20px rgba(255, 107, 53, 0.4);
    }
    
    /* Animated loading */
    .loading-text {
        background: linear-gradient(45deg, #FF6B35, #F7931E, #667eea, #764ba2);
        background-size: 400% 400%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradientShift 3s ease infinite;
        font-weight: 600;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50% }
        50% { background-position: 100% 50% }
        100% { background-position: 0% 50% }
    }
    
    /* Premium tips section */
    .tip-item {
        background: rgba(255, 255, 255, 0.9);
        padding: 14px;
        border-radius: 10px;
        margin: 8px 0;
        border-left: 4px solid #667eea;
        color: #4A5568;
        font-size: 0.9rem;
        transition: all 0.3s ease;
    }
    
    .tip-item:hover {
        transform: translateX(5px);
        background: rgba(255, 255, 255, 0.95);
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(45deg, #FF6B35, #F7931E);
        border-radius: 10px;
        box-shadow: inset 0 0 6px rgba(0,0,0,0.1);
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(45deg, #E55A2B, #E58217);
    }
    
    /* Text readability enhancements */
    .readable-text {
        color: #2D3748;
        line-height: 1.7;
        font-size: 1.05rem;
        font-family: 'Georgia', serif;
    }
    
    .white-text {
        color: #ffffff;
        text-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    
    /* Section headers */
    .section-header {
        background: linear-gradient(45deg, #FF6B35, #F7931E);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        margin-bottom: 15px;
        font-size: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_rag_system():
    """Load the complete Mahabharata RAG system"""
    from src.rag_system import MahabharataRAG
    
    # Initialize with complete data
    rag = MahabharataRAG(use_complete_data=True)
    
    # Try to load complete Mahabharata first
    complete_data_path = "data/processed/complete_mahabharata.json"
    single_data_path = "data/processed/chunks.json"
    
    if os.path.exists(complete_data_path):
        try:
            with open(complete_data_path, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            
            # Use substantial portion for good coverage
            rag.add_documents(chunks[:400])
            return rag, "complete"
            
        except Exception as e:
            st.error(f"❌ Error loading complete data: {e}")
            # Fallback to single file
            return load_single_file(rag, single_data_path), "single"
    else:
        # Fallback to single file
        return load_single_file(rag, single_data_path), "single"

def load_single_file(rag, data_path):
    """Fallback to loading single file"""
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        rag.add_documents(chunks[:100])
        return rag
    except Exception as e:
        st.error(f"❌ Error loading any data: {e}")
        return rag

def create_metric_card(value, label, icon):
    """Create a beautiful metric card"""
    return f"""
    <div class="metric-card">
        <div class="metric-value">{icon} {value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """

def main():
    # Premium header with animated gradient
    st.markdown("""
    <div style='text-align: center; padding: 20px 0;'>
        <h1 class="main-header">🕉️ Mahabharata AI Scholar</h1>
        <p class="sub-header">Complete Edition • All 18 Parvas • Premium AI Experience</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize RAG system
    try:
        rag, data_type = load_rag_system()
        if data_type == "complete":
            st.sidebar.success("✨ Complete Mahabharata Loaded (18 Sacred Parvas)")
        else:
            st.sidebar.warning("📖 Using Adi Parva Only")
    except Exception as e:
        st.error(f"❌ System initialization failed: {e}")
        st.info("💡 Ensure Ollama is running: `ollama serve`")
        return
    
    # Luxury sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Premium Settings")
        
        # Knowledge base info with luxury design
        if data_type == "complete":
            st.markdown("""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                       padding: 20px; border-radius: 16px; color: white; margin: 15px 0;
                       box-shadow: 0 10px 30px rgba(102,126,234,0.3);'>
                <h4 style='margin:0 0 10px 0; color:white; font-size:1.2rem;'>📚 Knowledge Base</h4>
                <p style='margin:5px 0; font-size:0.95em; opacity:0.9;'>✨ Complete Mahabharata</p>
                <p style='margin:5px 0; font-size:0.95em; opacity:0.9;'>🔄 18 Sacred Parvas</p>
                <p style='margin:5px 0; font-size:0.95em; opacity:0.9;'>🤖 Llama 2 7B Chat</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background: linear-gradient(135deg, #fcb69f 0%, #ffecd2 100%); 
                       padding: 20px; border-radius: 16px; color: #2D3748; margin: 15px 0;
                       box-shadow: 0 10px 30px rgba(252,182,159,0.3);'>
                <h4 style='margin:0 0 10px 0; color:#2D3748; font-size:1.2rem;'>📚 Knowledge Base</h4>
                <p style='margin:5px 0; font-size:0.95em; opacity:0.8;'>📖 Adi Parva Only</p>
                <p style='margin:5px 0; font-size:0.95em; opacity:0.8;'>🤖 Llama 2 7B Chat</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Search settings
        st.markdown("### 🔍 Search Configuration")
        col1, col2 = st.columns(2)
        with col1:
            num_sources = st.slider("Sources", 3, 8, 5, help="Number of text sources to reference")
        with col2:
            temperature = st.slider("Creativity", 0.1, 1.0, 0.3, help="AI response creativity level")
        
        # Premium example questions
        st.markdown("### 🌟 Curated Questions")
        example_questions = [
            "Explain the concept of Dharma in Mahabharata",
            "What is the Bhagavad Gita about?",
            "Describe Krishna's role in the Kurukshetra war", 
            "Who are the Pandava brothers?",
            "What caused the Kurukshetra war?",
            "Explain the concept of Karma yoga",
            "Tell me about Draupadi's character",
            "What are the main philosophical teachings?"
        ]
        
        for question in example_questions:
            if st.button(f"• {question}", key=question):
                st.session_state.user_question = question
                st.rerun()
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Premium question input
        st.markdown("""
        <div class="question-container">
            <h3 style='color: #2D3748; margin-bottom: 15px;'>💭 Ask Your Question</h3>
        """, unsafe_allow_html=True)
        
        user_question = st.text_input(
            "Enter your question about Mahabharata:",
            placeholder="e.g., What is the main message of Bhagavad Gita?",
            value=getattr(st.session_state, 'user_question', ''),
            label_visibility="collapsed"
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("🚀 Get AI Answer", type="primary", use_container_width=True):
            if user_question:
                with st.spinner("""
                    <div class='loading-text'>
                        🔍 Searching through ancient wisdom...
                    </div>
                """):
                    start_time = time.time()
                    result = rag.query(user_question, k=num_sources)
                    response_time = time.time() - start_time
                
                # Display premium answer
                st.markdown('<h3 class="section-header">📜 AI Response</h3>', unsafe_allow_html=True)
                st.markdown(f'<div class="answer-box"><div class="white-text">{result["answer"]}</div></div>', unsafe_allow_html=True)
                
                # Luxury metrics
                st.markdown('<h3 class="section-header">📊 Performance Analytics</h3>', unsafe_allow_html=True)
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(create_metric_card(
                        f"{response_time:.2f}s", "Response Time", "⏱️"
                    ), unsafe_allow_html=True)
                with col2:
                    confidence_color = "🟢" if result['confidence'] > 0.5 else "🟡" if result['confidence'] > 0.3 else "🔴"
                    st.markdown(create_metric_card(
                        f"{result['confidence']:.3f}", "Confidence", f"{confidence_color}"
                    ), unsafe_allow_html=True)
                with col3:
                    st.markdown(create_metric_card(
                        result['sources_count'], "Sources Used", "📚"
                    ), unsafe_allow_html=True)
                with col4:
                    st.markdown(create_metric_card(
                        f"{result['timing']['retrieval']:.2f}s", "Retrieval Speed", "⚡"
                    ), unsafe_allow_html=True)
                
                # Source information with luxury design
                if result['sources']:
                    unique_parvas = set(s['metadata']['full_parva'] for s in result['sources'])
                    st.markdown('<h3 class="section-header">📖 Source Information</h3>', unsafe_allow_html=True)
                    parva_html = " ".join([f'<span class="parva-badge">{parva}</span>' for parva in unique_parvas])
                    st.markdown(f"**Knowledge sourced from:** {parva_html}", unsafe_allow_html=True)
                
                # Premium source viewer
                with st.expander(f"📋 Detailed Source Analysis ({len(result['sources'])} sources)", expanded=False):
                    for i, source in enumerate(result['sources']):
                        relevance_score = source['similarity_score']
                        if relevance_score > 0.7:
                            relevance_emoji = "🎯"
                            relevance_text = "Highly Relevant"
                        elif relevance_score > 0.4:
                            relevance_emoji = "✅"
                            relevance_text = "Relevant"
                        else:
                            relevance_emoji = "📝"
                            relevance_text = "Contextual"
                        
                        st.markdown(f"""
                        <div class="source-box">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                                <h4 style="margin:0; color:white; font-size:1.1rem;">
                                    Source {i+1} • {source['metadata']['full_parva']}
                                </h4>
                                <span style="background: rgba(255,255,255,0.2); color: white; padding: 6px 14px; 
                                       border-radius: 15px; font-size: 0.85rem; font-weight: 600;">
                                    {relevance_emoji} {relevance_text} ({relevance_score:.3f})
                                </span>
                            </div>
                            <p style="margin:0; color:white; opacity: 0.95; font-size: 1rem; line-height: 1.6;">
                                {source['content'][:280]}...
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.warning("📝 Please enter a question to begin")
    
    with col2:
        # Premium facts section
        st.markdown("### 🏆 Epic Insights")
        facts = [
            "📖 18 Sacred Parvas",
            "👑 100+ Legendary Characters", 
            "⚔️ Great Kurukshetra War",
            "🕉️ Divine Bhagavad Gita",
            "📜 World's Longest Epic",
            "🌍 Ancient Indian Wisdom",
            "📚 Philosophical Treasure",
            "🎭 Complex Moral Dilemmas"
        ]
        
        for fact in facts:
            st.markdown(f'<div class="fact-card">{fact}</div>', unsafe_allow_html=True)
        
        # Premium tips
        st.markdown("### 💎 Expert Tips")
        tips = [
            "Use specific character names",
            "Ask about philosophical concepts",
            "Request event explanations", 
            "Be precise for better answers",
            "Explore moral teachings",
            "Compare character journeys"
        ]
        
        for tip in tips:
            st.markdown(f'<div class="tip-item">✨ {tip}</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()