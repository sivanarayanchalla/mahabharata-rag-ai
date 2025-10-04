import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pytest
from src.data_processor import MahabharataDataProcessor
from src.rag_system import MahabharataRAG

class TestMahabharataRAG:
    def test_data_processor_initialization(self):
        """Test that data processor initializes correctly"""
        processor = MahabharataDataProcessor()
        assert processor is not None
        assert processor.sections == []
    
    def test_section_extraction(self):
        """Test section extraction from sample text"""
        processor = MahabharataDataProcessor()
        
        sample_text = """
        SECTION I
        
        Some content here about Mahabharata.
        
        SECTION II
        
        More content here.
        """
        
        sections = processor.extract_sections(sample_text)
        assert len(sections) == 2
        assert sections[0]['section_id'] == 'I'
        assert sections[1]['section_id'] == 'II'
    
    def test_chunk_creation(self):
        """Test chunk creation from sections"""
        processor = MahabharataDataProcessor()
        
        sections = [
            {
                'section_id': 'I',
                'content': 'word ' * 250,  # 250 words
                'parva': 'ADI PARVA'
            }
        ]
        
        chunks = processor.chunk_sections(sections, chunk_size=100)
        assert len(chunks) == 3  # Should create 3 chunks from 250 words
        assert all('chunk_id' in chunk for chunk in chunks)

def test_rag_initialization():
    """Test RAG system initialization"""
    # This will only work if Ollama is running
    try:
        rag = MahabharataRAG()
        assert rag is not None
    except Exception as e:
        pytest.skip(f"Ollama not available: {e}")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])