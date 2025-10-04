import re
import json
from typing import List, Dict, Any
import os

class MahabharataDataProcessor:
    def __init__(self):
        self.sections = []
    
    def load_text_file(self, file_path: str) -> str:
        """Load Mahabharata text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            print(f"✅ Loaded {len(content)} characters from {file_path}")
            return content
        except Exception as e:
            print(f"❌ Error loading file: {e}")
            return ""
    
    def extract_sections(self, content: str) -> List[Dict[str, Any]]:
        """Extract sections from the text"""
        # Pattern to match sections like "SECTION I", "SECTION XXV", etc.
        section_pattern = r'SECTION\s+([IVXLCDM]+)\s*\n(.*?)(?=SECTION\s+[IVXLCDM]+\s*\n|\Z)'
        
        matches = re.finditer(section_pattern, content, re.DOTALL)
        sections = []
        
        for match in matches:
            section_num = match.group(1)
            section_content = match.group(2).strip()
            
            # Detect parva from context
            parva = self._detect_parva(section_content)
            
            sections.append({
                'section_id': section_num,
                'content': section_content,
                'parva': parva,
                'word_count': len(section_content.split())
            })
        
        print(f"📖 Extracted {len(sections)} sections")
        return sections
    
    def _detect_parva(self, content: str) -> str:
        """Detect which parva the section belongs to"""
        content_upper = content.upper()
        
        if 'ADI PARVA' in content_upper:
            return 'ADI PARVA'
        elif 'SABHA PARVA' in content_upper:
            return 'SABHA PARVA'
        elif 'ARANYA PARVA' in content_upper:
            return 'ARANYA PARVA'
        elif 'VIRATA PARVA' in content_upper:
            return 'VIRATA PARVA'
        else:
            return 'UNKNOWN PARVA'
    
    def chunk_sections(self, sections: List[Dict], chunk_size: int = 200) -> List[Dict]:
        """Split sections into smaller chunks for better retrieval"""
        chunks = []
        chunk_id = 0
        
        for section in sections:
            text = section['content']
            words = text.split()
            
            # Create chunks of specified word count
            for i in range(0, len(words), chunk_size):
                chunk_words = words[i:i + chunk_size]
                chunk_text = ' '.join(chunk_words)
                
                chunks.append({
                    'chunk_id': f"{section['section_id']}_{chunk_id}",
                    'section_id': section['section_id'],
                    'parva': section['parva'],
                    'content': chunk_text,
                    'word_count': len(chunk_words),
                    'chunk_index': chunk_id
                })
                chunk_id += 1
        
        print(f"✂️  Created {len(chunks)} chunks from {len(sections)} sections")
        return chunks
    
    def save_processed_data(self, chunks: List[Dict], output_path: str):
        """Save processed chunks to JSON"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved processed data to {output_path}")
    
    def process_file(self, input_path: str, output_path: str, max_chunks: int = 100):
        """Complete processing pipeline"""
        print("🚀 Starting data processing...")
        
        # Load raw text
        content = self.load_text_file(input_path)
        if not content:
            return []
        
        # Extract sections
        sections = self.extract_sections(content)
        
        # Create chunks
        chunks = self.chunk_sections(sections)
        
        # Limit chunks for development
        if len(chunks) > max_chunks:
            chunks = chunks[:max_chunks]
            print(f"🔬 Using {max_chunks} chunks for development")
        
        # Save processed data
        self.save_processed_data(chunks, output_path)
        
        return chunks