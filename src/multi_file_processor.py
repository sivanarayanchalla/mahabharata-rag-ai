import os
import json
import re
from typing import List, Dict, Any
from .data_processor import MahabharataDataProcessor

class MultiFileMahabharataProcessor:
    def __init__(self):
        self.processor = MahabharataDataProcessor()
        self.parva_mapping = self._create_parva_mapping()
    
    def _create_parva_mapping(self):
        """Map file names to Parva names"""
        return {
            'maha01.txt': 'ADI PARVA',
            'maha02.txt': 'SABHA PARVA', 
            'maha03.txt': 'ARANYA PARVA',
            'maha04.txt': 'VIRATA PARVA',
            'maha05.txt': 'UDYOGA PARVA',
            'maha06.txt': 'BHISHMA PARVA',
            'maha07.txt': 'DRONA PARVA',
            'maha08.txt': 'KARNA PARVA',
            'maha09.txt': 'SHALYA PARVA',
            'maha10.txt': 'SAUPTIKA PARVA',
            'maha11.txt': 'STRI PARVA',
            'maha12.txt': 'SHANTI PARVA',
            'maha13.txt': 'ANUSHASANA PARVA',
            'maha14.txt': 'ASHVAMEDHIKA PARVA',
            'maha15.txt': 'ASHRAMAVASIKA PARVA',
            'maha16.txt': 'MOUSALA PARVA',
            'maha17.txt': 'MAHAPRASTHANIKA PARVA',
            'maha18.txt': 'SVARGAROHANIKA PARVA'
        }
    
    def get_available_files(self, raw_data_path: str) -> List[str]:
        """Get all Mahabharata files in raw data directory"""
        if not os.path.exists(raw_data_path):
            print(f"❌ Raw data path not found: {raw_data_path}")
            return []
        
        all_files = os.listdir(raw_data_path)
        mahabharata_files = [f for f in all_files if f.startswith('maha') and f.endswith('.txt')]
        mahabharata_files.sort()  # Sort to maintain order
        
        print(f"📚 Found {len(mahabharata_files)} Mahabharata files:")
        for file in mahabharata_files:
            parva_name = self.parva_mapping.get(file, 'UNKNOWN PARVA')
            print(f"   - {file} → {parva_name}")
        
        return mahabharata_files
    
    def process_all_files(self, raw_data_path: str, output_path: str, max_chunks_per_file: int = 50) -> List[Dict]:
        """Process all Mahabharata files"""
        print("🚀 PROCESSING ALL MAHABHARATA FILES")
        print("=" * 50)
        
        mahabharata_files = self.get_available_files(raw_data_path)
        if not mahabharata_files:
            return []
        
        all_chunks = []
        total_files = len(mahabharata_files)
        
        for i, filename in enumerate(mahabharata_files, 1):
            print(f"\n📖 Processing file {i}/{total_files}: {filename}")
            
            file_path = os.path.join(raw_data_path, filename)
            parva_name = self.parva_mapping.get(filename, 'UNKNOWN PARVA')
            
            if not os.path.exists(file_path):
                print(f"❌ File not found: {file_path}")
                continue
            
            # Process individual file
            chunks = self.processor.process_file(
                file_path, 
                f"data/processed/{filename.replace('.txt', '_chunks.json')}",
                max_chunks_per_file
            )
            
            # Add parva metadata to each chunk
            for chunk in chunks:
                chunk['source_file'] = filename
                chunk['full_parva'] = parva_name
                chunk['global_chunk_id'] = f"{filename}_{chunk['chunk_id']}"
            
            all_chunks.extend(chunks)
            print(f"✅ Added {len(chunks)} chunks from {parva_name}")
        
        # Save combined chunks
        self._save_combined_chunks(all_chunks, output_path)
        
        # Print summary
        self._print_summary(all_chunks, mahabharata_files)
        
        return all_chunks
    
    def _save_combined_chunks(self, chunks: List[Dict], output_path: str):
        """Save all chunks to a combined file"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Saved {len(chunks)} total chunks to {output_path}")
    
    def _print_summary(self, chunks: List[Dict], files_processed: List[str]):
        """Print processing summary"""
        print("\n" + "=" * 50)
        print("📊 PROCESSING SUMMARY")
        print("=" * 50)
        
        # Chunks per parva
        chunks_by_parva = {}
        for chunk in chunks:
            parva = chunk['full_parva']
            chunks_by_parva[parva] = chunks_by_parva.get(parva, 0) + 1
        
        print(f"📚 Total Files Processed: {len(files_processed)}")
        print(f"📄 Total Chunks Created: {len(chunks)}")
        print(f"📖 Chunks per Parva:")
        for parva, count in sorted(chunks_by_parva.items()):
            print(f"   - {parva}: {count} chunks")
        
        total_words = sum(chunk['word_count'] for chunk in chunks)
        print(f"📝 Total Words: {total_words:,}")
        
        print("✅ All Mahabharata files processed successfully!")

# Quick usage function
def process_complete_mahabharata(max_chunks_per_file=30):
    """One-function call to process entire Mahabharata"""
    processor = MultiFileMahabharataProcessor()
    chunks = processor.process_all_files(
        raw_data_path="data/raw",
        output_path="data/processed/complete_mahabharata.json",
        max_chunks_per_file=max_chunks_per_file
    )
    return chunks

if __name__ == "__main__":
    process_complete_mahabharata()