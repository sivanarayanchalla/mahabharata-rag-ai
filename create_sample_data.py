# create_sample_data.py
import json
import os

# Create sample Mahabharata data for cloud deployment
sample_data = [
    {
        "chunk_id": "adi_1",
        "section_id": "I", 
        "parva": "ADI PARVA",
        "full_parva": "ADI PARVA",
        "content": "The Mahabharata begins with the story of King Shantanu and Ganga, and the birth of Bhishma. This foundational parva establishes the lineages of both Kauravas and Pandavas, setting the stage for the great epic war.",
        "word_count": 28
    },
    {
        "chunk_id": "adi_2", 
        "section_id": "II",
        "parva": "ADI PARVA",
        "full_parva": "ADI PARVA",
        "content": "The five Pandava brothers - Yudhishthira, Bhima, Arjuna, Nakula, and Sahadeva - were born to Kunti and Madri through divine blessings. Each brother possessed unique qualities that would shape their destinies.",
        "word_count": 32
    },
    {
        "chunk_id": "bhagavad_1",
        "section_id": "I",
        "parva": "BHISHMA PARVA", 
        "full_parva": "BHISHMA PARVA",
        "content": "The Bhagavad Gita is contained within the Bhishma Parva, where Lord Krishna imparts spiritual wisdom to Arjuna on the battlefield of Kurukshetra. It covers Dharma, Karma Yoga, and the nature of the soul.",
        "word_count": 35
    }
]

os.makedirs('data/processed', exist_ok=True)
with open('data/processed/complete_mahabharata.json', 'w') as f:
    json.dump(sample_data, f, indent=2)

print("✅ Sample data created for cloud deployment!")