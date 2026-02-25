import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import DocumentLoader
from summarizer import DocumentSummarizer

def test_real_pdf(pdf_path):
    loader = DocumentLoader()
    doc_ai = DocumentSummarizer()

    print(f"\n--- Processing Real PDF: {pdf_path} ---")
    
    # 1. Extract and Chunk
    chunks = loader.extract_and_chunk_pdf(pdf_path)
    
    # 2. Map-Reduce Summarization
    final_summary = doc_ai.summarize_long_document(chunks)

    print("\n" + "="*50)
    print("🏆 FINAL MASTER SUMMARY:")
    print(final_summary)
    print("="*50 + "\n")

if __name__ == "__main__":
    # Replace with your actual PDF name
    test_real_pdf("sample_docs/testing.pdf")