import fitz

class DocumentLoader:
    def __init__(self):
        self.chunk_size_words = 300 # Safe size for T5-Large

    def extract_and_chunk_pdf(self, pdf_path: str):
        """
        Extracts text from a PDF and chunks it into smaller pieces.
        """
        print(f"📄 Extracting text from {pdf_path}...")
        doc = fitz.open(pdf_path)
        full_text = ""
        
        # 1. Extract all text page by page
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            full_text += page.get_text("text") + " "  # type: ignore
            
        # Clean up whitespace
        full_text = " ".join(full_text.split())
        
        if not full_text:
            raise ValueError("No text could be extracted from the PDF. It might be an image-only PDF.")

        # 2. CHUNKING: Split the text into safe chunks of ~300 words
        words = full_text.split()
        chunks = []
        for i in range(0, len(words), self.chunk_size_words):
            chunk = " ".join(words[i : i + self.chunk_size_words])
            chunks.append(chunk)
            
        print(f"🔪 Split document into {len(chunks)} chunks.")
        return chunks