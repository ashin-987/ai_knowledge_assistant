"""
Document Processor - Handles reading and chunking documents
Author: Your Name
"""

from pathlib import Path
from typing import List
import PyPDF2

class DocumentProcessor:
    def __init__(self, chunk_size=1200, chunk_overlap=300):
        """
        Initialize the document processor.
        
        Args:
            chunk_size: Maximum characters per chunk
            chunk_overlap: Characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def load_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF file."""
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except Exception as e:
            print(f"Error reading PDF {pdf_path}: {e}")
            return ""
    
    def load_txt(self, txt_path: str) -> str:
        """Load a text file."""
        try:
            with open(txt_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            print(f"Error reading text file {txt_path}: {e}")
            return ""
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + self.chunk_size
            chunk = text[start:end]
            
            if chunk.strip():
                chunks.append(chunk.strip())
            
            start += self.chunk_size - self.chunk_overlap
        
        return chunks
    
    def process_directory(self, directory: str) -> List[dict]:
        """
        Process all PDFs and text files in a directory.
        
        Returns:
            List of dictionaries containing chunks and metadata
        """
        documents = []
        
        for file_path in Path(directory).rglob('*'):
            if file_path.suffix.lower() == '.pdf':
                print(f"Processing PDF: {file_path.name}")
                text = self.load_pdf(str(file_path))
                chunks = self.chunk_text(text)
                
                for i, chunk in enumerate(chunks):
                    documents.append({
                        'text': chunk,
                        'source': file_path.name,
                        'chunk_id': i
                    })
            
            elif file_path.suffix.lower() == '.txt':
                print(f"Processing TXT: {file_path.name}")
                text = self.load_txt(str(file_path))
                chunks = self.chunk_text(text)
                
                for i, chunk in enumerate(chunks):
                    documents.append({
                        'text': chunk,
                        'source': file_path.name,
                        'chunk_id': i
                    })
        
        print(f"✅ Total chunks created: {len(documents)}")
        return documents