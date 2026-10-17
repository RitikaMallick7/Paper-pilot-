"""Module to parse PDFs and extract text by sections."""
import pypdf
from typing import List, Dict, Tuple
import re


class PDFParser:
    """Parses PDFs and extracts text with section detection."""
    
    # Common section headers in academic papers
    SECTION_PATTERNS = [
        r'^\s*(Abstract|Introduction|Related Work|Background|Methods?|Methodology|'
        r'Experiments?|Results?|Discussion|Conclusion|Conclusions|References|'
        r'Acknowledgments?|Future Work)\s*$',
        r'^\s*\d+\.\s+(Abstract|Introduction|Related Work|Background|Methods?|'
        r'Methodology|Experiments?|Results?|Discussion|Conclusion|Conclusions|'
        r'References|Acknowledgments?|Future Work)\s*$',
    ]
    
    def __init__(self):
        """Initialize the PDF parser."""
        self.section_regex = re.compile('|'.join(self.SECTION_PATTERNS), re.IGNORECASE | re.MULTILINE)
    
    def extract_text(self, pdf_path: str) -> str:
        """
        Extract all text from a PDF.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Full text content of the PDF
        """
        text = ""
        with open(pdf_path, 'rb') as file:
            reader = pypdf.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        
        return text
    
    def split_by_sections(self, text: str) -> Dict[str, str]:
        """
        Split text into logical sections based on headers.
        
        Args:
            text: Full text of the paper
            
        Returns:
            Dictionary mapping section names to their content
        """
        sections = {}
        current_section = "Introduction"
        current_content = []
        
        lines = text.split('\n')
        
        for line in lines:
            # Check if this line is a section header
            match = self.section_regex.match(line.strip())
            
            if match:
                # Save previous section
                if current_content:
                    sections[current_section] = '\n'.join(current_content).strip()
                
                # Start new section
                current_section = match.group(1) or match.group(2) if match.groups() else line.strip()
                current_content = []
            else:
                current_content.append(line)
        
        # Save the last section
        if current_content:
            sections[current_section] = '\n'.join(current_content).strip()
        
        return sections
    
    def parse(self, pdf_path: str) -> Dict[str, str]:
        """
        Parse a PDF and return sections.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary of sections
        """
        text = self.extract_text(pdf_path)
        sections = self.split_by_sections(text)
        
        # If no sections were detected, fall back to treating the whole text as one section
        if not sections or len(sections) == 1:
            return {"Full Text": text}
        
        return sections
