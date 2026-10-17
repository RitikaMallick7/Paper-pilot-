"""Module to fetch papers from arXiv API."""
import arxiv
import os
from typing import List, Dict
from dataclasses import dataclass


@dataclass
class PaperMetadata:
    """Metadata for a research paper."""
    title: str
    arxiv_id: str
    authors: List[str]
    published: str
    pdf_url: str
    abstract: str


class ArxivFetcher:
    """Fetches papers from arXiv API."""
    
    def __init__(self, category: str = "cs.AI", max_results: int = 10):
        """
        Initialize the arXiv fetcher.
        
        Args:
            category: arXiv category (e.g., "cs.AI", "cs.LG")
            max_results: Maximum number of papers to fetch
        """
        self.category = category
        self.max_results = max_results
    
    def fetch_papers(self) -> List[PaperMetadata]:
        """
        Fetch papers from arXiv by category.
        
        Returns:
            List of PaperMetadata objects
        """
        # Build search query for the category
        query = f"cat:{self.category}"
        
        search = arxiv.Search(
            query=query,
            max_results=self.max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        
        papers = []
        for result in search.results():
            paper = PaperMetadata(
                title=result.title,
                arxiv_id=result.entry_id.split("/")[-1],
                authors=[author.name for author in result.authors],
                published=result.published.strftime("%Y-%m-%d"),
                pdf_url=result.pdf_url,
                abstract=result.summary
            )
            papers.append(paper)
        
        return papers
    
    def download_pdf(self, pdf_url: str, save_path: str) -> str:
        """
        Download a PDF from arXiv.
        
        Args:
            pdf_url: URL to the PDF
            save_path: Directory to save the PDF
            
        Returns:
            Path to the downloaded PDF
        """
        import requests
        
        os.makedirs(save_path, exist_ok=True)
        filename = pdf_url.split("/")[-1]
        filepath = os.path.join(save_path, filename)
        
        response = requests.get(pdf_url, stream=True)
        response.raise_for_status()
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return filepath
