import os
import requests
import tempfile
from datetime import datetime
from typing import (
    List,
    Optional,
    Dict,
    Union,
    Any,
)
from functools import lru_cache

import fitz  # PyMuPDF
import arxiv
from langchain.tools import tool
from langchain_core.pydantic_v1 import (
    BaseModel,
    Field,
)
from langchain_community.retrievers import ArxivRetriever


class ArxivMetadata(BaseModel):
    """Schema for arXiv paper metadata"""
    entry_id: str = Field(..., description="The unique identifier of the paper", alias="Entry_ID")
    published: datetime = Field(..., description="Publication date", alias="Published")
    title: str = Field(..., description="Title of the paper", alias="Title")
    authors: str = Field(..., description="Authors of the paper", alias="Authors")

    class Config:
        allow_population_by_field_name = True


class RetrievedArxivDoc(BaseModel):
    """Schema for retrieved arXiv document"""
    metadata: ArxivMetadata
    page_content: Optional[str] = Field(None, description="The content of the paper")

    class Config:
        arbitrary_types_allowed = True

    def dict(self, *args, **kwargs):
        d = super().dict(*args, **kwargs)
        d['metadata']['Published'] = d['metadata']['Published'].isoformat()
        return d


class ArxivPaperRetriever:
    """Helper class for retrieving and processing arXiv papers."""

    @staticmethod
    def download_and_extract_text(pdf_url: str) -> str:
        """
        Download PDF and extract text content.

        Args:
            pdf_url: URL of the PDF to download

        Returns:
            Extracted text from the PDF

        Raises:
            requests.RequestException: If PDF download fails
            ValueError: If text extraction fails
        """
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_pdf:
            try:
                response = requests.get(pdf_url, timeout=30)
                response.raise_for_status()
                temp_pdf.write(response.content)
                temp_pdf_path = temp_pdf.name

                doc = fitz.open(temp_pdf_path)
                text = " ".join(page.get_text() for page in doc)
                doc.close()
                return text

            except requests.RequestException as e:
                raise ValueError(f"Failed to download PDF: {str(e)}")
            except Exception as e:
                raise ValueError(f"Failed to extract text: {str(e)}")
            finally:
                try:
                    os.unlink(temp_pdf_path)
                except (OSError, UnboundLocalError):
                    pass


@tool
def retrieve_arxiv_papers(
        query: str,
        load_max_docs: int = 2,
) -> List[Dict[str, Any]]:
    """
    Retrieve top papers from arXiv based on the query.

    Args:
        query: Search query string
        load_max_docs: Maximum number of documents to retrieve (default: 2)

    Returns:
        List of retrieved arXiv documents

    Raises:
        ValueError: If retrieval fails
    """
    try:
        retriever = ArxivRetriever(
            load_max_docs=load_max_docs,
            get_full_documents=False,
        )
        response = retriever.invoke(query)

        return [
            {
                'metadata': {
                    **doc.metadata,
                    'Published': str(doc.metadata['Published'])
                },
                'page_content': doc.page_content
            }
            for doc in response
        ]
    except Exception as e:
        raise ValueError(f"Failed to retrieve papers: {str(e)}")


@tool
def retrieve_arxiv_paper_by_id(paper_id: str) -> Dict[str, Any]:
    """
    Retrieve paper information and full content from arXiv based on its ID.

    Args:
        paper_id: arXiv paper ID (e.g., '2307.09288' or 'quant-ph/0201082')

    Returns:
        Dictionary containing paper metadata, abstract, and full content

    Raises:
        ValueError: If paper retrieval or processing fails
    """
    try:
        client = arxiv.Client()
        search = arxiv.Search(id_list=[paper_id])
        paper = next(client.results(search))

        try:
            full_text = ArxivPaperRetriever.download_and_extract_text(paper.pdf_url)
        except ValueError as e:
            full_text = f"Error extracting full text: {str(e)}"

        return {
            'metadata': {
                'Title': paper.title,
                'Authors': [str(author) for author in paper.authors],
                'Published': str(paper.published),
                'Updated': str(paper.updated) if paper.updated else None,
                'Primary_category': paper.primary_category,
                'Categories': paper.categories,
            },
            'abstract': paper.summary,
            'full_text': full_text
        }

    except StopIteration:
        raise ValueError(f'Paper with ID {paper_id} not found')
    except Exception as e:
        raise ValueError(f'Failed to retrieve paper: {str(e)}')
