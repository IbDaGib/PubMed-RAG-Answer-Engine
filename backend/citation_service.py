from typing import List, Dict, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def format_citation(article_metadata: Dict) -> Optional[str]:
    """Formats a basic citation for a single article (APA-like)."""
    try:
        authors = article_metadata.get('authors', "N.A.")
        year = article_metadata.get('year', "n.d.")
        title = article_metadata.get('title', "No Title Provided")
        pmid = article_metadata.get('pmid', None)

        # Basic formatting (can be expanded)
        # Example: Doe J (2023). Article Title. PMID: 12345
        citation = f"{authors} ({year}). {title}."
        if pmid:
            citation += f" PMID: {pmid}."
        return citation

    except Exception as e:
        logger.error(f"Error formatting citation for article {article_metadata.get('pmid', '')}: {e}")
        return None

def generate_citations(context_abstracts: List[Dict]) -> List[str]:
    """Generates a list of formatted citations from the context abstracts."""
    citations = []
    if not context_abstracts:
        return citations

    for article in context_abstracts:
        metadata = article.get('metadata')
        if metadata:
            formatted = format_citation(metadata)
            if formatted:
                citations.append(formatted)
        else:
            logger.warning("Article missing metadata, cannot generate citation.")

    return citations

# Example usage (can be removed later)
if __name__ == '__main__':
    print("--- Testing Citation Generation ---")
    test_context = [
        {
            'metadata': {
                'pmid': '12345',
                'title': 'Study on Drug X Efficacy',
                'authors': 'Doe J, Smith A',
                'year': '2023'
            },
            'document': '...' # Document not needed for citation
        },
        {
            'metadata': {
                'pmid': '67890',
                'title': 'Side Effects of Drug X'
                # Missing authors/year
            },
            'document': '...'
        }
    ]
    generated_citations = generate_citations(test_context)
    print("Generated Citations:")
    for cit in generated_citations:
        print(f"- {cit}") 