import requests
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional
import os
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
PUBMED_API_KEY = os.getenv("PUBMED_API_KEY")

if not PUBMED_API_KEY:
    logger.warning("PUBMED_API_KEY not found in environment variables. Proceeding with anonymous access (lower rate limits).")

PUBMED_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
PUBMED_SEARCH_URL = PUBMED_BASE_URL + "esearch.fcgi"
PUBMED_FETCH_URL = PUBMED_BASE_URL + "efetch.fcgi"

MAX_RESULTS = 10 # Limit the number of abstracts to fetch initially

def search_pubmed(query: str, max_results: int = MAX_RESULTS) -> List[str]:
    """Searches PubMed for article IDs matching the query."""
    params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": str(max_results),
        "sort": "relevance" # Or 'pub+date'
    }
    # Add API key if available
    if PUBMED_API_KEY:
        params["api_key"] = PUBMED_API_KEY

    try:
        logger.info(f"Searching PubMed with query: '{query}', max_results: {max_results}")
        response = requests.get(PUBMED_SEARCH_URL, params=params)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        data = response.json()
        id_list = data.get("esearchresult", {}).get("idlist", [])
        logger.info(f"Found {len(id_list)} PubMed IDs for query: '{query}'")
        return id_list
    except requests.exceptions.RequestException as e:
        logger.error(f"Error searching PubMed: {e}")
        return []
    except Exception as e:
        logger.error(f"An unexpected error occurred during PubMed search: {e}")
        return []


def fetch_abstracts(pmids: List[str]) -> List[Dict[str, Optional[str]]]:
    """Fetches abstracts and metadata for a list of PubMed IDs (PMIDs)."""
    if not pmids:
        return []

    ids_str = ",".join(pmids)
    params = {
        "db": "pubmed",
        "id": ids_str,
        "retmode": "xml",
        "rettype": "abstract"
    }
    # Add API key if available
    if PUBMED_API_KEY:
        params["api_key"] = PUBMED_API_KEY

    try:
        logger.info(f"Fetching abstracts for {len(pmids)} PMIDs.")
        response = requests.get(PUBMED_FETCH_URL, params=params)
        response.raise_for_status()

        root = ET.fromstring(response.content)
        articles_data = []

        for article in root.findall('.//PubmedArticle'):
            pmid_element = article.find('.//PMID')
            pmid = pmid_element.text if pmid_element is not None else None

            article_title_element = article.find('.//ArticleTitle')
            title = article_title_element.text if article_title_element is not None else "No Title Available"

            abstract_text_element = article.find('.//AbstractText')
            # Handle potential structured abstracts (multiple AbstractText elements)
            abstract_parts = []
            abstract_elements = article.findall('.//Abstract/AbstractText')
            if abstract_elements:
                for elem in abstract_elements:
                    if elem.text:
                        label = elem.get('Label')
                        text = elem.text.strip()
                        if label:
                            abstract_parts.append(f"{label}: {text}")
                        else:
                            abstract_parts.append(text)
                abstract = "\n".join(abstract_parts)
            elif abstract_text_element is not None and abstract_text_element.text:
                 abstract = abstract_text_element.text.strip()
            else:
                abstract = "No Abstract Available"

            # Attempt to extract year from PubDate
            year = None
            pub_date = article.find('.//PubDate')
            if pub_date is not None:
                year_element = pub_date.find('.//Year')
                if year_element is not None:
                    year = year_element.text
                else: # Fallback for MedlineDate format e.g., <MedlineDate>2023 Dec</MedlineDate>
                    medline_date_element = pub_date.find('.//MedlineDate')
                    if medline_date_element is not None and medline_date_element.text:
                        # Try to extract the year (first 4 digits)
                        if len(medline_date_element.text) >= 4 and medline_date_element.text[:4].isdigit():
                             year = medline_date_element.text[:4]

            # Extract authors (simple concatenation for now)
            authors_list = []
            author_elements = article.findall('.//Author')
            for author in author_elements:
                last_name = author.find('.//LastName')
                initials = author.find('.//Initials')
                if last_name is not None and initials is not None and last_name.text and initials.text:
                    authors_list.append(f"{last_name.text.strip()} {initials.text.strip()}")
                else: # Handle CollectiveName e.g., <CollectiveName>Research Group</CollectiveName>
                    collective_name = author.find('.//CollectiveName')
                    if collective_name is not None and collective_name.text:
                         authors_list.append(collective_name.text.strip())

            authors_str = ", ".join(authors_list) if authors_list else "No Authors Listed"

            if pmid and title and abstract != "No Abstract Available": # Require abstract
                 articles_data.append({
                    "pmid": pmid,
                    "title": title.strip() if title else title,
                    "abstract": abstract,
                    "authors": authors_str,
                    "year": year,
                    # Add other relevant fields if needed, e.g., Journal Title, DOI
                })

        logger.info(f"Successfully fetched and parsed details for {len(articles_data)} articles.")
        return articles_data

    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching PubMed abstracts: {e}")
        return []
    except ET.ParseError as e:
        logger.error(f"Error parsing PubMed XML response: {e}")
        return []
    except Exception as e:
        logger.error(f"An unexpected error occurred during abstract fetching: {e}")
        return []

# Example usage (can be removed later)
if __name__ == '__main__':
    logger.info("--- Testing PubMed Service --- ")
    test_query = "diabetes management guidelines"
    found_ids = search_pubmed(test_query, max_results=5)
    if found_ids:
        abstract_details = fetch_abstracts(found_ids)
        for i, article in enumerate(abstract_details):
            print(f"--- Article {i+1} ---")
            print(f"PMID: {article['pmid']}")
            print(f"Title: {article['title']}")
            print(f"Year: {article['year']}")
            print(f"Authors: {article['authors']}")
            print(f"Abstract: {article['abstract'][:200]}...") # Print first 200 chars
            print("-" * 20) 