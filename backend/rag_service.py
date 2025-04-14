import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
# Using a smaller, faster model for initial development.
# For better biomedical domain performance, consider:
# 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
# or sentence-transformer specific biomedical models if available.
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
COLLECTION_NAME = "pubmed_abstracts"
CHROMA_PATH = "chroma_db" # Directory to persist ChromaDB data
N_RESULTS_RETRIEVAL = 5 # Number of relevant documents to retrieve

# --- Initialization ---
try:
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    logger.info(f"Successfully loaded sentence transformer model: {EMBEDDING_MODEL_NAME}")
except Exception as e:
    logger.error(f"Error loading sentence transformer model {EMBEDDING_MODEL_NAME}: {e}")
    embedding_model = None

# Initialize ChromaDB client (persistent)
# This will create the 'chroma_db' directory if it doesn't exist
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
logger.info(f"Initialized ChromaDB client with persistence path: {CHROMA_PATH}")

# --- ChromaDB Collection Management ---
def get_or_create_collection(name: str = COLLECTION_NAME):
    """Gets or creates a ChromaDB collection."""
    try:
        collection = chroma_client.get_or_create_collection(
            name=name,
            # Optional: Specify metadata for embedding function if not using default
            # metadata={"hnsw:space": "cosine"} # Example: use cosine distance
        )
        logger.info(f"Successfully got or created collection: {name}")
        return collection
    except Exception as e:
        logger.error(f"Error getting or creating collection {name}: {e}")
        return None

pubmed_collection = get_or_create_collection()

# --- Core RAG Functions ---
def add_documents_to_collection(articles: List[Dict[str, Optional[str]]]):
    """Generates embeddings and adds documents to the ChromaDB collection."""
    if not pubmed_collection or not embedding_model:
        logger.error("Collection or embedding model not initialized. Cannot add documents.")
        return
    if not articles:
        logger.warning("No articles provided to add to the collection.")
        return

    ids = []
    documents = [] # Text content to be embedded (abstracts)
    metadatas = [] # Associated metadata

    for article in articles:
        # Ensure required fields are present
        pmid = article.get('pmid')
        abstract = article.get('abstract')
        title = article.get('title')
        authors = article.get('authors')
        year = article.get('year')

        if pmid and abstract:
            ids.append(str(pmid)) # Chroma requires string IDs
            documents.append(abstract)
            # Store other info as metadata
            metadatas.append({
                "title": title or "",
                "authors": authors or "",
                "year": year or "",
                "pmid": str(pmid) # Also store pmid in metadata for easy access
            })
        else:
            logger.warning(f"Skipping article due to missing PMID or abstract: {article.get('title', 'N/A')}")

    if not ids:
        logger.warning("No valid documents found to add after filtering.")
        return

    try:
        # Generate embeddings in batches (model handles batching internally)
        logger.info(f"Generating embeddings for {len(documents)} documents...")
        embeddings = embedding_model.encode(documents, show_progress_bar=True).tolist()
        logger.info("Embeddings generated.")

        # Add to collection (upsert=True updates if ID exists)
        pubmed_collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents, # Store the original abstract text
            metadatas=metadatas
        )
        logger.info(f"Successfully added/updated {len(ids)} documents in collection '{COLLECTION_NAME}'.")

    except Exception as e:
        logger.error(f"Error adding documents to collection '{COLLECTION_NAME}': {e}")

def query_collection(query: str, n_results: int = N_RESULTS_RETRIEVAL) -> List[Dict]:
    """Queries the collection for relevant documents based on the query text."""
    if not pubmed_collection or not embedding_model:
        logger.error("Collection or embedding model not initialized. Cannot query.")
        return []
    if not query:
        logger.warning("Empty query received.")
        return []

    try:
        # Generate embedding for the query
        query_embedding = embedding_model.encode([query]).tolist()

        # Query the collection
        results = pubmed_collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            include=["metadatas", "documents", "distances"] # Include metadata, text, and distance score
        )
        logger.info(f"Retrieved {len(results.get('ids', [[]])[0])} results for query: '{query}'")

        # Format results
        formatted_results = []
        if results and results.get('ids') and results['ids'][0]:
            for i, doc_id in enumerate(results['ids'][0]):
                formatted_results.append({
                    "id": doc_id,
                    "metadata": results['metadatas'][0][i] if results.get('metadatas') else None,
                    "document": results['documents'][0][i] if results.get('documents') else None,
                    "distance": results['distances'][0][i] if results.get('distances') else None,
                })
        return formatted_results

    except Exception as e:
        logger.error(f"Error querying collection '{COLLECTION_NAME}': {e}")
        return []

# Example usage (can be removed later)
if __name__ == '__main__':
    # Simulate adding some articles (replace with actual fetched data)
    sample_articles = [
        {'pmid': '123', 'title': 'Article 1', 'abstract': 'This is the first abstract about science.', 'authors': 'Doe J', 'year': '2023'},
        {'pmid': '456', 'title': 'Article 2', 'abstract': 'A second paper discussing technology.', 'authors': 'Smith A', 'year': '2024'},
        {'pmid': '789', 'title': 'Article 3', 'abstract': 'Science and technology combined in this third abstract.', 'authors': 'Lee B', 'year': '2023'},
    ]
    logger.info("--- Testing add_documents_to_collection ---")
    add_documents_to_collection(sample_articles)
    logger.info(f"Collection count: {pubmed_collection.count()}")

    logger.info("\n--- Testing query_collection ---")
    test_query = "Tell me about science and tech"
    retrieved_docs = query_collection(test_query)
    print(f"Query: {test_query}")
    print("Retrieved documents:")
    for doc in retrieved_docs:
        print(f"  ID: {doc['id']}, Distance: {doc['distance']:.4f}, Title: {doc['metadata'].get('title')}")
        print(f"    Abstract: {doc['document'][:100]}...") 