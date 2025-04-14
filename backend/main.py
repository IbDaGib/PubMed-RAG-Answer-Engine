from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

# Import functions from the pubmed_service module
from pubmed_service import search_pubmed, fetch_abstracts, MAX_RESULTS
# Import functions from the rag_service module
from rag_service import add_documents_to_collection, query_collection
# Import LLM and Citation services
from llm_service import generate_answer
from citation_service import generate_citations

app = FastAPI()

# --- Pydantic Models ---

class SearchQuery(BaseModel):
    query: str
    max_results: Optional[int] = MAX_RESULTS # Allow overriding default max results

class Article(BaseModel):
    pmid: str
    title: str
    abstract: str
    authors: str
    year: Optional[str] = None

class SearchResponse(BaseModel):
    articles: List[Article]
    count: int

# New response model for the /answer endpoint
class AnswerResponse(BaseModel):
    answer: str
    citations: List[str]
    retrieved_articles: List[Article] # Include retrieved articles for context/display

# --- API Endpoints ---

@app.get("/")
def read_root():
    return {"message": "PubMed RAG Backend is running"}

@app.post("/search", response_model=SearchResponse)
async def search_articles(search_query: SearchQuery):
    """
    Receives a search query, finds relevant PubMed article IDs,
    fetches their details, and returns them.
    """
    print(f"Received search query: {search_query.query}, Max results: {search_query.max_results}")
    try:
        # 1. Search PubMed for PMIDs
        pmids = search_pubmed(search_query.query, search_query.max_results)
        if not pmids:
            return SearchResponse(articles=[], count=0)

        # 2. Fetch details for the found PMIDs
        articles_data = fetch_abstracts(pmids)

        # --- Add fetched articles to ChromaDB ---
        if articles_data:
            # Run in background? For now, do it synchronously
            add_documents_to_collection(articles_data)
        # ----------------------------------------

        # 3. Validate and structure the response
        validated_articles = [Article(**article) for article in articles_data]

        return SearchResponse(articles=validated_articles, count=len(validated_articles))

    except Exception as e:
        # Log the exception for debugging
        print(f"Error during search endpoint processing: {e}")
        # Return a generic error response
        raise HTTPException(status_code=500, detail="An error occurred while processing the search request.")

@app.post("/answer", response_model=AnswerResponse)
async def get_answer(search_query: SearchQuery):
    """
    Receives a query, retrieves relevant abstracts from ChromaDB,
    generates an answer using an LLM, formats citations, and returns the response.
    """
    print(f"Received answer query: {search_query.query}")
    try:
        # 1. Retrieve relevant documents from ChromaDB
        # We use the query from the request payload
        retrieved_docs = query_collection(search_query.query)

        if not retrieved_docs:
            # Optional: Could implement fallback to PubMed search here
            # e.g., pmids = search_pubmed(...); articles_data = fetch_abstracts(...)
            # add_documents_to_collection(articles_data)
            # retrieved_docs = query_collection(...) # Retry query
            # If still no docs, return specific message
            return AnswerResponse(answer="No relevant documents found in the database to answer the query.", citations=[], retrieved_articles=[])

        # 2. Prepare context and call LLM
        # The 'retrieved_docs' already contain metadata and documents needed
        llm_answer = generate_answer(search_query.query, retrieved_docs)

        # 3. Generate Citations from retrieved documents
        citations = generate_citations(retrieved_docs)

        # 4. Format retrieved articles for the response payload
        # Map ChromaDB result structure to our Pydantic Article model
        response_articles = []
        for doc in retrieved_docs:
            meta = doc.get('metadata', {})
            response_articles.append(Article(
                pmid=meta.get('pmid', 'N/A'), # Get pmid from metadata
                title=meta.get('title', 'N/A'),
                abstract=doc.get('document', ''), # Get abstract from document field
                authors=meta.get('authors', 'N/A'),
                year=meta.get('year', None)
            ))

        return AnswerResponse(
            answer=llm_answer,
            citations=citations,
            retrieved_articles=response_articles
        )

    except Exception as e:
        logger.error(f"Error during answer endpoint processing: {e}") # Use logger here too
        # Log the exception for debugging
        # print(f"Error during answer endpoint processing: {e}") # Original print
        # Return a generic error response
        raise HTTPException(status_code=500, detail="An error occurred while processing the answer request.")

# Add logger import
import logging
logger = logging.getLogger(__name__)

# --- Optional: Add CORS middleware if frontend is on a different origin ---
# from fastapi.middleware.cors import CORSMiddleware
# origins = [
#     "http://localhost:3000",  # Assuming frontend runs on port 3000
#     # Add other allowed origins if necessary
# ]
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# Placeholder for RAG/LLM logic (to be added later)
# @app.post("/answer")
# async def get_answer(search_query: SearchQuery):
#     # 1. Search PubMed (call search_articles logic or reuse parts)
#     # 2. Embed query and abstracts
#     # 3. Retrieve relevant abstracts from vector store
#     # 4. Call LLM with query and context
#     # 5. Format response with answer and citations
#     pass 