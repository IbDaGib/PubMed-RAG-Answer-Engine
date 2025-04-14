import os
import google.generativeai as genai
from dotenv import load_dotenv
from typing import List, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Initialize Google Gemini client
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    logger.warning("GOOGLE_API_KEY not found in environment variables. LLM functionality will be disabled.")
    genai_configured = False
else:
    try:
        genai.configure(api_key=api_key)
        genai_configured = True
        logger.info("Google Generative AI client configured successfully.")
    except Exception as e:
        logger.error(f"Error configuring Google Generative AI: {e}")
        genai_configured = False

# Configuration
# Select a Gemini model - gemini-1.5-flash is fast and capable
DEFAULT_MODEL = "gemini-1.5-flash"
# Gemini API uses different configuration parameters
GENERATION_CONFIG = {
    "temperature": 0.7,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 512, # Adjust as needed
}
SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

def generate_answer(query: str, context_abstracts: List[Dict]) -> str:
    """Generates an answer using the Google Gemini model based on the query and context."""
    if not genai_configured:
        return "Google Generative AI client not configured. Please set GOOGLE_API_KEY."
    if not context_abstracts:
        return "No relevant context found to generate an answer."

    # --- Construct the prompt --- #
    context_str = "\n\n".join([
        f"Source PMID: {a['metadata'].get('pmid', 'N/A')}\nTitle: {a['metadata'].get('title', 'N/A')}\nAbstract: {a['document']}"
        for a in context_abstracts
    ])

    # Updated prompt for Gemini
    prompt = f"""
    You are a helpful assistant specialized in answering questions based *only* on the provided context from PubMed abstracts. Do not use any external knowledge. If the context does not contain the information needed to answer the query, clearly state that the answer cannot be found in the provided context.

    User Query: {query}

    Provided Context:
    --- START CONTEXT ---
    {context_str}
    --- END CONTEXT ---

    Based *only* on the provided context, answer the user query.
    Answer:
    """

    # --- Call the LLM --- #
    try:
        model = genai.GenerativeModel(
            model_name=DEFAULT_MODEL,
            generation_config=GENERATION_CONFIG,
            safety_settings=SAFETY_SETTINGS
        )
        logger.info(f"Sending request to Google Gemini model: {DEFAULT_MODEL}")
        response = model.generate_content(prompt)
        logger.info("Received response from Google Gemini.")

        # Extract the answer
        # Handle potential lack of response or blocked content
        if response.parts:
            answer = response.text # Accessing .text directly is often sufficient
        else:
            # Log the finish reason if available
            try:
                finish_reason = response.candidates[0].finish_reason if response.candidates else 'UNKNOWN'
                safety_ratings = response.candidates[0].safety_ratings if response.candidates else 'UNKNOWN'
                logger.warning(f"Gemini response blocked or empty. Finish Reason: {finish_reason}, Safety Ratings: {safety_ratings}")
            except Exception:
                 logger.warning(f"Gemini response blocked or empty. Could not retrieve details.")
            answer = "The model could not generate an answer based on the provided context, possibly due to safety filters or lack of relevant information."

        return answer.strip()

    except Exception as e:
        logger.error(f"Error calling Google Generative AI API: {e}")
        return f"An error occurred while generating the answer: {e}"

# Example usage (can be removed later)
if __name__ == '__main__':
    # Make sure you have a .env file with GOOGLE_API_KEY="your_key"
    if not genai_configured:
        print("Please create a .env file with your GOOGLE_API_KEY and ensure it's configured correctly.")
    else:
        print("--- Testing Gemini Answer Generation ---")
        test_query = "What are the main findings regarding drug X?"
        test_context = [
            {
                'metadata': {'pmid': '111', 'title': 'Study on Drug X Efficacy'},
                'document': 'This abstract discusses the positive results of drug X in phase 3 trials. It significantly reduced symptoms.'
            },
            {
                'metadata': {'pmid': '222', 'title': 'Side Effects of Drug X'},
                'document': 'Common side effects include nausea and dizziness, as reported in the study.'
            }
        ]
        generated_answer = generate_answer(test_query, test_context)
        print(f"Query: {test_query}")
        print(f"Generated Answer:\n{generated_answer}") 