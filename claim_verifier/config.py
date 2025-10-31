"""Configuration settings for claim verifier"""

import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Configuration class for claim verifier"""
    
    # Google Custom Search API for fact-checking
    GOOGLE_FACT_CHECK_API_KEY = os.getenv('GOOGLE_FACT_CHECK_API_KEY') or os.getenv('GOOGLE_API_KEY')
    GOOGLE_FACT_CHECK_CX = os.getenv('GOOGLE_FACT_CHECK_CX')
    
    # Gemini API for analysis
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
    GEMINI_MODEL = "gemini-2.5-flash"
    
    # Fact-checking settings
    MAX_SEARCH_RESULTS = 10
    RELEVANCE_THRESHOLD = 0.05
    MAX_ALTERNATIVE_QUERIES = 2

# Global config instance
config = Config()