"""Configuration for Explanation Agent"""

import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Configuration settings for Explanation Agent"""
    
    # Gemini AI configuration
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')
    
    # Output settings
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'aegis_feed_posts')
    
    # Post formatting settings
    MAX_HEADING_LENGTH = 100
    MAX_BODY_LENGTH = 2000
    CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence to publish

config = Config()