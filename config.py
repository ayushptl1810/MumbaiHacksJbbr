"""
Configuration module for loading environment variables
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class config:
    """Configuration class for accessing environment variables"""
    
    # API Keys
    SERP_API_KEY = os.getenv('SERP_API_KEY')
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    NEWS_API_KEY = os.getenv('NEWS_API_KEY')
    GNEWS_API_KEY = os.getenv('GNEWS_API_KEY')
    
    # Google Custom Search
    GOOGLE_FACT_CHECK_API_KEY = os.getenv('GOOGLE_FACT_CHECK_API_KEY')
    GOOGLE_FACT_CHECK_CX = os.getenv('GOOGLE_FACT_CHECK_CX')
    
    # Pinecone
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
    PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')
    
    # Reddit
    REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID')
    REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET')
    REDDIT_USER_AGENT = os.getenv('REDDIT_USER_AGENT')
    
    # MongoDB
    MONGO_PASS = os.getenv('MONGO_PASS')
    MONGO_CONNECTION_STRING = os.getenv('MONGO_CONNECTION_STRING')
    
    # Telegram
    TELEGRAM_API_ID = os.getenv('TELEGRAM_API_ID')
    TELEGRAM_API_HASH = os.getenv('TELEGRAM_API_HASH')
    TELEGRAM_PHONE = os.getenv('TELEGRAM_PHONE')
    TELEGRAM_SESSION_NAME = os.getenv('TELEGRAM_SESSION_NAME')
    
    # Twitter/X
    TWITTER_USERNAME = os.getenv('TWITTER_USERNAME')
    TWITTER_EMAIL = os.getenv('TWITTER_EMAIL')
    TWITTER_PASSWORD = os.getenv('TWITTER_PASSWORD')
    TWITTER_COOKIES_FILE = os.getenv('TWITTER_COOKIES_FILE')
    TWITTER_AUTO_DISCOVER_KEYWORDS = os.getenv('TWITTER_AUTO_DISCOVER_KEYWORDS', 'False').lower() == 'true'
    
    # Cloudinary (optional)
    CLOUDINARY_CLOUD_NAME = os.getenv('CLOUDINARY_CLOUD_NAME')
    CLOUDINARY_API_KEY = os.getenv('CLOUDINARY_API_KEY')
    CLOUDINARY_API_SECRET = os.getenv('CLOUDINARY_API_SECRET')
