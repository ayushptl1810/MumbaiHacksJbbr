import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Configuration class for the Visual Verification Service"""
    
    # API Configuration
    SERP_API_KEY: Optional[str] = os.getenv("SERP_API_KEY")
    SERPAPI_BASE_URL: str = "https://serpapi.com/search"
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    GEMINI_TEMPERATURE: float = float(os.getenv("GEMINI_TEMPERATURE", "0.1"))
    GEMINI_TOP_P: float = float(os.getenv("GEMINI_TOP_P", "0.8"))
    GEMINI_MAX_TOKENS: int = int(os.getenv("GEMINI_MAX_TOKENS", "1000000"))

    # Google API Configuration (replaces deprecated Fact Check Tools API)
    GOOGLE_API_KEY: Optional[str] = os.getenv("GOOGLE_API_KEY")
    GOOGLE_FACT_CHECK_CX: Optional[str] = os.getenv("GOOGLE_FACT_CHECK_CX")
    
    # Low-priority (social/UGC) domains to downrank (override via LOW_PRIORITY_DOMAINS)
    LOW_PRIORITY_DOMAINS: set = set((os.getenv(
        "LOW_PRIORITY_DOMAINS",
        ",".join([
            "twitter.com","www.twitter.com","x.com","www.x.com",
            "reddit.com","www.reddit.com",
            "facebook.com","www.facebook.com","m.facebook.com",
            "instagram.com","www.instagram.com",
            "tiktok.com","www.tiktok.com",
            "threads.net","www.threads.net"
        ])
    ) or "").split(","))
    # Analysis thresholds (kept configurable to avoid hardcoding)
    CONTEXT_SIM_THRESHOLD: float = float(os.getenv("CONTEXT_SIM_THRESHOLD", "0.6"))

    # Streaming downloader (yt-dlp) integration
    # If true, prefer yt-dlp for any video_url (works for YouTube/Instagram/Twitter/etc.)
    USE_STREAM_DOWNLOADER: bool = os.getenv("USE_STREAM_DOWNLOADER", "true").lower() == "true"
    # Binary path for yt-dlp (auto-resolved in code if not absolute)
    YTDLP_BIN: str = os.getenv("YTDLP_BIN", "yt-dlp")
    STREAM_DOWNLOAD_TIMEOUT: int = int(os.getenv("STREAM_DOWNLOAD_TIMEOUT", "30"))
    # Optional comma-separated list of domains to always treat as streaming
    STREAMING_DOMAINS: str = os.getenv("STREAMING_DOMAINS", "youtube.com,youtu.be,instagram.com,twitter.com,x.com,tiktok.com,facebook.com,fb.watch")
    
    # Cloudinary
    CLOUDINARY_CLOUD_NAME: Optional[str] = os.getenv("CLOUDINARY_CLOUD_NAME")
    CLOUDINARY_API_KEY: Optional[str] = os.getenv("CLOUDINARY_API_KEY")
    CLOUDINARY_API_SECRET: Optional[str] = os.getenv("CLOUDINARY_API_SECRET")
    CLOUDINARY_UPLOAD_PRESET: Optional[str] = os.getenv("CLOUDINARY_UPLOAD_PRESET")
    CLOUDINARY_FOLDER: str = os.getenv("CLOUDINARY_FOLDER", "frames")
    
    # Service Configuration
    SERVICE_HOST: str = os.getenv("SERVICE_HOST", "0.0.0.0")
    SERVICE_PORT: int = int(os.getenv("SERVICE_PORT", "7860"))
    
    # File Processing Configuration
    MAX_FILE_SIZE: int = int(os.getenv("MAX_FILE_SIZE", "50")) * 1024 * 1024  # 50MB default
    ALLOWED_IMAGE_EXTENSIONS: set = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    ALLOWED_VIDEO_EXTENSIONS: set = {'.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm', '.mkv', '.m4v'}
    
    # Video Processing Configuration
    FRAME_EXTRACTION_INTERVAL: int = int(os.getenv("FRAME_INTERVAL", "4"))  # seconds
    MAX_FRAMES_TO_ANALYZE: int = int(os.getenv("MAX_FRAMES", "10"))
    CLIP_DURATION: int = int(os.getenv("CLIP_DURATION", "5"))  # seconds
    
    # Image Processing Configuration
    COUNTER_MEASURE_WIDTH: int = int(os.getenv("IMAGE_WIDTH", "400"))
    COUNTER_MEASURE_HEIGHT: int = int(os.getenv("IMAGE_HEIGHT", "300"))
    
    # Temporary Storage Configuration
    TEMP_DIR: str = os.getenv("TEMP_DIR", "/tmp")
    CLEANUP_INTERVAL: int = int(os.getenv("CLEANUP_INTERVAL", "3600"))  # seconds
    
    # Logging Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Debug Configuration
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # Upstash Redis Configuration
    UPSTASH_REDIS_URL: Optional[str] = os.getenv("UPSTASH_REDIS_URL")
    UPSTASH_REDIS_TOKEN: Optional[str] = os.getenv("UPSTASH_REDIS_TOKEN")
    REDIS_TTL: int = int(os.getenv("REDIS_TTL", "86400"))  # 24 hours in seconds
    
    # Razorpay Configuration
    RAZORPAY_ID: Optional[str] = os.getenv("RAZORPAY_ID")
    RAZORPAY_KEY: Optional[str] = os.getenv("RAZORPAY_KEY")
    RAZORPAY_WEBHOOK_SECRET: Optional[str] = os.getenv("RAZORPAY_WEBHOOK_SECRET")
    
    @classmethod
    def validate(cls) -> bool:
        """Validate configuration values"""
        if not cls.SERP_API_KEY:
            print("Warning: SERP_API_KEY not set. Service will not function without it.")
            return False
        
        if not cls.GOOGLE_API_KEY:
            print("Warning: GOOGLE_API_KEY not set. Text fact-checking will not function without it.")
            return False
        
        if not cls.GOOGLE_FACT_CHECK_CX:
            print("Warning: GOOGLE_FACT_CHECK_CX not set. Text fact-checking will not function without it.")
            return False
        
        if cls.MAX_FILE_SIZE <= 0:
            print("Error: MAX_FILE_SIZE must be positive")
            return False
        
        if cls.FRAME_EXTRACTION_INTERVAL <= 0:
            print("Error: FRAME_EXTRACTION_INTERVAL must be positive")
            return False
        
        if cls.CLIP_DURATION <= 0:
            print("Error: CLIP_DURATION must be positive")
            return False
        
        return True
    
    @classmethod
    def get_allowed_extensions(cls) -> set:
        """Get all allowed file extensions"""
        return cls.ALLOWED_IMAGE_EXTENSIONS.union(cls.ALLOWED_VIDEO_EXTENSIONS)
    
    @classmethod
    def is_image_file(cls, filename: str) -> bool:
        """Check if file is a valid image"""
        from pathlib import Path
        return Path(filename).suffix.lower() in cls.ALLOWED_IMAGE_EXTENSIONS
    
    @classmethod
    def is_video_file(cls, filename: str) -> bool:
        """Check if file is a valid video"""
        from pathlib import Path
        return Path(filename).suffix.lower() in cls.ALLOWED_VIDEO_EXTENSIONS

# Global configuration instance
config = Config()
