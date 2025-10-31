"""trend_scanner package init

This package now includes Google Agents SDK integration for enhanced misinformation detection.
This file intentionally avoids importing submodules at package import time so
that tooling and lightweight checks won't require heavy external dependencies.
Use explicit imports like `from trend_scanner.scraper import WebContentScraper`.
"""

__all__ = [
    'models', 'scraper', 'tools', 'google_agents'
]

__version__ = "2.0.0"

