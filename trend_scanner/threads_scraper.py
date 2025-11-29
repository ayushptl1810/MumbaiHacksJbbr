"""Threads scraper using Playwright for browser automation"""

import json
import logging
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

# Try to import required libraries
try:
    from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
    from parsel import Selector
    from nested_lookup import nested_lookup
    import jmespath
    PLAYWRIGHT_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Playwright or related libraries not available: {e}")
    logger.warning("Install with: pip install playwright parsel nested-lookup jmespath")
    PLAYWRIGHT_AVAILABLE = False


def parse_thread(data: Dict) -> Dict:
    """Parse Threads post JSON dataset for the most important fields"""
    try:
        result = jmespath.search(
            """{
                text: post.caption.text,
                published_on: post.taken_at,
                id: post.id,
                pk: post.pk,
                code: post.code,
                username: post.user.username,
                user_pic: post.user.profile_pic_url,
                user_verified: post.user.is_verified,
                user_pk: post.user.pk,
                user_id: post.user.id,
                has_audio: post.has_audio,
                reply_count: view_replies_cta_string,
                like_count: post.like_count,
                images: post.carousel_media[].image_versions2.candidates[1].url,
                image_count: post.carousel_media_count,
                videos: post.video_versions[].url
            }""",
            data,
        )
        
        # Clean up the result
        if result:
            result["videos"] = list(set(result.get("videos") or []))
            
            # Parse reply count if it's a string
            if result.get("reply_count") and type(result["reply_count"]) != int:
                try:
                    result["reply_count"] = int(result["reply_count"].split(" ")[0])
                except:
                    result["reply_count"] = 0
            
            # Construct URL
            result["url"] = f"https://www.threads.net/@{result.get('username', 'unknown')}/post/{result.get('code', '')}"
            
        return result or {}
        
    except Exception as e:
        logger.error(f"Failed to parse thread data: {e}")
        return {}


def parse_profile(data: Dict) -> Dict:
    """Parse Threads profile JSON dataset for the most important fields"""
    try:
        result = jmespath.search(
            """{
                is_private: text_post_app_is_private,
                is_verified: is_verified,
                profile_pic: hd_profile_pic_versions[-1].url,
                username: username,
                full_name: full_name,
                bio: biography,
                bio_links: bio_links[].url,
                followers: follower_count
            }""",
            data,
        )
        
        if result:
            result["url"] = f"https://www.threads.net/@{result.get('username', '')}"
        
        return result or {}
        
    except Exception as e:
        logger.error(f"Failed to parse profile data: {e}")
        return {}


class ThreadsScraper:
    """Scraper for Threads posts using Playwright"""
    
    def __init__(self, headless: bool = True, cache_enabled: bool = True):
        """
        Initialize Threads scraper
        
        Args:
            headless: Run browser in headless mode
            cache_enabled: Enable caching of scraped content
        """
        if not PLAYWRIGHT_AVAILABLE:
            raise ImportError(
                "Playwright is not installed. Install with: "
                "pip install playwright parsel nested-lookup jmespath && playwright install chromium"
            )
        
        self.headless = headless
        self.cache_enabled = cache_enabled
        self._cache = {}
        
        logger.info(f"ThreadsScraper initialized (headless={headless}, cache={cache_enabled})")
    
    def _get_cache_key(self, url: str) -> str:
        """Generate cache key for URL"""
        return hashlib.md5(url.encode()).hexdigest()
    
    def scrape_thread(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Scrape a single Threads post and its replies
        
        Args:
            url: Full URL to the Threads post
            
        Returns:
            Dict with 'thread' (main post) and 'replies' (list of reply posts)
        """
        # Check cache
        cache_key = None
        if self.cache_enabled:
            cache_key = self._get_cache_key(url)
            if cache_key in self._cache:
                logger.info(f"Returning cached result for {url}")
                return self._cache[cache_key]
        
        try:
            with sync_playwright() as pw:
                # Launch browser
                browser = pw.chromium.launch(headless=self.headless)
                context = browser.new_context(
                    viewport={"width": 1920, "height": 1080},
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                )
                page = context.new_page()
                
                # Navigate to URL
                logger.info(f"Scraping Threads post: {url}")
                page.goto(url, wait_until="domcontentloaded")
                
                # Wait for content to load
                try:
                    page.wait_for_selector("[data-pressable-container=true]", timeout=10000)
                except PlaywrightTimeout:
                    logger.warning(f"Timeout waiting for content on {url}")
                
                # Get page content
                html_content = page.content()
                browser.close()
                
                # Parse with Parsel
                selector = Selector(html_content)
                hidden_datasets = selector.css('script[type="application/json"][data-sjs]::text').getall()
                
                # Find thread data
                for hidden_dataset in hidden_datasets:
                    # Skip datasets that don't contain thread data
                    if '"ScheduledServerJS"' not in hidden_dataset:
                        continue
                    if "thread_items" not in hidden_dataset:
                        continue
                    
                    try:
                        data = json.loads(hidden_dataset)
                        
                        # Find thread_items using nested_lookup
                        thread_items = nested_lookup("thread_items", data)
                        
                        if not thread_items:
                            continue
                        
                        # Parse threads
                        threads = [parse_thread(t) for thread in thread_items for t in thread]
                        
                        result = {
                            "thread": threads[0] if threads else {},
                            "replies": threads[1:] if len(threads) > 1 else [],
                        }
                        
                        # Cache result
                        if self.cache_enabled and cache_key:
                            self._cache[cache_key] = result
                        
                        logger.info(f"Successfully scraped thread with {len(result.get('replies', []))} replies")
                        return result
                        
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to decode JSON: {e}")
                        continue
                    except Exception as e:
                        logger.error(f"Error parsing thread data: {e}")
                        continue
                
                logger.warning(f"Could not find thread data in page: {url}")
                return None
                
        except Exception as e:
            logger.error(f"Error scraping thread {url}: {e}")
            return None
    
    def _scrape_with_playwright(self, url: str) -> Optional[str]:
        """
        Helper method to scrape using Playwright (runs in separate thread if needed)
        
        Args:
            url: URL to scrape
            
        Returns:
            HTML content or None
        """
        try:
            with sync_playwright() as pw:
                browser = pw.chromium.launch(headless=self.headless)
                context = browser.new_context(
                    viewport={"width": 1920, "height": 1080},
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                )
                page = context.new_page()
                
                logger.info(f"Scraping Threads profile: {url}")
                page.goto(url, wait_until="domcontentloaded")
                
                try:
                    page.wait_for_selector("[data-pressable-container=true]", timeout=10000)
                except PlaywrightTimeout:
                    logger.warning(f"Timeout waiting for profile content on {url}")
                
                html_content = page.content()
                browser.close()
                return html_content
        except Exception as e:
            logger.error(f"Playwright scraping error for {url}: {e}")
            return None
    
    def scrape_profile(self, url: str, include_threads: bool = True) -> Optional[Dict[str, Any]]:
        """
        Scrape a Threads profile and optionally their recent posts
        
        Args:
            url: Full URL to the Threads profile (e.g., https://www.threads.net/@username)
            include_threads: Whether to include recent threads from the profile
            
        Returns:
            Dict with 'user' (profile info) and 'threads' (list of recent posts)
        """
        # Check cache
        cache_key = None
        if self.cache_enabled:
            cache_key = self._get_cache_key(url + str(include_threads))
            if cache_key in self._cache:
                logger.info(f"Returning cached profile for {url}")
                return self._cache[cache_key]
        
        try:
            # Use a separate thread to avoid asyncio subprocess issues on Windows
            import asyncio
            loop = None
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                pass
            
            if loop:
                # Running in async context - use thread pool
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(self._scrape_with_playwright, url)
                    html_content = future.result(timeout=30)
            else:
                # Not in async context - use directly
                html_content = self._scrape_with_playwright(url)
            
            if not html_content:
                return None
            
            # Parse with Parsel
            selector = Selector(html_content)
            hidden_datasets = selector.css('script[type="application/json"][data-sjs]::text').getall()
            
            parsed = {
                "user": {},
                "threads": [],
            }
            
            # Find profile and thread data
            for hidden_dataset in hidden_datasets:
                if '"ScheduledServerJS"' not in hidden_dataset:
                    continue
                
                is_profile = 'follower_count' in hidden_dataset
                is_threads = 'thread_items' in hidden_dataset
                
                if not is_profile and not is_threads:
                        continue
                    
                    try:
                        data = json.loads(hidden_dataset)
                        
                        if is_profile:
                            user_data = nested_lookup('user', data)
                            if user_data:
                                parsed['user'] = parse_profile(user_data[0])
                        
                        if is_threads and include_threads:
                            thread_items = nested_lookup('thread_items', data)
                            if thread_items:
                                threads = [parse_thread(t) for thread in thread_items for t in thread]
                                parsed['threads'].extend(threads)
                        
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to decode JSON in profile: {e}")
                        continue
                    except Exception as e:
                        logger.error(f"Error parsing profile data: {e}")
                        continue
                
                # Cache result
                if self.cache_enabled and cache_key and (parsed['user'] or parsed['threads']):
                    self._cache[cache_key] = parsed
                
                logger.info(f"Successfully scraped profile with {len(parsed.get('threads', []))} threads")
                return parsed
                
        except Exception as e:
            logger.error(f"Error scraping profile {url}: {e}")
            return None
    
    def clear_cache(self):
        """Clear the scraper cache"""
        self._cache.clear()
        logger.info("Cache cleared")


# Convenience functions
def scrape_thread(url: str) -> Optional[Dict[str, Any]]:
    """Convenience function to scrape a single thread"""
    scraper = ThreadsScraper()
    return scraper.scrape_thread(url)


def scrape_profile(url: str, include_threads: bool = True) -> Optional[Dict[str, Any]]:
    """Convenience function to scrape a profile"""
    scraper = ThreadsScraper()
    return scraper.scrape_profile(url, include_threads)
