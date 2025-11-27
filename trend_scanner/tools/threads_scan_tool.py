"""Threads scanning tool for trend_scanner"""

import time
import json
import logging
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from pydantic import BaseModel, Field
from datetime import datetime
from ..scraper import WebContentScraper
from ..google_agents import GoogleAgentsManager
from ..models import BatchPostData, BatchRiskAssessment

logger = logging.getLogger(__name__)


# Tool base class
class GoogleTool:
    def __init__(self):
        self.name = getattr(self, 'name', self.__class__.__name__)
        self.description = getattr(self, 'description', 'A Google-powered tool')
    
    def _run(self, *args, **kwargs):
        raise NotImplementedError("Tool must implement _run method")
    
    def run(self, *args, **kwargs):
        return self._run(*args, **kwargs)


class ThreadsScanInput(BaseModel):
    profile_username: str = Field(description="Threads profile username to scan (without @)")
    limit: int = Field(default=10, description="Number of threads to scan")


class ThreadsScanOutput(BaseModel):
    trending_threads: List[Dict[str, Any]] = Field(description="List of trending Threads posts found")
    scan_summary: str = Field(description="Summary of the scan results")


class ThreadsScanTool(GoogleTool):
    name: str = "threads_scanner"
    description: str = "Scans Threads profiles for posts and ranks them by potential misinformation risk using Google Agents SDK"

    def __init__(self, llm_wrapper, velocity_threshold=10, min_like_threshold=100, google_api_key=None):
        super().__init__()
        object.__setattr__(self, '_llm_wrapper', llm_wrapper)
        object.__setattr__(self, '_velocity_threshold', velocity_threshold)
        object.__setattr__(self, '_min_like_threshold', min_like_threshold)
        object.__setattr__(self, '_tracked_threads', {})
        object.__setattr__(self, '_web_scraper', WebContentScraper())  # For scraping external links
        object.__setattr__(self, '_scraped_cache', {})
        
        # Try to import ThreadsScraper (for scraping Threads profiles)
        try:
            from ..threads_scraper import ThreadsScraper
            object.__setattr__(self, '_threads_scraper', ThreadsScraper(headless=True, cache_enabled=True))
            object.__setattr__(self, '_scraper_available', True)
            logger.info("ThreadsScraper initialized successfully")
        except ImportError as e:
            logger.warning(f"ThreadsScraper not available: {e}")
            logger.warning("Install with: pip install playwright parsel nested-lookup jmespath")
            object.__setattr__(self, '_threads_scraper', None)
            object.__setattr__(self, '_scraper_available', False)
        
        # Initialize Google Agents Manager
        try:
            object.__setattr__(self, '_google_agents', GoogleAgentsManager(api_key=google_api_key))
            logger.info("Google Agents SDK initialized for Threads scanner")
        except Exception as e:
            logger.warning(f"Failed to initialize Google Agents SDK: {e}")
            object.__setattr__(self, '_google_agents', None)

    def calculate_velocity(self, thread_id: str, current_likes: int, published_on: float) -> float:
        """Calculate engagement velocity for a thread"""
        current_time = time.time()
        if thread_id in self._tracked_threads:
            metric = self._tracked_threads[thread_id]
            time_diff = current_time - metric.current_time
            likes_diff = current_likes - metric.current_likes
            metric.current_likes = current_likes
            metric.current_time = current_time
            if time_diff > 0:
                velocity = (likes_diff / time_diff) * 3600
                metric.velocity = velocity
                return velocity
            return metric.velocity
        else:
            age_seconds = max(current_time - published_on, 1.0)
            hours = age_seconds / 3600.0
            proxy_velocity = current_likes / hours if hours > 0 else float(current_likes) * 3600.0
            self._tracked_threads[thread_id] = type('VM', (), {
                'initial_likes': current_likes,
                'current_likes': current_likes,
                'initial_time': current_time,
                'current_time': current_time,
                'velocity': proxy_velocity
            })()
            return proxy_velocity

    def extract_thread_content(self, thread: Dict[str, Any]) -> Tuple[str, Optional[str], str, int]:
        """
        Extract content from a Threads post, including scraping external links.
        Returns: (combined_content, scraped_content, content_source, scraped_count)
        """
        thread_content = thread.get('text', '')
        scraped_content = None
        content_source = "threads_text"
        scraped_count = 0
        
        # Check if there's a URL in the thread text (Threads doesn't have separate URL field like Reddit)
        # We'll look for URLs in the text content
        import re
        url_pattern = r'https?://[^\s]+'
        urls = re.findall(url_pattern, thread_content)
        
        if urls:
            # Try to scrape the first URL found
            url = urls[0]
            if self._web_scraper.is_scrapeable_url(url):
                url_hash = hashlib.md5(url.encode()).hexdigest()
                if url_hash in self._scraped_cache:
                    scraped_content = self._scraped_cache[url_hash]
                    content_source = "cached_scraped"
                else:
                    scraped_content, scrape_method = self._web_scraper.scrape_content(url)
                    if scraped_content:
                        self._scraped_cache[url_hash] = scraped_content
                        content_source = f"scraped_{scrape_method}"
                        scraped_count = 1
                    else:
                        content_source = "link_failed"
            else:
                content_source = "link_not_scrapeable"
        
        # Combine content if we have scraped data
        if scraped_content:
            combined_content = f"Threads Post: {thread_content}\n\nLinked Content: {scraped_content}"
            return combined_content, scraped_content, content_source, scraped_count
        else:
            return thread_content, None, content_source, scraped_count


    def assess_risk_level_batch(self, batch_threads: List[BatchPostData], llm_wrapper) -> List[BatchRiskAssessment]:
        """Assess risk level for a batch of threads in a single API call"""
        try:
            if not batch_threads:
                return []
            
            # Create batch prompt for all threads
            batch_prompt = self._create_batch_risk_assessment_prompt(batch_threads)
            
            # Single API call for the entire batch
            response = llm_wrapper.invoke(batch_prompt)
            response_text = getattr(response, 'content', str(response)).strip()
            
            # Parse the batch response
            risk_assessments = self._parse_batch_risk_response(response_text, batch_threads)
            
            logger.info(f"Batch risk assessment completed for {len(batch_threads)} threads")
            return risk_assessments
            
        except Exception as e:
            logger.warning(f"Batch risk assessment failed: {e} - defaulting all to LOW")
            return [BatchRiskAssessment(post_id=thread.post_id, risk_level='LOW') for thread in batch_threads]

    def _create_batch_risk_assessment_prompt(self, batch_threads: List[BatchPostData]) -> str:
        """Create a single prompt for batch risk assessment"""
        
        batch_prompt = """You are an expert misinformation detector. Analyze the following batch of Threads posts and assign risk levels.

For EACH post, respond with exactly this format:
POST_ID: [post_id] | RISK: [HIGH/MEDIUM/LOW] | REASON: [brief reason]

Risk Level Guidelines:
- HIGH: Contains unverified claims, conspiracy theories, medical misinformation, or political manipulation
- MEDIUM: Potentially misleading, lacks sources, or emotional manipulation  
- LOW: Factual, well-sourced, or clearly opinion-based content

POSTS TO ANALYZE:

"""
        
        for i, thread in enumerate(batch_threads, 1):
            batch_prompt += f"""
--- POST {i} ( ID: {thread.post_id}) ---
Text: {thread.content[:5000]}{'...' if len(thread.content) > 500 else ''}
Author: {thread.author}
Likes: {thread.score} | Comments: {thread.num_comments} | Age: {thread.age_hours:.1f}h
Verified User: {getattr(thread, 'user_verified', False)}

"""
        
        batch_prompt += """
Now provide risk assessment for each post using the exact format:
POST_ID: [post_id] | RISK: [HIGH/MEDIUM/LOW] | REASON: [brief reason]
"""
        
        return batch_prompt

    def _parse_batch_risk_response(self, response_text: str, batch_threads: List[BatchPostData]) -> List[BatchRiskAssessment]:
        """Parse the LLM response for batch risk assessment"""
        assessments = []
        post_id_to_post = {thread.post_id: thread for thread in batch_threads}
        
        # Parse each line looking for the expected format
        lines = response_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if 'POST_ID:' in line and '| RISK:' in line:
                try:
                    parts = line.split('|')
                    if len(parts) >= 2:
                        post_id_part = parts[0].replace('POST_ID:', '').strip()
                        risk_part = parts[1].replace('RISK:', '').strip().upper()
                        reason_part = parts[2].replace('REASON:', '').strip() if len(parts) > 2 else ""
                        
                        if risk_part in ['HIGH', 'MEDIUM', 'LOW'] and post_id_part in post_id_to_post:
                            assessments.append(BatchRiskAssessment(
                                post_id=post_id_part,
                                risk_level=risk_part,
                                reasoning=reason_part
                            ))
                except Exception as e:
                    logger.warning(f"Failed to parse risk assessment line: {line} - {e}")
        
        # Ensure we have assessment for all threads (fill missing with LOW)
        assessed_ids = {a.post_id for a in assessments}
        for thread in batch_threads:
            if thread.post_id not in assessed_ids:
                assessments.append(BatchRiskAssessment(post_id=thread.post_id, risk_level='LOW'))
                logger.warning(f"Missing risk assessment for thread {thread.post_id}, defaulting to LOW")
        
        return assessments

    def _run(self, profile_username: str, limit: int = 10) -> str:
        """
        Scan Threads profile for posts (synchronous, runs in separate thread if needed)
        
        Args:
            profile_username: Threads username (without @)
            limit: Number of threads to scan
        """
        # Check if we're in an async context and need to run in a separate thread
        try:
            import asyncio
            loop = asyncio.get_running_loop()
            # We're in an async context - run in a separate thread
            import threading
            
            result = [None]
            exception = [None]
            
            def run_in_thread():
                try:
                    result[0] = self._run_sync(profile_username, limit)
                except Exception as e:
                    exception[0] = e
            
            thread = threading.Thread(target=run_in_thread)
            thread.start()
            thread.join()
            
            if exception[0]:
                raise exception[0]
            return result[0]
            
        except RuntimeError:
            # No event loop, safe to run directly
            return self._run_sync(profile_username, limit)
    
    def _run_sync(self, profile_username: str, limit: int = 10) -> str:
        """
        Synchronous implementation of Threads scanning
        
        Args:
            profile_username: Threads username (without @)
            limit: Number of threads to scan
        """
        if not self._scraper_available:
            return json.dumps({
                'trending_posts': [],
                'scan_summary': 'ThreadsScraper not available. Install: pip install playwright parsel nested-lookup jmespath && playwright install chromium',
                'processed_count': 0,
                'error': 'Scraper not available'
            }, indent=2)
        
        try:
            trending_threads = []
            processed_count = 0
            
            # Build profile URL
            profile_url = f"https://www.threads.net/@{profile_username}"
            
            logger.info(f"Starting Threads scan for @{profile_username} (limit={limit})")
            
            # Scrape profile and threads
            profile_data = self._threads_scraper.scrape_profile(profile_url, include_threads=True)
            
            if not profile_data or not profile_data.get('threads'):
                logger.warning(f"No threads found for @{profile_username}")
                return json.dumps({
                    'trending_posts': [],
                    'scan_summary': f'No threads found for @{profile_username}',
                    'processed_count': 0
                }, indent=2)
            
            threads = profile_data['threads'][:limit]
            
            # First pass: collect all thread data for batch processing
            candidate_threads = []
            thread_data = {}
            total_scraped_count = 0
            
            for thread in threads:
                processed_count += 1
                
                thread_id = thread.get('id', f"thread_{processed_count}")
                
                # Extract content with link scraping
                combined_content, scraped_content, content_source, scraped_count = self.extract_thread_content(thread)
                total_scraped_count += scraped_count
                
                text_content = combined_content  # Use combined content that includes scraped data
                like_count = thread.get('like_count', 0) or 0  # Handle None
                reply_count = thread.get('reply_count', 0) or 0  # Handle None
                published_on = thread.get('published_on', time.time())
                username = thread.get('username', profile_username)
                user_verified = thread.get('user_verified', False)
                
                # Calculate velocity
                velocity = self.calculate_velocity(thread_id, like_count, published_on)
                engagement_rate = reply_count / max(like_count, 1)
                age_hours = (time.time() - published_on) / 3600
                is_recent = age_hours < 48  # 48 hours (2 days)
                meets_basic_threshold = like_count >= (self._min_like_threshold * 0.3)
                
                # Store thread data
                thread_data[thread_id] = {
                    'thread': thread,
                    'scraped_content': scraped_content,
                    'content_source': content_source,
                    'velocity': velocity,
                    'engagement_rate': engagement_rate,
                    'age_hours': age_hours,
                    'is_recent': is_recent,
                    'meets_basic_threshold': meets_basic_threshold
                }
                
                # Add to batch assessment if meets basic criteria
                if is_recent and meets_basic_threshold:
                    batch_thread = BatchPostData(
                        post_id=thread_id,
                        title=text_content[:100] if text_content else "No content",
                        content=text_content[:10000] if text_content else "",
                        scraped_content=scraped_content[:10000] if scraped_content else None,
                        subreddit=f"threads/@{username}",
                        score=like_count,
                        upvote_ratio=1.0,
                        num_comments=reply_count,
                        age_hours=age_hours,
                        author=username,
                        has_external_content=scraped_content is not None
                    )
                    # Add user_verified as custom attribute
                    batch_thread.user_verified = user_verified
                    candidate_threads.append(batch_thread)
            
            # Batch risk assessment
            logger.info(f"Performing batch risk assessment for {len(candidate_threads)} threads")
            risk_assessments = self.assess_risk_level_batch(candidate_threads, self._llm_wrapper)
            
            # Create risk assessment lookup
            risk_lookup = {assessment.post_id: assessment.risk_level for assessment in risk_assessments}
            
            # Second pass: apply risk levels and filtering
            for batch_thread in candidate_threads:
                thread_id = batch_thread.post_id
                data = thread_data[thread_id]
                thread = data['thread']
                
                # Get risk level from batch assessment
                risk_level = risk_lookup.get(thread_id, 'LOW')
                
                # Apply threshold adjustments based on risk level
                threshold_multiplier = {"HIGH": 0.3, "MEDIUM": 0.5, "LOW": 1.0}
                adjusted_threshold = self._velocity_threshold * threshold_multiplier[risk_level]
                meets_velocity = data['velocity'] >= adjusted_threshold
                meets_likes = thread.get('like_count', 0) >= self._min_like_threshold
                
                if risk_level == 'HIGH':
                    meets_likes = thread.get('like_count', 0) >= (self._min_like_threshold * 0.5)
                
                # Final filtering
                if (meets_velocity and meets_likes and data['is_recent']) or (risk_level == 'HIGH' and data['is_recent']):
                    thread_post = {
                        'post_id': thread_id,
                        'title': thread.get('text', '')[:200],
                        'content': thread.get('text', ''),
                        'selftext': thread.get('text', ''),  # For compatibility
                        'scraped_content': data.get('scraped_content'),  # Include scraped content
                        'content_source': data.get('content_source', 'threads_text'),
                        'author': thread.get('username', ''),
                        'subreddit': f"threads/@{thread.get('username', '')}",
                        'url': thread.get('url', ''),
                        'score': thread.get('like_count', 0),
                        'upvote_ratio': 1.0,
                        'num_comments': thread.get('reply_count', 0),
                        'created_utc': thread.get('published_on', time.time()),
                        'velocity_score': data['velocity'],
                        'engagement_rate': data['engagement_rate'],
                        'risk_level': risk_level,
                        'detected_at': datetime.now().isoformat(),
                        'permalink': thread.get('url', ''),
                        'platform': 'threads',
                        'user_verified': thread.get('user_verified', False)
                    }
                    trending_threads.append(thread_post)
            
            # Sort by combined score
            def combined_score(thread):
                risk_multiplier = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
                return thread['velocity_score'] * risk_multiplier[thread['risk_level']]
            
            trending_threads.sort(key=combined_score, reverse=True)
            
            logger.info(f"Batch processing: assessed {len(candidate_threads)} threads in 1 API call")
            logger.info(f"Scan summary: Scanned @{profile_username} ({processed_count} threads), scraped {total_scraped_count} links, found {len(trending_threads)} trending posts")
            
            result = {
                'trending_posts': trending_threads,  # Keep key name for compatibility
                'scan_summary': f"Scanned @{profile_username} ({processed_count} threads), scraped {total_scraped_count} links, found {len(trending_threads)} trending posts (batch processed)",
                'processed_count': processed_count,
                'scraped_count': total_scraped_count,
                'profile': profile_username,
                'batch_size': len(candidate_threads),
                'platform': 'threads'
            }
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            logger.error(f"Batch processing failed for @{profile_username}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return json.dumps({
                'trending_posts': [],
                'scan_summary': f"Batch processing error: {str(e)}",
                'processed_count': 0,
                'profile': profile_username,
                'error': str(e)
            }, indent=2)
