"""Twitter/X scanning tool for trend_scanner using Twikit"""

import time
import json
import logging
import hashlib
import os
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


class TwitterScanInput(BaseModel):
    target: str = Field(description="Twitter username (without @) or search query")
    scan_type: str = Field(default="user", description="Scan type: 'user', 'trending', or 'both'")
    limit: int = Field(default=50, description="Number of tweets to scan")


class TwitterScanOutput(BaseModel):
    trending_tweets: List[Dict[str, Any]] = Field(description="List of trending tweets found")
    scan_summary: str = Field(description="Summary of the scan results")


class TwitterScanTool(GoogleTool):
    name: str = "twitter_scanner"
    description: str = "Scans Twitter/X for tweets from users or trending topics and ranks them by potential misinformation risk using Google Agents SDK"

    def __init__(self, username: str = None, email: str = None, password: str = None, 
                 cookies_file: str = None, llm_wrapper=None, velocity_threshold=200, 
                 min_engagement_threshold=100, google_api_key=None):
        super().__init__()
        object.__setattr__(self, '_username', username)
        object.__setattr__(self, '_email', email)
        object.__setattr__(self, '_password', password)
        object.__setattr__(self, '_cookies_file', cookies_file)
        object.__setattr__(self, '_llm_wrapper', llm_wrapper)
        object.__setattr__(self, '_velocity_threshold', velocity_threshold)
        object.__setattr__(self, '_min_engagement_threshold', min_engagement_threshold)
        object.__setattr__(self, '_tracked_tweets', {})
        object.__setattr__(self, '_web_scraper', WebContentScraper())
        object.__setattr__(self, '_scraped_cache', {})
        object.__setattr__(self, '_client', None)
        object.__setattr__(self, '_client_authenticated', False)
        
        # Initialize Google Agents Manager
        try:
            object.__setattr__(self, '_google_agents', GoogleAgentsManager(api_key=google_api_key))
            logger.info("Google Agents SDK initialized for Twitter scanner")
        except Exception as e:
            logger.warning(f"Failed to initialize Google Agents SDK: {e}")
            object.__setattr__(self, '_google_agents', None)

    def _authenticate_client(self):
        """Authenticate Twitter client using Twikit (sync wrapper)"""
        if self._client_authenticated and self._client:
            return True
        
        # Run async authentication in event loop
        import asyncio
        import threading
        
        # Check if there's already a running event loop
        try:
            loop = asyncio.get_running_loop()
            # We're in an async context - run in a separate thread
            result_container = [False]
            exception_container = [None]
            
            def auth_in_thread():
                try:
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        result = new_loop.run_until_complete(self._authenticate_client_async())
                        result_container[0] = result
                    finally:
                        new_loop.close()
                except Exception as e:
                    exception_container[0] = e
            
            thread = threading.Thread(target=auth_in_thread)
            thread.start()
            thread.join()
            
            if exception_container[0]:
                raise exception_container[0]
            return result_container[0]
            
        except RuntimeError:
            # No running event loop, safe to create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(self._authenticate_client_async())
                return result
            finally:
                loop.close()
    
    async def _authenticate_client_async(self):
        """Async authentication for Twitter client using Twikit"""
        if self._client_authenticated and self._client:
            return True
        
        try:
            from twikit import Client
            
            client = Client('en-US')
            
            # Try to load cookies first
            if self._cookies_file and os.path.exists(self._cookies_file):
                logger.info(f"Loading cookies from {self._cookies_file}")
                client.load_cookies(self._cookies_file)
                object.__setattr__(self, '_client', client)
                object.__setattr__(self, '_client_authenticated', True)
                logger.info("Twitter client authenticated from cookies")
                return True
            
            # If no cookies, authenticate with credentials
            if not self._username or not self._password:
                logger.error("Twitter credentials required for authentication")
                return False
            
            logger.info(f"Authenticating Twitter client for user: {self._username}")
            
            # Login (email is optional but recommended) - ASYNC
            if self._email:
                await client.login(
                    auth_info_1=self._username,
                    auth_info_2=self._email,
                    password=self._password
                )
            else:
                await client.login(
                    auth_info_1=self._username,
                    password=self._password
                )
            
            # Save cookies for future use
            if self._cookies_file:
                client.save_cookies(self._cookies_file)
                logger.info(f"Cookies saved to {self._cookies_file}")
            
            object.__setattr__(self, '_client', client)
            object.__setattr__(self, '_client_authenticated', True)
            logger.info("Twitter client authenticated successfully")
            return True
            
        except ImportError:
            logger.error("Twikit not installed. Install with: pip install twikit")
            return False
        except Exception as e:
            logger.error(f"Failed to authenticate Twitter client: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def get_trending_topics(self, limit: int = 10) -> List[str]:
        """
        Fetch trending topics from Twitter
        
        Args:
            limit: Maximum number of trending topics to return
            
        Returns:
            List of trending topic names/hashtags
        """
        if not self._authenticate_client():
            logger.error("Cannot fetch trending topics - authentication failed")
            return []
        
        try:
            logger.info("Fetching Twitter trending topics...")
            
            # Twikit's get_trends is async, so we need to run it synchronously
            import asyncio
            
            # Check if we're already in an async context
            try:
                loop = asyncio.get_running_loop()
                # We're in an async context - run in a separate thread
                import threading
                
                result = [None]
                exception = [None]
                
                def fetch_in_thread():
                    try:
                        # Create new event loop for this thread
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        try:
                            trends = new_loop.run_until_complete(self._client.get_trends('trending'))
                            result[0] = trends
                        finally:
                            new_loop.close()
                    except Exception as e:
                        exception[0] = e
                
                thread = threading.Thread(target=fetch_in_thread)
                thread.start()
                thread.join()
                
                if exception[0]:
                    raise exception[0]
                trends = result[0]
                
            except RuntimeError:
                # No event loop, safe to run directly
                trends = asyncio.run(self._client.get_trends('trending'))
            
            if not trends:
                logger.warning("No trending topics found")
                return []
            
            # Extract topic names
            trending_topics = []
            for trend in trends[:limit]:
                # Twikit returns trends as objects with 'name' attribute
                topic_name = trend.name if hasattr(trend, 'name') else str(trend)
                trending_topics.append(topic_name)
                logger.info(f"Found trending topic: {topic_name}")
            
            logger.info(f"Successfully fetched {len(trending_topics)} trending topics")
            return trending_topics
            
        except Exception as e:
            error_msg = str(e)
            if '403' in error_msg or 'Forbidden' in error_msg:
                logger.warning(f"Twitter trending topics API returned 403 Forbidden - this feature may be restricted")
                logger.info("Suggestion: Use manual keywords in TARGET_TWITTER_KEYWORDS or set TWITTER_AUTO_DISCOVER_KEYWORDS=False")
            else:
                logger.error(f"Failed to fetch trending topics: {e}")
                import traceback
                logger.error(traceback.format_exc())
            return []

    def calculate_velocity(self, tweet_id: str, current_engagement: int, created_at: float) -> float:
        """Calculate tweet engagement velocity"""
        current_time = time.time()
        if tweet_id in self._tracked_tweets:
            metric = self._tracked_tweets[tweet_id]
            time_diff = current_time - metric.current_time
            engagement_diff = current_engagement - metric.current_engagement
            metric.current_engagement = current_engagement
            metric.current_time = current_time
            if time_diff > 0:
                velocity = (engagement_diff / time_diff) * 3600  # per hour
                metric.velocity = velocity
                return velocity
            return metric.velocity
        else:
            age_seconds = max(current_time - created_at, 1.0)
            hours = age_seconds / 3600.0
            proxy_velocity = current_engagement / hours if hours > 0 else float(current_engagement) * 3600.0
            self._tracked_tweets[tweet_id] = type('VM', (), {
                'initial_engagement': current_engagement,
                'current_engagement': current_engagement,
                'initial_time': current_time,
                'current_time': current_time,
                'velocity': proxy_velocity
            })()
            return proxy_velocity

    def extract_tweet_content(self, tweet) -> Tuple[str, Optional[str], str, List[str], int]:
        """
        Extract content from a tweet, including scraping external links.
        Returns: (combined_content, scraped_content, content_source, media_urls, scraped_count)
        """
        tweet_text = tweet.text or ""
        scraped_content = None
        content_source = "twitter_text"
        media_urls = []
        scraped_count = 0
        
        # Extract media URLs
        if hasattr(tweet, 'media') and tweet.media:
            for media in tweet.media:
                if hasattr(media, 'media_url_https'):
                    media_urls.append(media.media_url_https)
                elif hasattr(media, 'url'):
                    media_urls.append(media.url)
        
        # Extract URLs from tweet text
        import re
        url_pattern = r'https?://[^\s]+'
        urls = re.findall(url_pattern, tweet_text)
        
        # Filter out t.co links and media links
        urls = [url for url in urls if 't.co' not in url and url not in media_urls]
        
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
            combined_content = f"Tweet: {tweet_text}\n\nLinked Content: {scraped_content}"
            return combined_content, scraped_content, content_source, media_urls, scraped_count
        else:
            return tweet_text, None, content_source, media_urls, scraped_count

    def assess_risk_level_batch(self, batch_tweets: List[BatchPostData], llm_wrapper) -> List[BatchRiskAssessment]:
        """Assess risk level for a batch of tweets in a single API call"""
        try:
            if not batch_tweets:
                return []
            
            # Create batch prompt for all tweets
            batch_prompt = self._create_batch_risk_assessment_prompt(batch_tweets)
            
            # Single API call for the entire batch
            response = llm_wrapper.invoke(batch_prompt)
            response_text = getattr(response, 'content', str(response)).strip()
            
            # Parse the batch response
            risk_assessments = self._parse_batch_risk_response(response_text, batch_tweets)
            
            logger.info(f"Batch risk assessment completed for {len(batch_tweets)} tweets")
            return risk_assessments
            
        except Exception as e:
            logger.warning(f"Batch risk assessment failed: {e} - defaulting all to LOW")
            return [BatchRiskAssessment(post_id=tweet.post_id, risk_level='LOW') for tweet in batch_tweets]

    def _create_batch_risk_assessment_prompt(self, batch_tweets: List[BatchPostData]) -> str:
        """Create a single prompt for batch risk assessment"""
        
        batch_prompt = """You are an expert misinformation detector. Analyze the following batch of tweets and assign risk levels.

For EACH tweet, respond with exactly this format:
POST_ID: [post_id] | RISK: [HIGH/MEDIUM/LOW] | REASON: [brief reason]

Risk Level Guidelines:
- HIGH: Contains unverified claims, conspiracy theories, medical misinformation, or political manipulation
- MEDIUM: Potentially misleading, lacks sources, or emotional manipulation  
- LOW: Factual, well-sourced, or clearly opinion-based content

TWEETS TO ANALYZE:

"""
        
        for i, tweet in enumerate(batch_tweets, 1):
            batch_prompt += f"""
--- TWEET {i} (ID: {tweet.post_id}) ---
Text: {tweet.content[:5000]}{'...' if len(tweet.content) > 5000 else ''}
Author: {tweet.author}
Likes: {tweet.score} | Retweets: {tweet.num_comments} | Age: {tweet.age_hours:.1f}h
Verified User: {getattr(tweet, 'user_verified', False)}
Has External Content: {tweet.has_external_content}
{f'External Content: {tweet.scraped_content[:3000]}...' if tweet.scraped_content else ''}

"""
        
        batch_prompt += """
Now provide risk assessment for each tweet using the exact format:
POST_ID: [post_id] | RISK: [HIGH/MEDIUM/LOW] | REASON: [brief reason]
"""
        
        return batch_prompt

    def _parse_batch_risk_response(self, response_text: str, batch_tweets: List[BatchPostData]) -> List[BatchRiskAssessment]:
        """Parse the LLM response for batch risk assessment"""
        assessments = []
        post_id_to_post = {tweet.post_id: tweet for tweet in batch_tweets}
        
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
        
        # Ensure we have assessment for all tweets (fill missing with LOW)
        assessed_ids = {a.post_id for a in assessments}
        for tweet in batch_tweets:
            if tweet.post_id not in assessed_ids:
                assessments.append(BatchRiskAssessment(post_id=tweet.post_id, risk_level='LOW'))
                logger.warning(f"Missing risk assessment for tweet {tweet.post_id}, defaulting to LOW")
        
        return assessments

    def _run(self, target: str, scan_type: str = "user", limit: int = 50) -> str:
        """
        Scan Twitter for tweets - Always runs sequentially (no parallel execution)
        
        Args:
            target: Username (without @) for user scan, or keyword for trending/search
            scan_type: 'user', 'trending', or 'both'
            limit: Number of tweets to scan
        """
        # Twitter scans always run sequentially to avoid event loop conflicts
        # This is necessary because Twikit is fully async and doesn't support concurrent execution
        return self._run_sync(target, scan_type, limit)
    
    def _run_sync(self, target: str, scan_type: str = "user", limit: int = 50) -> str:
        """
        Synchronous wrapper for async Twitter scanning
        
        Args:
            target: Username (without @) for user scan, or keyword for trending/search
            scan_type: 'user', 'trending', or 'both'
            limit: Number of tweets to scan
        """
        # Authenticate client
        if not self._authenticate_client():
            return json.dumps({
                'trending_posts': [],
                'scan_summary': 'Failed to authenticate Twitter client. Check credentials.',
                'processed_count': 0,
                'error': 'Authentication failed'
            }, indent=2)
        
        # Run the async scanning in a new event loop
        import asyncio
        
        try:
            # Try to get existing event loop
            try:
                loop = asyncio.get_running_loop()
                # We're in an async context, need to use thread executor
                import concurrent.futures
                import threading
                
                result_container = []
                exception_container = []
                
                def run_in_new_loop():
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        result = new_loop.run_until_complete(self._run_async(target, scan_type, limit))
                        result_container.append(result)
                    except Exception as e:
                        exception_container.append(e)
                    finally:
                        new_loop.close()
                
                thread = threading.Thread(target=run_in_new_loop)
                thread.start()
                thread.join()
                
                if exception_container:
                    raise exception_container[0]
                return result_container[0]
                
            except RuntimeError:
                # No running event loop, safe to create one
                pass
            
            # Always create a fresh event loop for each scan
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(self._run_async(target, scan_type, limit))
                return result
            finally:
                # Properly cleanup async resources
                try:
                    # Cancel all pending tasks
                    pending = asyncio.all_tasks(loop)
                    for task in pending:
                        task.cancel()
                    # Run loop one more time to complete cancellations
                    if pending:
                        loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                except Exception:
                    pass
                finally:
                    loop.close()
                    # Clear the event loop reference to avoid conflicts
                    asyncio.set_event_loop(None)
                    
        except Exception as e:
            logger.error(f"Twitter scan failed: {e}")
            return json.dumps({
                'trending_posts': [],
                'scan_summary': f'Twitter scan error: {str(e)}',
                'processed_count': 0,
                'error': str(e)
            }, indent=2)
    
    async def _run_async(self, target: str, scan_type: str = "user", limit: int = 50) -> str:
        """
        Async implementation of Twitter scanning
        
        Args:
            target: Username (without @) for user scan, or keyword for trending/search
            scan_type: 'user', 'trending', or 'both'
            limit: Number of tweets to scan
        """
        try:
            trending_tweets = []
            processed_count = 0
            total_scraped_count = 0
            
            logger.info(f"Starting Twitter scan (type={scan_type}, target={target}, limit={limit})")
            
            # Create a fresh client for this scan to avoid event loop conflicts
            # The httpx client inside Twikit keeps references to the event loop
            from twikit import Client
            client = Client(language='en-US')
            
            # Re-authenticate with saved cookies
            if self._cookies_file and os.path.exists(self._cookies_file):
                client.load_cookies(self._cookies_file)
                logger.info(f"Loaded cookies for fresh client")
            else:
                # Fall back to login if no cookies
                await self._authenticate_client_async()
                client = self._client
            
            tweets = []
            
            # Scan based on type
            if scan_type in ["user", "both"]:
                logger.info(f"Fetching tweets from user: @{target}")
                try:
                    user = await client.get_user_by_screen_name(target)
                    user_tweets = await user.get_tweets('Tweets', count=limit)
                    tweets.extend(user_tweets)
                    logger.info(f"Fetched {len(user_tweets)} tweets from @{target}")
                except Exception as e:
                    error_msg = str(e)
                    if '403' in error_msg or 'Forbidden' in error_msg:
                        logger.warning(f"Twitter returned 403 Forbidden for user @{target}")
                        logger.info("This may be due to: rate limiting, account restrictions, or API access issues")
                    else:
                        logger.error(f"Failed to fetch user tweets: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
            
            if scan_type in ["trending", "both"]:
                logger.info(f"Searching tweets for: {target}")
                try:
                    # Search tweets using Twikit format: search_tweet(query, product_type)
                    # Product types: 'Top' (most popular) or 'Latest' (most recent)
                    search_results = await client.search_tweet(target, 'Latest')
                    # Limit to top 50 tweets from search results
                    limited_results = list(search_results)[:limit] if search_results else []
                    tweets.extend(limited_results)
                    logger.info(f"Fetched {len(limited_results)} tweets from search (limited to {limit})")
                except Exception as e:
                    error_msg = str(e)
                    if '403' in error_msg or 'Forbidden' in error_msg:
                        logger.warning(f"Twitter returned 403 Forbidden for search '{target}'")
                        logger.info("This may be due to: rate limiting, account restrictions, or API access issues")
                    elif '404' in error_msg or 'NotFound' in error_msg:
                        logger.warning(f"Twitter returned 404 Not Found for search '{target}' - no results available")
                    else:
                        logger.error(f"Failed to search tweets: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
            
            if not tweets:
                logger.warning(f"No tweets found for target: {target}")
                return json.dumps({
                    'trending_posts': [],
                    'scan_summary': f'No tweets found for {target} (type: {scan_type})',
                    'processed_count': 0
                }, indent=2)
            
            # First pass: collect all tweet data for batch processing
            candidate_tweets = []
            tweet_data = {}
            
            for tweet in tweets[:limit]:  # Limit total tweets processed
                processed_count += 1
                
                tweet_id = tweet.id
                
                # Extract content with link scraping
                combined_content, scraped_content, content_source, media_urls, scraped_count = self.extract_tweet_content(tweet)
                total_scraped_count += scraped_count
                
                # Get engagement metrics
                likes = getattr(tweet, 'favorite_count', 0) or 0
                retweets = getattr(tweet, 'retweet_count', 0) or 0
                replies = getattr(tweet, 'reply_count', 0) or 0
                total_engagement = likes + retweets + replies
                
                # Get created time
                created_at = getattr(tweet, 'created_at_datetime', None)
                if created_at:
                    created_timestamp = created_at.timestamp()
                else:
                    created_timestamp = time.time()
                
                # Calculate velocity
                velocity = self.calculate_velocity(tweet_id, total_engagement, created_timestamp)
                engagement_rate = (retweets + replies) / max(likes, 1)
                age_hours = (time.time() - created_timestamp) / 3600
                is_recent = age_hours < 48  # 48 hours
                meets_basic_threshold = total_engagement >= (self._min_engagement_threshold * 0.3)
                
                # Get user info
                user = tweet.user
                username = user.screen_name if user else target
                user_verified = getattr(user, 'verified', False) if user else False
                
                # Store tweet data
                tweet_data[tweet_id] = {
                    'tweet': tweet,
                    'combined_content': combined_content,
                    'scraped_content': scraped_content,
                    'content_source': content_source,
                    'media_urls': media_urls,
                    'velocity': velocity,
                    'engagement_rate': engagement_rate,
                    'age_hours': age_hours,
                    'is_recent': is_recent,
                    'meets_basic_threshold': meets_basic_threshold,
                    'user_verified': user_verified
                }
                
                # Add to batch assessment if meets basic criteria
                if is_recent and meets_basic_threshold:
                    batch_tweet = BatchPostData(
                        post_id=tweet_id,
                        title=combined_content[:100] if combined_content else "No content",
                        content=combined_content[:10000] if combined_content else "",
                        scraped_content=scraped_content[:10000] if scraped_content else None,
                        subreddit=f"twitter/@{username}",
                        score=likes,
                        upvote_ratio=1.0,
                        num_comments=retweets + replies,
                        age_hours=age_hours,
                        author=username,
                        has_external_content=scraped_content is not None or len(media_urls) > 0
                    )
                    batch_tweet.user_verified = user_verified
                    batch_tweet.retweets = retweets
                    batch_tweet.replies = replies
                    candidate_tweets.append(batch_tweet)
            
            # Batch risk assessment
            logger.info(f"Performing batch risk assessment for {len(candidate_tweets)} tweets")
            risk_assessments = self.assess_risk_level_batch(candidate_tweets, self._llm_wrapper)
            
            # Create risk assessment lookup
            risk_lookup = {assessment.post_id: assessment.risk_level for assessment in risk_assessments}
            
            # Second pass: apply risk levels and filtering
            for batch_tweet in candidate_tweets:
                tweet_id = batch_tweet.post_id
                data = tweet_data[tweet_id]
                tweet = data['tweet']
                
                # Get risk level from batch assessment
                risk_level = risk_lookup.get(tweet_id, 'LOW')
                
                # Apply threshold adjustments based on risk level
                threshold_multiplier = {"HIGH": 0.3, "MEDIUM": 0.5, "LOW": 1.0}
                adjusted_threshold = self._velocity_threshold * threshold_multiplier[risk_level]
                meets_velocity = data['velocity'] >= adjusted_threshold
                
                likes = getattr(tweet, 'favorite_count', 0) or 0
                retweets = getattr(tweet, 'retweet_count', 0) or 0
                replies = getattr(tweet, 'reply_count', 0) or 0
                total_engagement = likes + retweets + replies
                
                meets_engagement = total_engagement >= self._min_engagement_threshold
                
                # Debug logging for threshold checks
                logger.debug(f"Tweet {tweet_id[:10]}: velocity={data['velocity']:.1f} (threshold={adjusted_threshold:.1f}), "
                           f"engagement={total_engagement} (threshold={self._min_engagement_threshold}), "
                           f"risk={risk_level}, meets_velocity={meets_velocity}, meets_engagement={meets_engagement}")
                
                if risk_level == 'HIGH':
                    meets_engagement = total_engagement >= (self._min_engagement_threshold * 0.5)
                
                # Final filtering
                if (meets_velocity and meets_engagement and data['is_recent']) or (risk_level == 'HIGH' and data['is_recent']):
                    user = tweet.user
                    username = user.screen_name if user else target
                    
                    tweet_post = {
                        'post_id': tweet_id,
                        'title': data['combined_content'][:200] if data['combined_content'] else "No content",
                        'content': data['combined_content'],
                        'scraped_content': data['scraped_content'],
                        'content_source': data['content_source'],
                        'author': username,
                        'subreddit': f"twitter/@{username}",
                        'url': f"https://twitter.com/{username}/status/{tweet_id}",
                        'score': likes,
                        'likes': likes,
                        'retweets': retweets,
                        'replies': replies,
                        'upvote_ratio': 1.0,
                        'num_comments': retweets + replies,
                        'created_utc': data['tweet'].created_at_datetime.timestamp() if hasattr(data['tweet'], 'created_at_datetime') and data['tweet'].created_at_datetime else time.time(),
                        'velocity_score': data['velocity'],
                        'engagement_rate': data['engagement_rate'],
                        'risk_level': risk_level,
                        'detected_at': datetime.now().isoformat(),
                        'permalink': f"https://twitter.com/{username}/status/{tweet_id}",
                        'platform': 'twitter',
                        'media_urls': data['media_urls'],
                        'has_media': len(data['media_urls']) > 0,
                        'user_verified': data['user_verified']
                    }
                    trending_tweets.append(tweet_post)
            
            # Sort by combined score
            def combined_score(tweet):
                risk_multiplier = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
                return tweet['velocity_score'] * risk_multiplier[tweet['risk_level']]
            
            trending_tweets.sort(key=combined_score, reverse=True)
            
            logger.info(f"Batch processing: assessed {len(candidate_tweets)} tweets in 1 API call")
            logger.info(f"Scan summary: Scanned {target} ({processed_count} tweets, type={scan_type}), scraped {total_scraped_count} links, found {len(trending_tweets)} trending tweets")
            
            result = {
                'trending_posts': trending_tweets,
                'scan_summary': f"Scanned {target} ({processed_count} tweets, type={scan_type}), scraped {total_scraped_count} links, found {len(trending_tweets)} trending tweets (batch processed)",
                'processed_count': processed_count,
                'scraped_count': total_scraped_count,
                'target': target,
                'scan_type': scan_type,
                'batch_size': len(candidate_tweets),
                'platform': 'twitter'
            }
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            logger.error(f"Batch processing failed for {target}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return json.dumps({
                'trending_posts': [],
                'scan_summary': f"Batch processing error: {str(e)}",
                'processed_count': 0,
                'target': target,
                'error': str(e)
            }, indent=2)

    def __del__(self):
        """Cleanup"""
        pass
