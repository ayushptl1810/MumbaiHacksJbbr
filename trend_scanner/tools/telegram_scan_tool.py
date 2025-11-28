"""Telegram channel scanning tool for trend_scanner"""

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


class TelegramScanInput(BaseModel):
    channel_name: str = Field(description="Telegram channel username (with or without @)")
    limit: int = Field(default=50, description="Number of messages to scan")


class TelegramScanOutput(BaseModel):
    trending_messages: List[Dict[str, Any]] = Field(description="List of trending Telegram messages found")
    scan_summary: str = Field(description="Summary of the scan results")


class TelegramScanTool(GoogleTool):
    name: str = "telegram_scanner"
    description: str = "Scans Telegram channels for messages and ranks them by potential misinformation risk using Google Agents SDK"

    def __init__(self, api_id: str, api_hash: str, phone: str = None, session_name: str = "trend_scanner_session",
                 llm_wrapper=None, velocity_threshold=50, min_views_threshold=1000, google_api_key=None):
        super().__init__()
        object.__setattr__(self, '_api_id', api_id)
        object.__setattr__(self, '_api_hash', api_hash)
        object.__setattr__(self, '_phone', phone)
        object.__setattr__(self, '_session_name', session_name)
        object.__setattr__(self, '_llm_wrapper', llm_wrapper)
        object.__setattr__(self, '_velocity_threshold', velocity_threshold)
        object.__setattr__(self, '_min_views_threshold', min_views_threshold)
        object.__setattr__(self, '_tracked_messages', {})
        object.__setattr__(self, '_web_scraper', WebContentScraper())
        object.__setattr__(self, '_scraped_cache', {})
        object.__setattr__(self, '_client', None)
        object.__setattr__(self, '_client_connected', False)
        
        # Initialize Google Agents Manager
        try:
            object.__setattr__(self, '_google_agents', GoogleAgentsManager(api_key=google_api_key))
            logger.info("Google Agents SDK initialized for Telegram scanner")
        except Exception as e:
            logger.warning(f"Failed to initialize Google Agents SDK: {e}")
            object.__setattr__(self, '_google_agents', None)

    def _connect_client(self):
        """Connect to Telegram using Telethon"""
        if self._client_connected and self._client:
            return True
        
        try:
            from telethon import TelegramClient
            from telethon.errors import SessionPasswordNeededError
            
            client = TelegramClient(self._session_name, self._api_id, self._api_hash)
            
            # Connect synchronously
            client.connect()
            
            # Check if already authorized
            if not client.is_user_authorized():
                logger.info("Telegram client not authorized, starting authentication...")
                
                # Send code request
                if not self._phone:
                    logger.error("Phone number required for first-time authentication")
                    return False
                
                client.send_code_request(self._phone)
                logger.info(f"Code sent to {self._phone}. Please check your Telegram app.")
                
                # In production, you'd need to handle code input
                # For now, we'll assume session already exists or manual setup
                code = input('Enter the code you received: ')
                
                try:
                    client.sign_in(self._phone, code)
                except SessionPasswordNeededError:
                    password = input('Two-step verification enabled. Please enter your password: ')
                    client.sign_in(password=password)
                
                logger.info("Successfully authenticated with Telegram")
            
            object.__setattr__(self, '_client', client)
            object.__setattr__(self, '_client_connected', True)
            logger.info("Telegram client connected successfully")
            return True
            
        except ImportError:
            logger.error("Telethon not installed. Install with: pip install telethon")
            return False
        except Exception as e:
            logger.error(f"Failed to connect Telegram client: {e}")
            return False

    def calculate_velocity(self, message_id: str, current_views: int, date: float) -> float:
        """Calculate message spread velocity based on views"""
        current_time = time.time()
        if message_id in self._tracked_messages:
            metric = self._tracked_messages[message_id]
            time_diff = current_time - metric.current_time
            views_diff = current_views - metric.current_views
            metric.current_views = current_views
            metric.current_time = current_time
            if time_diff > 0:
                velocity = (views_diff / time_diff) * 3600  # views per hour
                metric.velocity = velocity
                return velocity
            return metric.velocity
        else:
            age_seconds = max(current_time - date, 1.0)
            hours = age_seconds / 3600.0
            proxy_velocity = current_views / hours if hours > 0 else float(current_views) * 3600.0
            self._tracked_messages[message_id] = type('VM', (), {
                'initial_views': current_views,
                'current_views': current_views,
                'initial_time': current_time,
                'current_time': current_time,
                'velocity': proxy_velocity
            })()
            return proxy_velocity

    def extract_message_content(self, message) -> Tuple[str, Optional[str], str, List[str], int]:
        """
        Extract content from a Telegram message, including scraping external links.
        Returns: (combined_content, scraped_content, content_source, media_urls, scraped_count)
        """
        message_text = message.message or ""
        scraped_content = None
        content_source = "telegram_text"
        media_urls = []
        scraped_count = 0
        
        # Extract media URLs
        if message.media:
            try:
                # Handle different media types
                if hasattr(message.media, 'photo'):
                    media_urls.append(f"[Photo: {message.id}]")
                elif hasattr(message.media, 'document'):
                    media_urls.append(f"[Document: {message.id}]")
                elif hasattr(message.media, 'webpage'):
                    webpage = message.media.webpage
                    if hasattr(webpage, 'url'):
                        media_urls.append(webpage.url)
            except Exception as e:
                logger.warning(f"Failed to extract media URLs: {e}")
        
        # Extract URLs from message text
        import re
        url_pattern = r'https?://[^\s]+'
        urls = re.findall(url_pattern, message_text)
        
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
            combined_content = f"Telegram Message: {message_text}\n\nLinked Content: {scraped_content}"
            return combined_content, scraped_content, content_source, media_urls, scraped_count
        else:
            return message_text, None, content_source, media_urls, scraped_count

    def assess_risk_level_batch(self, batch_messages: List[BatchPostData], llm_wrapper) -> List[BatchRiskAssessment]:
        """Assess risk level for a batch of Telegram messages in a single API call"""
        try:
            if not batch_messages:
                return []
            
            # Create batch prompt for all messages
            batch_prompt = self._create_batch_risk_assessment_prompt(batch_messages)
            
            # Single API call for the entire batch
            response = llm_wrapper.invoke(batch_prompt)
            response_text = getattr(response, 'content', str(response)).strip()
            
            # Parse the batch response
            risk_assessments = self._parse_batch_risk_response(response_text, batch_messages)
            
            logger.info(f"Batch risk assessment completed for {len(batch_messages)} Telegram messages")
            return risk_assessments
            
        except Exception as e:
            logger.warning(f"Batch risk assessment failed: {e} - defaulting all to LOW")
            return [BatchRiskAssessment(post_id=msg.post_id, risk_level='LOW') for msg in batch_messages]

    def _create_batch_risk_assessment_prompt(self, batch_messages: List[BatchPostData]) -> str:
        """Create a single prompt for batch risk assessment"""
        
        batch_prompt = """You are an expert misinformation detector. Analyze the following batch of Telegram messages and assign risk levels.

For EACH message, respond with exactly this format:
POST_ID: [post_id] | RISK: [HIGH/MEDIUM/LOW] | REASON: [brief reason]

Risk Level Guidelines:
- HIGH: Contains unverified claims, conspiracy theories, medical misinformation, or political manipulation
- MEDIUM: Potentially misleading, lacks sources, or emotional manipulation  
- LOW: Factual, well-sourced, or clearly opinion-based content

MESSAGES TO ANALYZE:

"""
        
        for i, msg in enumerate(batch_messages, 1):
            batch_prompt += f"""
--- MESSAGE {i} (ID: {msg.post_id}) ---
Text: {msg.content[:5000]}{'...' if len(msg.content) > 5000 else ''}
Channel: {msg.subreddit}
Views: {msg.score} | Age: {msg.age_hours:.1f}h
Author: {msg.author}
Has External Content: {msg.has_external_content}
{f'External Content: {msg.scraped_content[:3000]}...' if msg.scraped_content else ''}

"""
        
        batch_prompt += """
Now provide risk assessment for each message using the exact format:
POST_ID: [post_id] | RISK: [HIGH/MEDIUM/LOW] | REASON: [brief reason]
"""
        
        return batch_prompt

    def _parse_batch_risk_response(self, response_text: str, batch_messages: List[BatchPostData]) -> List[BatchRiskAssessment]:
        """Parse the LLM response for batch risk assessment"""
        assessments = []
        post_id_to_post = {msg.post_id: msg for msg in batch_messages}
        
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
        
        # Ensure we have assessment for all messages (fill missing with LOW)
        assessed_ids = {a.post_id for a in assessments}
        for msg in batch_messages:
            if msg.post_id not in assessed_ids:
                assessments.append(BatchRiskAssessment(post_id=msg.post_id, risk_level='LOW'))
                logger.warning(f"Missing risk assessment for message {msg.post_id}, defaulting to LOW")
        
        return assessments

    def _run(self, channel_name: str, limit: int = 50) -> str:
        """
        Scan Telegram channel for messages
        
        Args:
            channel_name: Telegram channel username (with or without @)
            limit: Number of messages to scan
        """
        # Ensure channel name starts with @
        if not channel_name.startswith('@'):
            channel_name = f'@{channel_name}'
        
        # Connect to Telegram
        if not self._connect_client():
            return json.dumps({
                'trending_posts': [],
                'scan_summary': 'Failed to connect to Telegram. Check credentials and authentication.',
                'processed_count': 0,
                'error': 'Connection failed'
            }, indent=2)
        
        try:
            trending_messages = []
            processed_count = 0
            total_scraped_count = 0
            
            logger.info(f"Starting Telegram scan for {channel_name} (limit={limit})")
            
            # Get channel entity
            try:
                channel = self._client.get_entity(channel_name)
            except Exception as e:
                logger.error(f"Failed to get channel {channel_name}: {e}")
                return json.dumps({
                    'trending_posts': [],
                    'scan_summary': f'Channel {channel_name} not found or not accessible',
                    'processed_count': 0,
                    'error': str(e)
                }, indent=2)
            
            # Get messages from channel
            messages = self._client.get_messages(channel, limit=limit)
            
            if not messages:
                logger.warning(f"No messages found in {channel_name}")
                return json.dumps({
                    'trending_posts': [],
                    'scan_summary': f'No messages found in {channel_name}',
                    'processed_count': 0
                }, indent=2)
            
            # First pass: collect all message data for batch processing
            candidate_messages = []
            message_data = {}
            
            for message in messages:
                processed_count += 1
                
                # Skip messages without content
                if not message.message and not message.media:
                    continue
                
                message_id = f"{channel_name}/{message.id}"
                
                # Extract content with link scraping
                combined_content, scraped_content, content_source, media_urls, scraped_count = self.extract_message_content(message)
                total_scraped_count += scraped_count
                
                views = message.views or 0
                forwards = message.forwards or 0
                date_timestamp = message.date.timestamp() if message.date else time.time()
                
                # Calculate velocity
                velocity = self.calculate_velocity(message_id, views, date_timestamp)
                engagement_rate = forwards / max(views, 1)
                age_hours = (time.time() - date_timestamp) / 3600
                is_recent = age_hours < 72  # 72 hours (3 days)
                meets_basic_threshold = views >= (self._min_views_threshold * 0.3)
                
                # Store message data
                message_data[message_id] = {
                    'message': message,
                    'combined_content': combined_content,
                    'scraped_content': scraped_content,
                    'content_source': content_source,
                    'media_urls': media_urls,
                    'velocity': velocity,
                    'engagement_rate': engagement_rate,
                    'age_hours': age_hours,
                    'is_recent': is_recent,
                    'meets_basic_threshold': meets_basic_threshold
                }
                
                # Add to batch assessment if meets basic criteria
                if is_recent and meets_basic_threshold:
                    batch_message = BatchPostData(
                        post_id=message_id,
                        title=combined_content[:100] if combined_content else "No content",
                        content=combined_content[:10000] if combined_content else "",
                        scraped_content=scraped_content[:10000] if scraped_content else None,
                        subreddit=f"telegram/{channel_name}",
                        score=views,
                        upvote_ratio=1.0,
                        num_comments=forwards,
                        age_hours=age_hours,
                        author=channel_name,
                        has_external_content=scraped_content is not None or len(media_urls) > 0
                    )
                    candidate_messages.append(batch_message)
            
            # Batch risk assessment
            logger.info(f"Performing batch risk assessment for {len(candidate_messages)} messages")
            risk_assessments = self.assess_risk_level_batch(candidate_messages, self._llm_wrapper)
            
            # Create risk assessment lookup
            risk_lookup = {assessment.post_id: assessment.risk_level for assessment in risk_assessments}
            
            # Second pass: apply risk levels and filtering
            for batch_message in candidate_messages:
                message_id = batch_message.post_id
                data = message_data[message_id]
                message = data['message']
                
                # Get risk level from batch assessment
                risk_level = risk_lookup.get(message_id, 'LOW')
                
                # Apply threshold adjustments based on risk level
                threshold_multiplier = {"HIGH": 0.3, "MEDIUM": 0.5, "LOW": 1.0}
                adjusted_threshold = self._velocity_threshold * threshold_multiplier[risk_level]
                meets_velocity = data['velocity'] >= adjusted_threshold
                meets_views = message.views >= self._min_views_threshold
                
                if risk_level == 'HIGH':
                    meets_views = message.views >= (self._min_views_threshold * 0.5)
                
                # Final filtering
                if (meets_velocity and meets_views and data['is_recent']) or (risk_level == 'HIGH' and data['is_recent']):
                    message_post = {
                        'post_id': message_id,
                        'title': data['combined_content'][:200] if data['combined_content'] else "No content",
                        'content': data['combined_content'],
                        'scraped_content': data['scraped_content'],
                        'content_source': data['content_source'],
                        'author': channel_name,
                        'channel': channel_name,
                        'subreddit': f"telegram/{channel_name}",
                        'url': f"https://t.me/{channel_name.replace('@', '')}/{message.id}",
                        'score': message.views or 0,
                        'views': message.views or 0,
                        'forwards': message.forwards or 0,
                        'upvote_ratio': 1.0,
                        'num_comments': message.forwards or 0,
                        'created_utc': message.date.timestamp() if message.date else time.time(),
                        'velocity_score': data['velocity'],
                        'engagement_rate': data['engagement_rate'],
                        'risk_level': risk_level,
                        'detected_at': datetime.now().isoformat(),
                        'permalink': f"https://t.me/{channel_name.replace('@', '')}/{message.id}",
                        'platform': 'telegram',
                        'media_urls': data['media_urls'],
                        'has_media': len(data['media_urls']) > 0
                    }
                    trending_messages.append(message_post)
            
            # Sort by combined score
            def combined_score(msg):
                risk_multiplier = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
                return msg['velocity_score'] * risk_multiplier[msg['risk_level']]
            
            trending_messages.sort(key=combined_score, reverse=True)
            
            logger.info(f"Batch processing: assessed {len(candidate_messages)} messages in 1 API call")
            logger.info(f"Scan summary: Scanned {channel_name} ({processed_count} messages), scraped {total_scraped_count} links, found {len(trending_messages)} trending messages")
            
            result = {
                'trending_posts': trending_messages,
                'scan_summary': f"Scanned {channel_name} ({processed_count} messages), scraped {total_scraped_count} links, found {len(trending_messages)} trending messages (batch processed)",
                'processed_count': processed_count,
                'scraped_count': total_scraped_count,
                'channel': channel_name,
                'batch_size': len(candidate_messages),
                'platform': 'telegram'
            }
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            logger.error(f"Batch processing failed for {channel_name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return json.dumps({
                'trending_posts': [],
                'scan_summary': f"Batch processing error: {str(e)}",
                'processed_count': 0,
                'channel': channel_name,
                'error': str(e)
            }, indent=2)
    
    def __del__(self):
        """Cleanup: disconnect Telegram client"""
        try:
            if self._client and self._client_connected:
                self._client.disconnect()
                logger.info("Telegram client disconnected")
        except:
            pass
