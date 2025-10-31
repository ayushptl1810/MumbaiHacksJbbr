import time
import json
import logging
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from pydantic import BaseModel, Field
from .scraper import WebContentScraper
from .google_agents import GoogleAgentsManager
from .models import BatchPostData, BatchRiskAssessment

logger = logging.getLogger(__name__)


# Tool base class to replace CrewAI BaseTool
class GoogleTool:
    def __init__(self):
        self.name = getattr(self, 'name', self.__class__.__name__)
        self.description = getattr(self, 'description', 'A Google-powered tool')
    
    def _run(self, *args, **kwargs):
        raise NotImplementedError("Tool must implement _run method")
    
    def run(self, *args, **kwargs):
        return self._run(*args, **kwargs)


# Long-form prompts moved here for easier editing and reuse by crew code.
RISK_ASSESSMENT_PROMPT = """
Analyze this Reddit post for potential misinformation risk. Consider both the Reddit metadata and the actual content.

REDDIT METADATA:
- Title: {title}
- Subreddit: r/{subreddit}
- Score: {score} upvotes
- Upvote Ratio: {upvote_ratio}
- Comments: {num_comments}
- Age: {age_hours} hours
- Author: {author}
- Has External Content: {has_external_content}

CONTENT TO ANALYZE:
{content}...

RISK ASSESSMENT CRITERIA:
1. HIGH RISK indicators:
    - Claims about health/medical information without sources
    - Political conspiracy theories or election fraud claims
    - Sensational headlines with unverified information
    - Content from known unreliable sources
    - Manipulated statistics or cherry-picked data
    - Urgent/emergency claims designed to cause panic
    - Anti-science or pseudoscience content

2. MEDIUM RISK indicators:
    - Controversial topics with one-sided presentation
    - Emotional language designed to provoke strong reactions
    - Unverified claims about current events
    - Content lacking proper sources or context
    - Potentially misleading headlines

3. LOW RISK indicators:
    - Well-sourced information from reputable outlets
    - Personal opinions clearly marked as such
    - Entertainment or non-factual content
    - Properly contextualized information

ADDITIONAL FACTORS:
- Unusual engagement patterns (comments vs upvotes ratio)
- Rapid viral growth in controversial subreddits
- Content that contradicts established scientific consensus

Based on the content analysis above, respond with ONLY one word: HIGH, MEDIUM, or LOW
"""

SCAN_TASK_PROMPT = """
Perform comprehensive scanning of r/{subreddit} for trending posts that could contain misinformation.

Your enhanced analysis should include:
1. Posts with high velocity (rapid upvote growth)
2. Content analysis of Reddit posts AND linked external content
3. Detection of suspicious claims, unsourced assertions, or misleading information
4. Evaluation of source credibility for linked content
5. Assessment of emotional manipulation techniques
6. Identification of conspiracy theories or pseudoscience

Subreddit: {subreddit}
Scan parameters: limit=20, sort_type=new

Pay special attention to:
- Posts linking to external articles or sources
- Health/medical misinformation
- Political conspiracy theories
- Pseudoscientific claims
- Sensational headlines with unverified content

Return detailed information about trending posts with full content analysis.
"""

ASSESSMENT_TASK_PROMPT = """
Perform comprehensive analysis of all trending posts found across all subreddits, including full content analysis of scraped external sources.

Scan results from all subreddits:
{scan_results}

Your comprehensive analysis should:
1. Rank posts by potential misinformation risk using both metadata and content analysis
2. Identify the most urgent posts needing immediate verification
3. Categorize posts by misinformation type (health, political, scientific, etc.)
4. Highlight posts with the highest viral potential and risk combination
5. Assess the credibility and bias of external sources that were scraped
6. Identify coordinated misinformation campaigns or similar narratives across posts

Priority factors:
- HIGH risk posts with scraped external content containing false claims
- Posts with high velocity in controversial subreddits
- Content contradicting scientific consensus without proper evidence
- Sensational claims designed to provoke emotional responses
- Unsourced medical or health advice
- Political conspiracy theories gaining rapid traction

For each high-priority post, provide:
- Risk level justification based on content analysis
- Key claims that need fact-checking
- Source credibility assessment
- Viral potential score
- Recommended verification priority
"""


class RedditScanInput(BaseModel):
    subreddit_name: str = Field(description="Name of the subreddit to scan")
    limit: int = Field(default=20, description="Number of posts to scan")
    sort_type: str = Field(default="new", description="Sort type: new, rising, hot")


class RedditScanOutput(BaseModel):
    trending_posts: List[Dict[str, Any]] = Field(description="List of trending posts found")
    scan_summary: str = Field(description="Summary of the scan results")


class RedditScanTool(GoogleTool):
    name: str = "reddit_scanner"
    description: str = "Scans Reddit subreddits for rapidly trending posts and ranks them by potential misinformation risk using Google Agents SDK"

    def __init__(self, reddit_client, llm_wrapper, velocity_threshold=15, min_score_threshold=30, google_api_key=None):
        super().__init__()
        object.__setattr__(self, '_reddit', reddit_client)
        object.__setattr__(self, '_llm_wrapper', llm_wrapper)
        object.__setattr__(self, '_velocity_threshold', velocity_threshold)
        object.__setattr__(self, '_min_score_threshold', min_score_threshold)
        object.__setattr__(self, '_tracked_posts', {})
        object.__setattr__(self, '_scraper', WebContentScraper())
        object.__setattr__(self, '_scraped_cache', {})
        
        # Initialize Google Agents Manager (for enhanced analysis, no fact-checking)
        try:
            object.__setattr__(self, '_google_agents', GoogleAgentsManager(api_key=google_api_key))
            logger.info("Google Agents SDK initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize Google Agents SDK: {e}")
            object.__setattr__(self, '_google_agents', None)

    def calculate_velocity(self, post_id: str, current_score: int, created_utc: float) -> float:
        current_time = time.time()
        if post_id in self._tracked_posts:
            metric = self._tracked_posts[post_id]
            time_diff = current_time - metric.current_time
            score_diff = current_score - metric.current_score
            metric.current_score = current_score
            metric.current_time = current_time
            if time_diff > 0:
                velocity = (score_diff / time_diff) * 3600
                metric.velocity = velocity
                return velocity
            return metric.velocity
        else:
            age_seconds = max(current_time - created_utc, 1.0)
            hours = age_seconds / 3600.0
            proxy_velocity = current_score / hours if hours > 0 else float(current_score) * 3600.0
            self._tracked_posts[post_id] = type('VM', (), {
                'initial_score': current_score,
                'current_score': current_score,
                'initial_time': current_time,
                'current_time': current_time,
                'velocity': proxy_velocity
            })()
            return proxy_velocity

    def extract_post_content(self, submission) -> Tuple[str, Optional[str], str]:
        reddit_content = ""
        scraped_content = None
        content_source = "reddit"

        if submission.selftext:
            reddit_content = submission.selftext
            content_source = "selftext"
        elif submission.url:
            if self._scraper.is_scrapeable_url(submission.url):
                url_hash = hashlib.md5(submission.url.encode()).hexdigest()
                if url_hash in self._scraped_cache:
                    scraped_content = self._scraped_cache[url_hash]
                    content_source = "cached_scraped"
                else:
                    scraped_content, scrape_method = self._scraper.scrape_content(submission.url)
                    if scraped_content:
                        self._scraped_cache[url_hash] = scraped_content
                        content_source = f"scraped_{scrape_method}"
                    else:
                        content_source = "link_failed"
            else:
                reddit_content = f"Link post: {submission.url}"
                content_source = "link_not_scrapeable"

        if scraped_content:
            combined_content = f"Reddit Title: {submission.title}\n"
            if reddit_content:
                combined_content += f"Reddit Content: {reddit_content}\n\n"
            combined_content += f"Linked Content: {scraped_content}"
            return combined_content, scraped_content, content_source
        else:
            return reddit_content or f"Link: {submission.url}", None, content_source

    def assess_risk_level(self, submission, content: str, scraped_content: Optional[str], llm_wrapper) -> str:
        try:
            # Use original LLM assessment (no Google AI credibility analysis)
            metadata = {
                'title': submission.title,
                'subreddit': submission.subreddit.display_name,
                'score': submission.score,
                'upvote_ratio': f"{submission.upvote_ratio:.2f}",
                'num_comments': submission.num_comments,
                'age_hours': f"{(time.time() - submission.created_utc) / 3600:.1f}",
                'author': str(submission.author) if submission.author else "[deleted]",
                'has_external_content': str(scraped_content is not None),
                'content': (content or '')[:2000]
            }

            prompt = RISK_ASSESSMENT_PROMPT.format(**metadata)
            response = llm_wrapper.invoke(prompt)
            risk_level = getattr(response, 'content', str(response)).strip().upper()

            if risk_level in ['HIGH', 'MEDIUM', 'LOW']:
                logger.debug(f"LLM risk assessment: {risk_level} for post {getattr(submission,'id',None)}")
                return risk_level
            else:
                logger.warning(f"assess_risk_level: unexpected LLM response '{risk_level}' - defaulting to LOW")
                return 'LOW'
        except Exception as e:
            logger.warning(f"Risk assessment failed: {e} - defaulting to LOW")
            return 'LOW'

    def assess_risk_level_batch(self, batch_posts: List[BatchPostData], llm_wrapper) -> List[BatchRiskAssessment]:
        """Assess risk level for a batch of posts in a single API call"""
        try:
            if not batch_posts:
                return []
            
            # Create batch prompt for all posts
            batch_prompt = self._create_batch_risk_assessment_prompt(batch_posts)
            
            # Single API call for the entire batch
            response = llm_wrapper.invoke(batch_prompt)
            response_text = getattr(response, 'content', str(response)).strip()
            
            # Parse the batch response
            risk_assessments = self._parse_batch_risk_response(response_text, batch_posts)
            
            logger.info(f"Batch risk assessment completed for {len(batch_posts)} posts")
            return risk_assessments
            
        except Exception as e:
            logger.warning(f"Batch risk assessment failed: {e} - defaulting all to LOW")
            # Return LOW risk for all posts if batch fails
            return [BatchRiskAssessment(post_id=post.post_id, risk_level='LOW') for post in batch_posts]

    def _create_batch_risk_assessment_prompt(self, batch_posts: List[BatchPostData]) -> str:
        """Create a single prompt for batch risk assessment"""
        
        batch_prompt = """You are an expert misinformation detector. Analyze the following batch of Reddit posts and assign risk levels.

For EACH post, respond with exactly this format:
POST_ID: [post_id] | RISK: [HIGH/MEDIUM/LOW] | REASON: [brief reason]

Risk Level Guidelines:
- HIGH: Contains unverified claims, conspiracy theories, medical misinformation, or political manipulation
- MEDIUM: Potentially misleading, lacks sources, or emotional manipulation  
- LOW: Factual, well-sourced, or clearly opinion-based content

POSTS TO ANALYZE:

"""
        
        for i, post in enumerate(batch_posts, 1):
            batch_prompt += f"""
--- POST {i} (ID: {post.post_id}) ---
Title: {post.title}
Content: {post.content[:50000]}{'...' if len(post.content) > 500 else ''}
Subreddit: r/{post.subreddit}
Score: {post.score} | Comments: {post.num_comments} | Age: {post.age_hours:.1f}h
Author: {post.author}
Has External Content: {post.has_external_content}
{f'External Content: {post.scraped_content[:30000]}...' if post.scraped_content else ''}

"""

        batch_prompt += """
Now provide risk assessment for each post using the exact format:
POST_ID: [post_id] | RISK: [HIGH/MEDIUM/LOW] | REASON: [brief reason]
"""
        
        return batch_prompt

    def _parse_batch_risk_response(self, response_text: str, batch_posts: List[BatchPostData]) -> List[BatchRiskAssessment]:
        """Parse the LLM response for batch risk assessment"""
        assessments = []
        post_id_to_post = {post.post_id: post for post in batch_posts}
        
        # Parse each line looking for the expected format
        lines = response_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if 'POST_ID:' in line and '| RISK:' in line:
                try:
                    # Extract post_id, risk_level, and reason
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
        
        # Ensure we have assessment for all posts (fill missing with LOW)
        assessed_ids = {a.post_id for a in assessments}
        for post in batch_posts:
            if post.post_id not in assessed_ids:
                assessments.append(BatchRiskAssessment(post_id=post.post_id, risk_level='LOW'))
                logger.warning(f"Missing risk assessment for post {post.post_id}, defaulting to LOW")
        
        return assessments

    def _run(self, subreddit_name: str, limit: int = 20, sort_type: str = "new") -> str:
        try:
            subreddit = self._reddit.subreddit(subreddit_name)
            trending_posts = []
            processed_count = 0
            scraped_count = 0

            if sort_type == "new":
                submissions = subreddit.new(limit=limit)
            elif sort_type == "rising":
                submissions = subreddit.rising(limit=limit)
            elif sort_type == "hot":
                submissions = subreddit.hot(limit=limit)
            else:
                submissions = subreddit.new(limit=limit)

            # Debug: Log that we're about to iterate submissions
            logger.info(f"Starting to fetch submissions from r/{subreddit_name} (limit={limit}, sort={sort_type})")
            
            # First pass: collect all post data for batch processing
            candidate_posts = []
            submission_data = {}
            
            for submission in submissions:
                logger.debug(f"Processing submission: {submission.id} - {submission.title[:50]}...")
                processed_count += 1
                content, scraped_content, content_source = self.extract_post_content(submission)
                if scraped_content:
                    scraped_count += 1
                    
                velocity = self.calculate_velocity(submission.id, submission.score, submission.created_utc)
                engagement_rate = submission.num_comments / max(submission.score, 1)
                is_recent = (time.time() - submission.created_utc) < 86400
                meets_basic_score = submission.score >= (self._min_score_threshold * 0.3)
                
                # Debug: Log filtering criteria for first few posts
                if processed_count <= 3:
                    age_hours = (time.time() - submission.created_utc) / 3600
                    logger.info(f"Post {processed_count}: score={submission.score}, velocity={velocity:.1f}, age={age_hours:.1f}h, recent={is_recent}, basic_score_threshold={self._min_score_threshold * 0.3}")
                
                # Store submission data for later use
                submission_data[submission.id] = {
                    'submission': submission,
                    'content': content,
                    'scraped_content': scraped_content,
                    'content_source': content_source,
                    'velocity': velocity,
                    'engagement_rate': engagement_rate,
                    'is_recent': is_recent,
                    'meets_basic_score': meets_basic_score
                }
                
                # Debug: Log why posts are being filtered out
                if processed_count <= 5:
                    logger.info(f"Post {processed_count} filter check: recent={is_recent}, score={submission.score}>={self._min_score_threshold * 0.3}({meets_basic_score})")
                
                # Only add to batch assessment if it meets basic criteria
                if is_recent and meets_basic_score:
                    batch_post = BatchPostData(
                        post_id=submission.id,
                        title=submission.title,
                        content=content[:100000] if content else "",
                        scraped_content=scraped_content[:100000] if scraped_content else None,
                        subreddit=submission.subreddit.display_name,
                        score=submission.score,
                        upvote_ratio=submission.upvote_ratio,
                        num_comments=submission.num_comments,
                        age_hours=(time.time() - submission.created_utc) / 3600,
                        author=str(submission.author) if submission.author else "[deleted]",
                        has_external_content=scraped_content is not None
                    )
                    candidate_posts.append(batch_post)
                else:
                    if processed_count <= 5:
                        logger.info(f"Post {processed_count} FILTERED OUT: recent={is_recent}, meets_score={meets_basic_score}")

            # Batch risk assessment for all candidate posts
            logger.info(f"Performing batch risk assessment for {len(candidate_posts)} posts")
            risk_assessments = self.assess_risk_level_batch(candidate_posts, self._llm_wrapper)
            
            # Create risk assessment lookup
            risk_lookup = {assessment.post_id: assessment.risk_level for assessment in risk_assessments}
            
            # Second pass: apply risk levels and filtering
            for batch_post in candidate_posts:
                post_id = batch_post.post_id
                data = submission_data[post_id]
                submission = data['submission']
                
                # Get risk level from batch assessment
                risk_level = risk_lookup.get(post_id, 'LOW')
                
                # Apply threshold adjustments based on risk level
                threshold_multiplier = {"HIGH": 0.3, "MEDIUM": 0.5, "LOW": 1.0}
                if data['scraped_content']:
                    threshold_multiplier = {"HIGH": 0.2, "MEDIUM": 0.4, "LOW": 0.8}

                adjusted_threshold = self._velocity_threshold * threshold_multiplier[risk_level]
                meets_velocity = data['velocity'] >= adjusted_threshold
                meets_score = submission.score >= self._min_score_threshold
                
                if risk_level == 'HIGH' and data['scraped_content']:
                    meets_score = submission.score >= (self._min_score_threshold * 0.5)

                # Final filtering
                if (meets_velocity and meets_score and data['is_recent']) or (risk_level == 'HIGH' and data['scraped_content'] and data['is_recent']):
                    post_data = {
                        'post_id': submission.id,
                        'title': submission.title,
                        'content': data['content'][:1000] if data['content'] else "",
                        'scraped_content': data['scraped_content'][:1000] if data['scraped_content'] else None,
                        'content_source': data['content_source'],
                        'author': str(submission.author) if submission.author else "[deleted]",
                        'subreddit': submission.subreddit.display_name,
                        'url': submission.url,
                        'score': submission.score,
                        'upvote_ratio': submission.upvote_ratio,
                        'num_comments': submission.num_comments,
                        'created_utc': submission.created_utc,
                        'velocity_score': data['velocity'],
                        'engagement_rate': data['engagement_rate'],
                        'risk_level': risk_level,
                        'detected_at': __import__('datetime').datetime.now().isoformat(),
                        'permalink': f"https://reddit.com{submission.permalink}"
                    }
                    trending_posts.append(post_data)

            # Sort by combined score
            def combined_score(post):
                risk_multiplier = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
                return post['velocity_score'] * risk_multiplier[post['risk_level']]

            trending_posts.sort(key=combined_score, reverse=True)

            # Log batch processing efficiency 
            logger.info(f"Batch processing: assessed {len(candidate_posts)} posts in 1 API call vs {len(candidate_posts)} individual calls")
            logger.info(f"Scan summary: Scanned r/{subreddit_name} ({processed_count} posts), scraped {scraped_count} links, found {len(trending_posts)} trending posts")

            result = {
                'trending_posts': trending_posts,
                'scan_summary': f"Scanned r/{subreddit_name} ({processed_count} posts), scraped {scraped_count} links, found {len(trending_posts)} trending posts (batch processed)",
                'processed_count': processed_count,
                'scraped_count': scraped_count,
                'subreddit': subreddit_name,
                'batch_size': len(candidate_posts)
            }

            return json.dumps(result, indent=2)
        except Exception as e:
            logger.error(f"Batch processing failed for r/{subreddit_name}: {e}")
            return json.dumps({
                'trending_posts': [], 
                'scan_summary': f"Batch processing error: {str(e)}", 
                'processed_count': 0, 
                'scraped_count': 0, 
                'subreddit': subreddit_name
            }, indent=2)