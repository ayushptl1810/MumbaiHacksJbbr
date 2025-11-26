"""
PTI Fact Check Scraper Agent with Perplexity AI Integration

This agent scrapes PTI fact-check articles and uses Perplexity AI to:
1. Find original misinformation sources
2. Gather rich context about the claims
3. Store comprehensive data in MongoDB

Usage:
    python pti_scraper_agent_v3.py --max-items 5
"""

import os
import sys
import json
import uuid
import re
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, DuplicateKeyError
import requests

# Add parent directory to path to import the scraper
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pti_fact_check_scraper import scrape_pti_fact_checks, FactCheckEntry

load_dotenv()


class PTIScraperAgentV3:
    """Agent that scrapes PTI fact-checks and enriches them with Perplexity AI"""
    
    def __init__(self, connection_string: Optional[str] = None):
        self.connection_string = connection_string or os.getenv('MONGO_CONNECTION_STRING')
        
        if not self.connection_string:
            raise ValueError("MongoDB connection string is required.")
        
        self.perplexity_api_key = os.getenv('PERPLEXITY_API_KEY')
        if not self.perplexity_api_key:
            print("⚠️ WARNING: PERPLEXITY_API_KEY not found. Will use basic extraction.")
        
        self.client = None
        self.db = None
        self.collection = None
        
        self._connect()
    
    def _connect(self):
        """Establish MongoDB connection"""
        try:
            self.client = MongoClient(self.connection_string)
            self.client.admin.command('ping')
            
            self.db = self.client['aegis']
            self.collection = self.db['weekly_posts']
            
            # Create indexes for uniqueness and efficient querying
            self.collection.create_index("post_id", unique=True)
            self.collection.create_index("metadata.source_url")  # For fast duplicate checking
            
            print("✅ Successfully connected to MongoDB (aegis.weekly_posts)")
            
        except ConnectionFailure as e:
            print(f"❌ Failed to connect to MongoDB: {e}")
            raise
    
    def _generate_post_id(self, timestamp: datetime) -> str:
        return f"pti_post_{timestamp.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    
    def _generate_pipeline_run_id(self, timestamp: datetime) -> str:
        return f"pti_scraper_workflow_{timestamp.strftime('%Y%m%d_%H%M%S')}"
    
    def _extract_title_from_url(self, url: str) -> str:
        """Extract readable title from PTI URL"""
        try:
            parts = url.split('/fact-detail/')
            if len(parts) > 1:
                title_part = parts[1].split('/')[0]
                title = title_part.replace('-', ' ').title()
                title = title.replace('This ', '').replace('Is ', '')
                return f"Fact Check: {title[:100]}"
        except Exception as e:
            print(f"⚠️ Failed to extract title: {e}")
        
        return "PTI Fact Check Article"
    
    def _research_with_perplexity(self, heading: str, pti_url: str) -> Dict[str, Any]:
        """Use Perplexity AI to research the claim and find misinformation sources"""
        try:
            if not self.perplexity_api_key:
                print(f"   ⚠️ PERPLEXITY_API_KEY not found in environment")
                return self._get_fallback_details()
            
            # Extract the core claim from the heading
            # The heading format is usually: "Fact Check: [False Claim Description]"
            # We need to extract what people were CLAIMING (the false narrative)
            
            claim_query = heading.replace('Fact Check:', '').replace('Fact Check', '').strip()
            
            # Remove fact-check language to get the viral claim
            claim_query = claim_query.replace('No ', '').replace('Did Not ', '').replace('Did not ', '')
            claim_query = claim_query.replace('Falsely Shared', '').replace('False Claim', '').replace('Misleading', '')
            claim_query = claim_query.replace('Digitally Edited', '').replace('AI Generated', '').replace('Fake', '')
            claim_query = claim_query.strip()
            
            # Extract the actual claim subject (what the viral post was about)
            # E.g., "Video Haryana Youth Trapped In Russia" from "Fake Video Haryana Youth..."
            
            print(f"   🔍 Researching with Perplexity AI...")
            print(f"   📝 Searching for viral posts about: {claim_query[:80]}...")
            
            # Improved prompt for better paragraph flow
            prompt = f"""Research this misinformation case: "{claim_query}"

This is a FALSE claim that went viral on social media. Write a clear, well-structured fact-check report with proper paragraph flow.

**1. The False Claim:**
Write a flowing paragraph (3-4 sentences) explaining what the viral content claimed. Describe what people saw or heard in the misinformation. Make it read naturally.

**2. The Truth:**
Write a flowing paragraph (3-4 sentences) explaining what actually happened and why the claim is false. Include key facts and evidence. Make it read naturally.

**3. Full Context:**
Write 2-3 well-structured paragraphs providing complete context. Explain:
- Why this misinformation spread
- What the real story is
- How it was debunked
- Any important background details

Use proper transitions between sentences. Write in a journalistic style that flows naturally, not as bullet points or choppy sentences.

**4. Where It Spread:**
Mention which platforms this spread on (X/Twitter, Instagram, Facebook, YouTube). If you find usernames or accounts, mention them.

IMPORTANT: Write in complete, flowing paragraphs with natural transitions. Avoid choppy, disconnected sentences. Make it read like a professional news article."""
            
            headers = {
                'Authorization': f'Bearer {self.perplexity_api_key}',
                'Content-Type': 'application/json'
            }
            
            # Use sonar-pro for Perplexity Pro users
            payload = {
                'model': 'sonar-pro',  # Perplexity Pro model
                'messages': [
                    {
                        'role': 'system',
                        'content': 'You are a fact-checking assistant. Provide clear, concise explanations of misinformation cases. Focus on explaining what the false claim was and what the truth is.'
                    },
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ]
            }
            
            print(f"   🌐 Calling Perplexity API...")
            
            response = requests.post(
                'https://api.perplexity.ai/chat/completions',
                headers=headers,
                json=payload,
                timeout=45  # Increased timeout for online search
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Extract the AI response
            ai_response = result['choices'][0]['message']['content']
            
            print(f"   ✓ Received Perplexity response ({len(ai_response)} chars)")
            
            # Extract citations if available
            citations = result.get('citations', [])
            print(f"   ✓ Found {len(citations)} citations")
            
            # Parse the response to extract structured data
            # Split into paragraphs and clean up
            lines = ai_response.split('\n')
            paragraphs = []
            current_para = []
            
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#') and not line.startswith('**') and not line.startswith('-'):
                    current_para.append(line)
                elif current_para:
                    paragraphs.append(' '.join(current_para))
                    current_para = []
            
            if current_para:
                paragraphs.append(' '.join(current_para))
            
            # Filter out paragraphs that are just about finding URLs
            content_paragraphs = [
                p for p in paragraphs 
                if len(p) > 50 
                and not any(phrase in p.lower() for phrase in [
                    'direct url', 'exact url', 'post url', 'fact-checkers redact',
                    'urls are often', 'no exact url', 'format deduced'
                ])
            ]
            
            # Extract claim and truth
            claim_description = ""
            truth_explanation = ""
            
            for para in content_paragraphs:
                para_lower = para.lower()
                if any(word in para_lower for word in ['false', 'fake', 'claim', 'viral', 'video', 'post']):
                    if not claim_description and len(para) > 50:
                        claim_description = para
                elif any(word in para_lower for word in ['actually', 'truth', 'fact', 'real', 'correct']):
                    if not truth_explanation and len(para) > 50:
                        truth_explanation = para
            
            # Fallback
            if not claim_description and len(content_paragraphs) > 0:
                claim_description = content_paragraphs[0]
            if not truth_explanation and len(content_paragraphs) > 1:
                truth_explanation = content_paragraphs[1]
            
            # Create full context from clean paragraphs
            full_context = ' '.join(content_paragraphs[:3]) if content_paragraphs else ai_response[:2000]
            
            # Extract misinformation sources from response
            misinformation_sources = []
            
            # Scan response for valid URLs only
            urls_found = re.findall(r'https?://[^\s<>")\]]+', ai_response)
            
            for url in urls_found:
                # Only add if it's a valid social media post URL
                if self._is_valid_post_url(url):
                    if not any(s.get('url') == url for s in misinformation_sources):
                        misinformation_sources.append({
                            'platform': self._identify_platform(url),
                            'url': url,
                            'description': 'Viral post spreading misinformation'
                        })
            
            # If no valid URLs found, look for platform mentions and usernames
            if len(misinformation_sources) == 0:
                # Look for mentions of platforms
                platform_mentions = []
                if 'twitter' in ai_response.lower() or 'x.com' in ai_response.lower():
                    platform_mentions.append('X (Twitter)')
                if 'instagram' in ai_response.lower():
                    platform_mentions.append('Instagram')
                if 'facebook' in ai_response.lower():
                    platform_mentions.append('Facebook')
                if 'youtube' in ai_response.lower():
                    platform_mentions.append('YouTube')
                
                # Add platform mentions as sources (without URLs)
                for platform in platform_mentions[:2]:  # Limit to 2
                    misinformation_sources.append({
                        'platform': platform,
                        'url': '',  # No specific URL
                        'description': f'Misinformation spread on {platform}'
                    })
            
            # Try to find actual social media posts using Google Search
            google_posts = self._search_social_media_posts(claim_query)
            
            # Merge Google results with Perplexity results
            for post in google_posts:
                # Check if not already in list
                if not any(s.get('url') == post['url'] for s in misinformation_sources if s.get('url')):
                    misinformation_sources.append(post)
            
            print(f"   ✓ Total misinformation sources: {len(misinformation_sources)}")
            
            # Create verdict statement
            verdict_statement = f"This claim is false. {truth_explanation[:200]}" if truth_explanation else "This claim has been fact-checked and found to be false."
            
            return {
                'claim_text': claim_description[:500] if claim_description else claim_query,
                'full_explanation': full_context[:2000],
                'fact_check_summary': truth_explanation[:500] if truth_explanation else full_context[:500],
                'verdict_statement': verdict_statement[:300],
                'misinformation_sources': misinformation_sources[:5]
            }
            
        except requests.exceptions.RequestException as e:
            print(f"    Perplexity API request failed: {e}")
            print(f"   ℹ Check your PERPLEXITY_API_KEY and internet connection")
            return self._get_fallback_details()
        except Exception as e:
            print(f"    Perplexity research failed: {e}")
            import traceback
            traceback.print_exc()
            return self._get_fallback_details()
    
    def _search_social_media_posts(self, claim_query: str) -> List[Dict[str, Any]]:
        """Use Google Search to find videos and articles about the claim"""
        try:
            serp_api_key = os.getenv('SERP_API_KEY')
            
            if not serp_api_key:
                print(f"   ⚠️ SERP_API_KEY not found, skipping Google search")
                return []
            
            print(f"   🔎 Searching Google Videos for: {claim_query[:60]}...")
            
            all_sources = []
            
            # 1. Search Google Videos
            video_params = {
                'api_key': serp_api_key,
                'engine': 'google_videos',
                'q': claim_query,
                'num': 5,
                'gl': 'in',
                'hl': 'en'
            }
            
            try:
                response = requests.get('https://serpapi.com/search', params=video_params, timeout=10)
                response.raise_for_status()
                results = response.json()
                
                video_results = results.get('video_results', [])
                
                for video in video_results[:3]:  # Top 3 videos
                    url = video.get('link', '')
                    title = video.get('title', '')
                    
                    if url:
                        # Determine platform
                        platform = 'YouTube' if 'youtube.com' in url or 'youtu.be' in url else 'Video'
                        
                        all_sources.append({
                            'platform': platform,
                            'url': url,
                            'description': title[:100] if title else 'Viral video'
                        })
                
                print(f"   ✓ Found {len(video_results[:3])} videos")
                
            except Exception as e:
                print(f"   ⚠️ Video search failed: {e}")
            
            # 2. Search regular Google for articles/posts
            article_params = {
                'api_key': serp_api_key,
                'engine': 'google',
                'q': claim_query,
                'num': 5,
                'gl': 'in',
                'hl': 'en'
            }
            
            try:
                response = requests.get('https://serpapi.com/search', params=article_params, timeout=10)
                response.raise_for_status()
                results = response.json()
                
                organic_results = results.get('organic_results', [])
                
                for result in organic_results[:2]:  # Top 2 articles
                    url = result.get('link', '')
                    title = result.get('title', '')
                    
                    # Skip fact-check sites
                    if any(word in title.lower() for word in ['fact check', 'factcheck', 'debunk']):
                        continue
                    
                    if url:
                        # Determine platform from URL
                        if 'x.com' in url or 'twitter.com' in url:
                            platform = 'X (Twitter)'
                        elif 'facebook.com' in url:
                            platform = 'Facebook'
                        elif 'instagram.com' in url:
                            platform = 'Instagram'
                        else:
                            platform = 'Article'
                        
                        all_sources.append({
                            'platform': platform,
                            'url': url,
                            'description': title[:100] if title else 'Source'
                        })
                
                print(f"   ✓ Found {len(all_sources)} total sources")
                
            except Exception as e:
                print(f"   ⚠️ Article search failed: {e}")
            
            return all_sources[:5]  # Return top 5 sources
            
        except Exception as e:
            print(f"   ⚠️ Google search failed: {e}")
            return []
    
    
    def _is_valid_post_url(self, url: str) -> bool:
        """Check if URL is an actual post/video URL, not a generic platform link"""
        url_lower = url.lower()
        
        # X/Twitter - must have /status/ for actual tweets
        if 'twitter.com' in url_lower or 'x.com' in url_lower:
            return '/status/' in url_lower
        
        # Facebook - must have /posts/ or /videos/ or /photo
        elif 'facebook.com' in url_lower:
            return any(pattern in url_lower for pattern in ['/posts/', '/videos/', '/photo', '/permalink/'])
        
        # Instagram - must have /p/ or /reel/
        elif 'instagram.com' in url_lower:
            return '/p/' in url_lower or '/reel/' in url_lower
        
        # YouTube - must have watch?v= or /shorts/
        elif 'youtube.com' in url_lower or 'youtu.be' in url_lower:
            return 'watch?v=' in url_lower or '/shorts/' in url_lower or 'youtu.be/' in url_lower
        
        return False
    
    def _identify_platform(self, url: str) -> str:
        """Identify social media platform from URL"""
        url_lower = url.lower()
        if 'twitter.com' in url_lower or 'x.com' in url_lower:
            return 'X (Twitter)'
        elif 'facebook.com' in url_lower:
            return 'Facebook'
        elif 'instagram.com' in url_lower:
            return 'Instagram'
        elif 'youtube.com' in url_lower or 'youtu.be' in url_lower:
            return 'YouTube'
        else:
            return 'Social Media'
    
    def _get_fallback_details(self) -> Dict[str, Any]:
        """Return fallback details when research fails"""
        return {
            'claim_text': "",
            'full_explanation': "",
            'fact_check_summary': "",
            'verdict_statement': "",
            'misinformation_sources': []
        }
    
    def _article_exists(self, pti_url: str) -> bool:
        """Check if article already exists in database by PTI URL"""
        try:
            existing = self.collection.find_one({
                "metadata.source_url": pti_url
            })
            return existing is not None
        except Exception as e:
            print(f"   ⚠️ Error checking for duplicates: {e}")
            return False
    
    def _convert_entry_to_document(self, entry: FactCheckEntry, pipeline_run_id: str) -> Dict[str, Any]:
        """Convert a FactCheckEntry to MongoDB document format"""
        now = datetime.now(timezone.utc)
        post_id = self._generate_post_id(now)
        
        # Extract title
        if entry.title and entry.title.strip():
            heading = entry.title
        else:
            heading = self._extract_title_from_url(entry.url)
        
        # Research with Perplexity AI
        print(f"   📄 Researching: {heading[:60]}...")
        article_details = self._research_with_perplexity(heading, entry.url)
        
        # Use researched data
        claim_text = article_details['claim_text'] or heading
        body = article_details['full_explanation'] or entry.summary or \
               f"PTI Fact Check article. Visit the source for full details: {entry.url}"
        summary = article_details['fact_check_summary'] or \
                  (entry.summary[:300] if entry.summary else f"PTI fact-check: {heading[:200]}")
        
        misinformation_sources = article_details['misinformation_sources']
        
        # Create document
        document = {
            "_id": str(uuid.uuid4()),
            "post_id": post_id,
            "pipeline_run_id": pipeline_run_id,
            "stored_at": now,
            "verification_date": now.isoformat(),
            
            "claim": {
                "text": claim_text[:500],
                "verdict": "fact_checked",
                "verified": True,
                "verdict_statement": article_details['verdict_statement']
            },
            
            "post_content": {
                "heading": heading,
                "body": body[:2000],
                "summary": summary[:500],
                "full_article_url": entry.url
            },
            
            "sources": {
                "misinformation_sources": misinformation_sources,
                "confirmation_sources": [
                    {
                        "title": heading,
                        "url": entry.url,
                        "source": "PTI Fact Check",
                        "published_at": entry.published_at
                    }
                ],
                "total_sources": 1 + len(misinformation_sources),
                "confidence_percentage": 95
            },
            
            "metadata": {
                "source": "PTI Fact Check",
                "source_url": entry.url,
                "published_at": entry.published_at,
                "image_url": entry.image_url,
                "scraper_agent": "pti_scraper_agent",
                "scraper_version": "3.0.0",  # Perplexity AI integration
                "has_misinformation_sources": len(misinformation_sources) > 0,
                "research_method": "perplexity_ai"
            }
        }
        
        return document
    
    def scrape_and_store(self, max_items: int = 10) -> Dict[str, Any]:
        """Scrape PTI fact-checks and store them in MongoDB"""
        print(f"🔍 Starting PTI scraper agent with Perplexity AI (max_items={max_items})...")
        
        try:
            print("📰 Scraping PTI fact-check website...")
            entries = scrape_pti_fact_checks(max_items=max_items)
            
            if not entries:
                print("⚠️ No entries found")
                return {
                    "success": True,
                    "total_scraped": 0,
                    "total_stored": 0,
                    "total_duplicates": 0,
                    "message": "No new fact-checks found"
                }
            
            print(f"✅ Scraped {len(entries)} fact-check entries")
            
            pipeline_run_id = self._generate_pipeline_run_id(datetime.now(timezone.utc))
            
            stored_count = 0
            duplicate_count = 0
            errors = []
            
            for i, entry in enumerate(entries, 1):
                try:
                    # Check if article already exists BEFORE making API calls
                    if self._article_exists(entry.url):
                        duplicate_count += 1
                        print(f"⏭️  [{i}/{len(entries)}] Skipping duplicate: {entry.title[:60]}...")
                        continue
                    
                    # Only process if it's new
                    document = self._convert_entry_to_document(entry, pipeline_run_id)
                    
                    # Insert into MongoDB
                    self.collection.insert_one(document)
                    stored_count += 1
                    print(f"✅ [{i}/{len(entries)}] Stored: {document['post_content']['heading'][:60]}...")
                    
                except DuplicateKeyError:
                    # This shouldn't happen now since we check first, but keep as safety net
                    duplicate_count += 1
                    print(f"⚠️ [{i}/{len(entries)}] Duplicate (post_id collision)")
                    
                except Exception as e:
                    errors.append(f"Error storing entry {i}: {str(e)}")
                    print(f"❌ [{i}/{len(entries)}] Error: {str(e)}")
            
            summary = {
                "success": True,
                "total_scraped": len(entries),
                "total_stored": stored_count,
                "total_duplicates": duplicate_count,
                "total_errors": len(errors),
                "pipeline_run_id": pipeline_run_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            if errors:
                summary["errors"] = errors
            
            print(f"\n📊 Summary:")
            print(f"   Scraped: {summary['total_scraped']}")
            print(f"   Stored: {summary['total_stored']}")
            print(f"   Duplicates: {summary['total_duplicates']}")
            print(f"   Errors: {summary['total_errors']}")
            
            return summary
            
        except Exception as e:
            print(f"❌ Scraper agent failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    def get_recent_posts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent posts from weekly_posts collection"""
        try:
            posts = list(self.collection.find().sort("stored_at", -1).limit(limit))
            
            for post in posts:
                if '_id' in post:
                    post['_id'] = str(post['_id'])
            
            return posts
            
        except Exception as e:
            print(f"❌ Failed to get recent posts: {e}")
            return []
    
    def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            print("🔌 MongoDB connection closed")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="PTI Scraper with Perplexity AI")
    parser.add_argument("--max-items", type=int, default=10, help="Max items to scrape")
    parser.add_argument("--show-recent", type=int, metavar="N", help="Show N recent posts")
    
    args = parser.parse_args()
    
    try:
        agent = PTIScraperAgentV3()
        
        if args.show_recent:
            print(f"📋 Fetching {args.show_recent} most recent posts...")
            posts = agent.get_recent_posts(limit=args.show_recent)
            print(json.dumps(posts, indent=2, default=str))
        else:
            summary = agent.scrape_and_store(max_items=args.max_items)
            
            summary_file = f"pti_scraper_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, default=str)
            print(f"\n💾 Summary saved to: {summary_file}")
        
        agent.close()
        return 0
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
