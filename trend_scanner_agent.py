"""Launcher for trend_scanner package with Google Agents SDK integration"""

import os
import logging
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

LOG_FILE = os.path.join(os.path.dirname(__file__), 'trend_scanner.log.txt')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8'), logging.StreamHandler()]
)

logger = logging.getLogger(__name__)


import os
import sys
from typing import List

# Predefined list of subreddits to scan
TARGET_SUBREDDITS = [
    'NoFilterNews',
    'badscience',
    'skeptic',
    'conspiracytheories'
]

# Predefined list of Threads profiles to scan (without @ symbol)
TARGET_THREADS_PROFILES = [
    # Add Threads usernames here, e.g.:
    'globaltimes_news',
    'trumplovernews',
]

# Enable/disable Threads scanning
THREADS_ENABLED = bool(TARGET_THREADS_PROFILES)  # Automatically enabled if profiles are configured

def main_one_scan() -> dict:
    """Run a single scan using Google Agents orchestration (no CrewAI needed)"""
    from trend_scanner.google_agents import TrendScannerOrchestrator

    REDDIT_CONFIG = {
        'client_id': os.getenv('REDDIT_CLIENT_ID', 'your_reddit_client_id'),
        'client_secret': os.getenv('REDDIT_CLIENT_SECRET', 'your_reddit_client_secret'),
        'user_agent': 'ProjectAegis-EnhancedTrendScanner/2.0-GoogleAgents'
    }

    try:
        print("🚀 Initializing Trend Scanner with Google Agents orchestration...")
        
        # Use the new TrendScannerOrchestrator with multi-platform support
        orchestrator = TrendScannerOrchestrator(
            REDDIT_CONFIG,
            threads_enabled=THREADS_ENABLED
        )
        
        # Configure Reddit targets
        print(f"🎯 Target subreddits: {', '.join([f'r/{s}' for s in TARGET_SUBREDDITS])}")
        orchestrator.set_target_subreddits(TARGET_SUBREDDITS)
        
        # Configure Threads targets if enabled
        if THREADS_ENABLED and TARGET_THREADS_PROFILES:
            print(f"🎯 Target Threads profiles: {', '.join([f'@{p}' for p in TARGET_THREADS_PROFILES])}")
            orchestrator.set_target_threads_profiles(TARGET_THREADS_PROFILES)
        
        platforms = ['Reddit']
        if THREADS_ENABLED:
            platforms.append('Threads')
        print(f"🔍 Running comprehensive trend analysis across {' + '.join(platforms)} with Google Agents...")
        results = orchestrator.scan_trending_content()
        
        # Get all posts for batch processing (multi-platform)
        all_posts = results.get('trending_posts', [])
        reddit_posts = results.get('reddit_posts', [])
        threads_posts = results.get('threads_posts', [])
        
        if not all_posts:
            final_output = {
                "timestamp": results.get('timestamp', datetime.now().isoformat()),
                "total_posts": 0,
                "posts_by_platform": {
                    "reddit": 0,
                    "threads": 0
                },
                "posts": []
            }
            print(json.dumps(final_output, indent=2, ensure_ascii=False))
            return final_output
        
        print(f"📊 Found {len(all_posts)} total posts (Reddit: {len(reddit_posts)}, Threads: {len(threads_posts)})")
        
        # Prepare posts data for Gemini batch processing (multi-platform)
        posts_for_gemini = []
        for i, post in enumerate(all_posts):
            platform = post.get('source_platform', post.get('platform', 'reddit'))
            post_data = {
                "post_id": i + 1,
                "title": post.get('title', ''),
                "content": post.get('selftext', post.get('content', '')),
                "scraped_content": post.get('scraped_content', ''),
                "subreddit": post.get('subreddit', ''),
                "url": post.get('url', ''),
                "permalink": post.get('permalink', ''),
                "score": post.get('score', 0),
                "platform": platform
            }
            posts_for_gemini.append(post_data)
        
        # Send to Gemini for batch summarization
        try:
            import google.generativeai as genai
            
            # Configure Gemini
            gemini_api_key = os.getenv('GEMINI_API_KEY')
            if gemini_api_key:
                genai.configure(api_key=gemini_api_key)
                model = genai.GenerativeModel('gemini-2.5-flash')
                
                # Create batch prompt for Gemini
                batch_prompt = f"""
You are a content analyzer. Analyze these {len(posts_for_gemini)} Reddit posts and return a JSON array with summaries and claims.

For each post, extract:
1. A clear, simple claim in plain English (what the post is asserting)
2. A comprehensive summary combining the post content and any scraped external content

Posts data:
{json.dumps(posts_for_gemini, indent=2)}

Return ONLY a JSON array in this exact format:
[
  {{
    "post_id": 1,
    "claim": "Simple claim in plain English",
    "summary": "Comprehensive summary combining all content"
  }},
  {{
    "post_id": 2, 
    "claim": "Another claim in plain English",
    "summary": "Another comprehensive summary"
  }}
]

Requirements:
- Keep claims simple and factual
- Make summaries detailed but concise
- Include key information from both post content and scraped content
- Return ONLY the JSON array, no other text
"""
                
                response = model.generate_content(batch_prompt)
                response_text = response.text.strip()
                
                # Clean up response if needed
                if response_text.startswith('```json'):
                    response_text = response_text.replace('```json', '').replace('```', '').strip()
                elif response_text.startswith('```'):
                    response_text = response_text.replace('```', '').strip()
                
                # Parse Gemini response
                gemini_results = json.loads(response_text)
                
                # Build final output using Gemini results
                output_posts = []
                for post, gemini_data in zip(all_posts, gemini_results):
                    # Determine platform
                    platform = post.get('source_platform', post.get('platform', 'reddit'))
                    
                    # Build post link based on platform
                    if platform == 'threads':
                        # Threads post link
                        if post.get('url'):
                            post_link = post['url']
                        else:
                            # Fallback to Threads profile
                            username = post.get('author', post.get('subreddit', '').replace('threads/@', ''))
                            post_link = f"https://www.threads.net/@{username}"
                    else:
                        # Reddit post link
                        if post.get('permalink'):
                            post_link = f"https://reddit.com{post['permalink']}"
                        elif post.get('url') and 'reddit.com' in post.get('url', ''):
                            post_link = post['url']
                        else:
                            post_link = f"https://reddit.com/r/{post.get('subreddit', 'unknown')}"
                    
                    formatted_post = {
                        "claim": gemini_data.get('claim', post.get('title', 'No claim identified')),
                        "summary": gemini_data.get('summary', 'No summary available'),
                        "platform": platform,
                        "Post_link": post_link
                    }
                    
                    output_posts.append(formatted_post)
                
            else:
                # Fallback if no Gemini API key
                logger.warning("No Gemini API key found, using basic summarization")
                output_posts = []
                
                for post in all_posts:
                    # Determine platform
                    platform = post.get('source_platform', post.get('platform', 'reddit'))
                    
                    # Basic fallback summarization
                    summary_parts = []
                    if post.get('title'):
                        summary_parts.append(f"Title: {post['title']}")
                    if post.get('selftext') and post['selftext'].strip():
                        summary_parts.append(f"Post Content: {post['selftext']}")
                    elif post.get('content') and post['content'].strip():
                        summary_parts.append(f"Post Content: {post['content']}")
                    if post.get('scraped_content'):
                        summary_parts.append(f"External Content: {post['scraped_content']}")
                    
                    claim = post.get('title', 'No specific claim identified')
                    summary = " | ".join(summary_parts) if summary_parts else "No content available"
                    
                    # Build post link based on platform
                    if platform == 'threads':
                        if post.get('url'):
                            post_link = post['url']
                        else:
                            username = post.get('author', post.get('subreddit', '').replace('threads/@', ''))
                            post_link = f"https://www.threads.net/@{username}"
                    else:
                        if post.get('permalink'):
                            post_link = f"https://reddit.com{post['permalink']}"
                        elif post.get('url') and 'reddit.com' in post.get('url', ''):
                            post_link = post['url']
                        else:
                            post_link = f"https://reddit.com/r/{post.get('subreddit', 'unknown')}"
                    
                    formatted_post = {
                        "claim": claim,
                        "summary": summary,
                        "platform": platform,
                        "Post_link": post_link
                    }
                    
                    output_posts.append(formatted_post)
        
        except Exception as e:
            logger.error(f"Error in Gemini batch processing: {e}")
            # Fallback to basic processing
            output_posts = []
            
            for post in all_posts:
                platform = post.get('source_platform', post.get('platform', 'reddit'))
                
                summary_parts = []
                if post.get('title'):
                    summary_parts.append(f"Title: {post['title']}")
                if post.get('selftext') and post['selftext'].strip():
                    summary_parts.append(f"Post Content: {post['selftext']}")
                elif post.get('content') and post['content'].strip():
                    summary_parts.append(f"Post Content: {post['content']}")
                if post.get('scraped_content'):
                    summary_parts.append(f"External Content: {post['scraped_content']}")
                
                claim = post.get('title', 'No specific claim identified')
                summary = " | ".join(summary_parts) if summary_parts else "No content available"
                
                # Build post link based on platform
                if platform == 'threads':
                    if post.get('url'):
                        post_link = post['url']
                    else:
                        username = post.get('author', post.get('subreddit', '').replace('threads/@', ''))
                        post_link = f"https://www.threads.net/@{username}"
                else:
                    if post.get('permalink'):
                        post_link = f"https://reddit.com{post['permalink']}"
                    elif post.get('url') and 'reddit.com' in post.get('url', ''):
                        post_link = post['url']
                    else:
                        post_link = f"https://reddit.com/r/{post.get('subreddit', 'unknown')}"
                
                formatted_post = {
                    "claim": claim,
                    "summary": summary,
                    "platform": platform,
                    "Post_link": post_link
                }
                
                output_posts.append(formatted_post)
        
        # Output as single JSON with multi-platform stats
        reddit_count = len([p for p in output_posts if p.get('platform') == 'reddit'])
        threads_count = len([p for p in output_posts if p.get('platform') == 'threads'])
        
        final_output = {
            "timestamp": results.get('timestamp', datetime.now().isoformat()),
            "total_posts": len(output_posts),
            "posts_by_platform": {
                "reddit": reddit_count,
                "threads": threads_count
            },
            "platforms": results.get('platforms', ['reddit']),
            "posts": output_posts
        }
        
        print(json.dumps(final_output, indent=2, ensure_ascii=False))
        
        return final_output
        
    except Exception as e:
        logger.error(f"Error running enhanced scan: {e}")
        print(f"❌ Error: {e}")


def show_installation_requirements():
    """Display installation and setup requirements"""
    print("""
🔧 INSTALLATION REQUIREMENTS FOR GOOGLE AGENTS ORCHESTRATION:

1. Install packages:
   pip install -r requirements.txt

2. Required API Keys:
   - Google API Key (for Google Agents SDK)
   - Gemini API Key (for LLM capabilities)  
   - Reddit API credentials

3. Environment Variables (.env file):
   GOOGLE_API_KEY=your_google_api_key
   GEMINI_API_KEY=your_gemini_api_key
   REDDIT_CLIENT_ID=your_reddit_client_id
   REDDIT_CLIENT_SECRET=your_reddit_client_secret

4. Key Features:
   ✅ Google Agents orchestration (replaces CrewAI)
   ✅ Multi-platform scanning (Reddit + Threads)
   ✅ Parallel execution across platforms
   ✅ Source credibility analysis with Gemini AI
   ✅ Cross-platform trend detection
   ✅ Multi-agent workflow coordination
   ✅ Enhanced misinformation pattern detection

5. Usage:
   python trend_scanner_agent.py

6. Scan Targets:
   Reddit Subreddits: {', '.join([f'r/{s}' for s in TARGET_SUBREDDITS])}
   Threads Profiles: {', '.join([f'@{p}' for p in TARGET_THREADS_PROFILES]) if TARGET_THREADS_PROFILES else 'None configured'}
   
   To modify:
   - Edit TARGET_SUBREDDITS for Reddit
   - Edit TARGET_THREADS_PROFILES for Threads

📚 See trend_scanner/README.md for detailed documentation.
📦 All functionality now in google_agents.py - no CrewAI dependencies!
    """)


if __name__ == '__main__':
    show_installation_requirements()
    
    print(f"📋 Scanning {len(TARGET_SUBREDDITS)} subreddits: {', '.join([f'r/{s}' for s in TARGET_SUBREDDITS])}")
    if TARGET_THREADS_PROFILES:
        print(f"📋 Scanning {len(TARGET_THREADS_PROFILES)} Threads profiles: {', '.join([f'@{p}' for p in TARGET_THREADS_PROFILES])}")
    else:
        print("ℹ️  Threads scanning disabled (no profiles configured)")
    
    # Check if API keys are configured
    if not os.getenv('GOOGLE_API_KEY') and not os.getenv('GEMINI_API_KEY'):
        print("⚠️  No Google API key found. Please configure GOOGLE_API_KEY or GEMINI_API_KEY")
        print("The system will attempt to run with fallback analysis.")
    
    if not os.getenv('REDDIT_CLIENT_ID'):
        print("⚠️  No Reddit API credentials found. Please configure REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET")
    
    main_one_scan()