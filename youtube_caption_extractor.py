"""YouTube Caption Extractor using yt-dlp with cookie authentication"""

import yt_dlp
import json
import os
from typing import Dict, List, Optional, Any
from datetime import datetime


class YouTubeCaptionExtractor:
    """Extract captions from YouTube videos using yt-dlp with cookie support"""
    
    def __init__(self, cookies_file: str = "youtube_cookies.txt"):
        """
        Initialize the caption extractor
        
        Args:
            cookies_file: Path to Netscape format cookies file
        """
        self.cookies_file = cookies_file
        
        # Check if cookies file exists
        if not os.path.exists(cookies_file):
            print(f"⚠️  Warning: Cookies file '{cookies_file}' not found!")
            print("Some videos may not be accessible without authentication.")
    
    def extract_captions(self, video_url: str, language: str = 'en') -> Dict[str, Any]:
        """
        Extract captions from a YouTube video
        
        Args:
            video_url: YouTube video URL or video ID
            language: Preferred caption language code (default: 'en' for English)
        
        Returns:
            Dictionary containing captions and metadata
        """
        try:
            # yt-dlp options
            ydl_opts = {
                'skip_download': True,  # Don't download video
                'writesubtitles': True,  # Write subtitle file
                'writeautomaticsub': True,  # Write automatic captions if no manual subs
                'subtitleslangs': [language],  # Preferred languages
                'subtitlesformat': 'json3',  # JSON format with timestamps
                'quiet': True,  # Suppress output
                'no_warnings': True,
            }
            
            # Add cookies if file exists
            if os.path.exists(self.cookies_file):
                ydl_opts['cookiefile'] = self.cookies_file
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Extract info
                info = ydl.extract_info(video_url, download=False)
                
                # Get video metadata
                video_data = {
                    'video_id': info.get('id'),
                    'title': info.get('title'),
                    'description': info.get('description'),
                    'uploader': info.get('uploader'),
                    'upload_date': info.get('upload_date'),
                    'duration': info.get('duration'),
                    'view_count': info.get('view_count'),
                    'like_count': info.get('like_count'),
                    'url': video_url,
                    'extracted_at': datetime.now().isoformat()
                }
                
                # Extract captions
                captions_text = []
                captions_with_timestamps = []
                
                # Try to get subtitles
                subtitles = info.get('subtitles', {})
                automatic_captions = info.get('automatic_captions', {})
                
                # Prefer manual subtitles, fall back to automatic
                available_subs = subtitles if subtitles else automatic_captions
                
                if language in available_subs:
                    # Get the JSON3 format captions
                    caption_data = available_subs[language]
                    
                    # Find json3 format
                    json3_caption = None
                    for caption_format in caption_data:
                        if caption_format.get('ext') == 'json3':
                            json3_caption = caption_format
                            break
                    
                    if json3_caption:
                        # Download caption data
                        caption_url = json3_caption.get('url')
                        if caption_url:
                            import urllib.request
                            with urllib.request.urlopen(caption_url) as response:
                                caption_json = json.loads(response.read().decode('utf-8'))
                                
                                # Parse events
                                events = caption_json.get('events', [])
                                for event in events:
                                    if 'segs' in event:
                                        timestamp = event.get('tStartMs', 0) / 1000.0  # Convert to seconds
                                        text_segments = [seg.get('utf8', '') for seg in event['segs']]
                                        text = ''.join(text_segments).strip()
                                        
                                        if text:
                                            captions_text.append(text)
                                            captions_with_timestamps.append({
                                                'timestamp': timestamp,
                                                'text': text
                                            })
                
                # Compile full caption text
                full_text = ' '.join(captions_text)
                
                result = {
                    'success': True,
                    'video': video_data,
                    'captions': {
                        'full_text': full_text,
                        'with_timestamps': captions_with_timestamps,
                        'language': language,
                        'caption_type': 'manual' if language in subtitles else 'automatic',
                        'word_count': len(full_text.split()),
                        'line_count': len(captions_with_timestamps)
                    }
                }
                
                return result
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'video_url': video_url
            }
    
    def extract_available_languages(self, video_url: str) -> Dict[str, Any]:
        """
        Get list of available caption languages for a video
        
        Args:
            video_url: YouTube video URL or video ID
        
        Returns:
            Dictionary with available languages
        """
        try:
            ydl_opts = {
                'skip_download': True,
                'quiet': True,
                'no_warnings': True,
            }
            
            if os.path.exists(self.cookies_file):
                ydl_opts['cookiefile'] = self.cookies_file
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=False)
                
                manual_subs = list(info.get('subtitles', {}).keys())
                auto_subs = list(info.get('automatic_captions', {}).keys())
                
                return {
                    'success': True,
                    'video_id': info.get('id'),
                    'title': info.get('title'),
                    'manual_subtitles': manual_subs,
                    'automatic_captions': auto_subs,
                    'all_languages': list(set(manual_subs + auto_subs))
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def batch_extract(self, video_urls: List[str], language: str = 'en') -> List[Dict[str, Any]]:
        """
        Extract captions from multiple videos
        
        Args:
            video_urls: List of YouTube video URLs
            language: Preferred caption language
        
        Returns:
            List of caption extraction results
        """
        results = []
        for url in video_urls:
            print(f"Extracting captions from: {url}")
            result = self.extract_captions(url, language)
            results.append(result)
        return results


def main():
    """Test the caption extractor"""
    print("🎬 YouTube Caption Extractor")
    print("=" * 50)
    
    # Initialize extractor
    extractor = YouTubeCaptionExtractor(cookies_file="youtube_cookies.txt")
    
    # Test video URL
    test_url = input("\nEnter YouTube video URL: ").strip()
    
    if not test_url:
        test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Default test video
        print(f"Using default test video: {test_url}")
    
    # Check available languages
    print("\n📝 Checking available caption languages...")
    lang_info = extractor.extract_available_languages(test_url)
    
    if lang_info['success']:
        print(f"\nVideo: {lang_info['title']}")
        print(f"Manual Subtitles: {', '.join(lang_info['manual_subtitles']) or 'None'}")
        print(f"Auto Captions: {', '.join(lang_info['automatic_captions']) or 'None'}")
    
    # Extract captions
    print("\n🔍 Extracting captions...")
    result = extractor.extract_captions(test_url, language='en')
    
    if result['success']:
        print("\n✅ Caption extraction successful!")
        print(f"\nVideo: {result['video']['title']}")
        print(f"Uploader: {result['video']['uploader']}")
        print(f"Duration: {result['video']['duration']}s")
        print(f"\nCaption Type: {result['captions']['caption_type']}")
        print(f"Language: {result['captions']['language']}")
        print(f"Word Count: {result['captions']['word_count']}")
        print(f"Line Count: {result['captions']['line_count']}")
        
        print("\n--- First 500 characters of captions ---")
        print(result['captions']['full_text'][:500])
        
        print("\n--- First 5 timestamped captions ---")
        for caption in result['captions']['with_timestamps'][:5]:
            print(f"[{caption['timestamp']:.2f}s] {caption['text']}")
        
        # Save to file
        output_file = f"captions_{result['video']['video_id']}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Full results saved to: {output_file}")
        
    else:
        print(f"\n❌ Caption extraction failed: {result['error']}")


if __name__ == "__main__":
    main()
