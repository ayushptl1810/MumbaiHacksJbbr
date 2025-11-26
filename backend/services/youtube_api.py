import os
import requests
from typing import Dict, Any, Optional
from config import config

class YouTubeDataAPI:
    """
    YouTube Data API v3 integration for video verification
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize YouTube Data API client
        
        Args:
            api_key: Google API key. If None, will try to get from environment
        """
        self.api_key = api_key or config.GOOGLE_API_KEY
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY environment variable or api_key parameter is required")
        
        self.base_url = "https://www.googleapis.com/youtube/v3"
    
    def extract_video_id(self, url: str) -> Optional[str]:
        """
        Extract video ID from YouTube URL
        
        Args:
            url: YouTube URL (various formats supported)
            
        Returns:
            Video ID or None if not found
        """
        import re
        
        # YouTube URL patterns
        patterns = [
            r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([a-zA-Z0-9_-]{11})',
            r'youtube\.com\/v\/([a-zA-Z0-9_-]{11})',
            r'youtube\.com\/shorts\/([a-zA-Z0-9_-]{11})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        return None
    
    def get_video_info(self, video_id: str) -> Dict[str, Any]:
        """
        Get video information from YouTube Data API
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            Dictionary with video information
        """
        try:
            url = f"{self.base_url}/videos"
            params = {
                'key': self.api_key,
                'id': video_id,
                'part': 'snippet,statistics,contentDetails'
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if not data.get('items'):
                return {
                    'success': False,
                    'error': 'Video not found or not accessible'
                }
            
            video = data['items'][0]
            snippet = video.get('snippet', {})
            statistics = video.get('statistics', {})
            content_details = video.get('contentDetails', {})
            
            return {
                'success': True,
                'video_id': video_id,
                'title': snippet.get('title', 'Unknown Title'),
                'description': snippet.get('description', ''),
                'channel_title': snippet.get('channelTitle', 'Unknown Channel'),
                'published_at': snippet.get('publishedAt', ''),
                'duration': content_details.get('duration', ''),
                'view_count': statistics.get('viewCount', '0'),
                'like_count': statistics.get('likeCount', '0'),
                'comment_count': statistics.get('commentCount', '0'),
                'tags': snippet.get('tags', []),
                'category_id': snippet.get('categoryId', ''),
                'thumbnail_url': snippet.get('thumbnails', {}).get('high', {}).get('url', ''),
                'raw_data': video
            }
            
        except requests.exceptions.RequestException as e:
            return {
                'success': False,
                'error': f'API request failed: {str(e)}'
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Unexpected error: {str(e)}'
            }
    
    def search_videos(self, query: str, max_results: int = 10) -> Dict[str, Any]:
        """
        Search for videos using YouTube Data API
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            Dictionary with search results
        """
        try:
            url = f"{self.base_url}/search"
            params = {
                'key': self.api_key,
                'q': query,
                'part': 'snippet',
                'type': 'video',
                'maxResults': max_results,
                'order': 'relevance'
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            videos = []
            for item in data.get('items', []):
                snippet = item.get('snippet', {})
                videos.append({
                    'video_id': item.get('id', {}).get('videoId', ''),
                    'title': snippet.get('title', ''),
                    'description': snippet.get('description', ''),
                    'channel_title': snippet.get('channelTitle', ''),
                    'published_at': snippet.get('publishedAt', ''),
                    'thumbnail_url': snippet.get('thumbnails', {}).get('high', {}).get('url', '')
                })
            
            return {
                'success': True,
                'videos': videos,
                'total_results': data.get('pageInfo', {}).get('totalResults', 0)
            }
            
        except requests.exceptions.RequestException as e:
            return {
                'success': False,
                'error': f'API request failed: {str(e)}'
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Unexpected error: {str(e)}'
            }
    
    def verify_video_exists(self, url: str) -> Dict[str, Any]:
        """
        Verify if a YouTube video exists and is accessible
        
        Args:
            url: YouTube URL
            
        Returns:
            Dictionary with verification results
        """
        video_id = self.extract_video_id(url)
        
        if not video_id:
            return {
                'verified': False,
                'message': 'Invalid YouTube URL format',
                'details': {'error': 'Could not extract video ID from URL'}
            }
        
        video_info = self.get_video_info(video_id)
        
        if not video_info.get('success'):
            return {
                'verified': False,
                'message': f'Video verification failed: {video_info.get("error", "Unknown error")}',
                'details': {
                    'video_id': video_id,
                    'error': video_info.get('error', 'Unknown error')
                }
            }
        
        return {
            'verified': True,
            'message': f'Video verified successfully: "{video_info["title"]}" by {video_info["channel_title"]}',
            'details': {
                'video_id': video_id,
                'title': video_info['title'],
                'channel_title': video_info['channel_title'],
                'published_at': video_info['published_at'],
                'duration': video_info['duration'],
                'view_count': video_info['view_count'],
                'thumbnail_url': video_info['thumbnail_url']
            }
        }
