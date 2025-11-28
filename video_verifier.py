import os
import tempfile
from typing import Dict, Any, Optional, List, Tuple
import cv2
import requests
from PIL import Image, ImageDraw, ImageFont
import subprocess
import json
import asyncio

from .image_verifier import ImageVerifier
from .youtube_api import YouTubeDataAPI
from config import config
import time

class VideoVerifier:
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the VideoVerifier with SerpApi credentials
        
        Args:
            api_key: SerpApi API key. If None, will try to get from environment
        """
        self.api_key = api_key or config.SERP_API_KEY
        if not self.api_key:
            raise ValueError("SERP_API_KEY environment variable or api_key parameter is required")
        
        # Initialize image verifier for frame analysis
        self.image_verifier = ImageVerifier(api_key)
        
        # Initialize YouTube Data API client
        self.youtube_api = YouTubeDataAPI(api_key)
        
        # Video processing parameters
        self.frame_interval = 4  # Extract frame every 4 seconds
        self.clip_duration = 5   # Duration of misleading clip in seconds
        
    async def verify(self, video_path: Optional[str] = None, claim_context: str = "", claim_date: str = "", video_url: Optional[str] = None) -> Dict[str, Any]:
        """
        Verify a video and generate a visual counter-measure video if false context is detected
        
        Args:
            video_path: Path to the video file
            claim_context: The claimed context of the video
            claim_date: The claimed date of the video
            
        Returns:
            Dictionary with verification results and output file path
        """
        try:
            # If a video URL is supplied, determine the best verification approach
            if video_url and not video_path:
                # Check if it's a YouTube URL and use API verification
                if self._is_youtube_url(video_url):
                    return await self._verify_youtube_video(video_url, claim_context, claim_date)
                
                # Check if it's a supported platform for yt-dlp
                if self._is_supported_platform(video_url):
                    return await self._verify_with_ytdlp(video_url, claim_context, claim_date)
                
                # For unsupported platforms, try direct download first; if not a real video, fallback to yt-dlp
                try:
                    video_path = await self._download_video(video_url)
                except Exception as direct_err:
                    # Always attempt yt-dlp as fallback when available
                    try:
                        video_path = await self._download_with_ytdlp(video_url)
                        used_ytdlp = True
                    except Exception as ytdlp_err:
                        # Return the more informative error
                        raise RuntimeError(f"Direct download failed: {direct_err}; yt-dlp failed: {ytdlp_err}")

            # Extract key frames from video
            frames = await self._extract_key_frames(video_path)

            # If extraction failed and we have a URL, try yt-dlp fallback once
            if (not frames) and video_url and config.USE_STREAM_DOWNLOADER and not used_ytdlp:
                video_path = await self._download_with_ytdlp(video_url)
                used_ytdlp = True
                frames = await self._extract_key_frames(video_path)
            
            if not frames:
                return {
                    "verified": False,
                    "message": "Could not extract frames from video",
                    "details": {"error": "Frame extraction failed"}
                }
            
            # STEP 0: Analyze frames with Gemini Vision first (direct frame analysis)
            preliminary_vision_analysis = await self._analyze_frames_with_vision(
                frames, claim_context, claim_date
            )
            print(f"✅ Gemini Vision analysis result: {preliminary_vision_analysis.get('overall_verdict', 'unknown')}")
            
            # STEP 1: Analyze frames with reverse image search (existing approach)
            # Wrap in try/except so vision analysis can still proceed if search fails
            reverse_search_analysis = None
            try:
                reverse_search_analysis = await self._analyze_frames(frames, claim_context, claim_date)
            except Exception as search_error:
                print(f"⚠️ Reverse image search analysis failed (will use vision analysis only): {search_error}")
                # Continue with vision analysis only
            
            # STEP 2: Synthesize vision analysis + reverse image search results
            if reverse_search_analysis:
                final_analysis = self._synthesize_video_analyses(
                    preliminary_vision_analysis=preliminary_vision_analysis,
                    reverse_search_analysis=reverse_search_analysis,
                    frames=frames,
                    claim_context=claim_context,
                    claim_date=claim_date,
                )
                
                if final_analysis:
                    analysis = final_analysis
                else:
                    # Fallback: use vision analysis if synthesis fails
                    if preliminary_vision_analysis.get("overall_verdict") in ["false", "true"]:
                        analysis = preliminary_vision_analysis
                    else:
                        analysis = reverse_search_analysis
            else:
                # No reverse search results, use vision analysis only
                print("⚠️ Using vision analysis only (reverse image search unavailable)")
                analysis = preliminary_vision_analysis
            
            if analysis.get("overall_verdict") != "false":
                return {
                    "verified": analysis.get("overall_verdict") == "true",
                    "message": analysis.get("overall_summary") or "No decisive false context detected in video frames",
                    "details": {
                        "frames_analyzed": len(frames),
                        "overall_verdict": analysis.get("overall_verdict"),
                        "frame_summaries": analysis.get("frame_summaries", []),
                    }
                }
            
            # Generate video counter-measure only if we have a specific false frame
            false_ctx = analysis.get("false_context_frame")
            if not false_ctx:
                return {
                    "verified": False,
                    "message": analysis.get("overall_summary") or "False context inferred but no specific frame identified for counter-measure.",
                    "details": {
                        "frames_analyzed": len(frames),
                        "overall_verdict": analysis.get("overall_verdict"),
                        "frame_summaries": analysis.get("frame_summaries", []),
                    }
                }
            output_path = await self._generate_video_counter_measure(
                video_path, false_ctx, claim_context, claim_date
            )
            
            result: Dict[str, Any] = {
                "verified": True,
                "message": "False context detected and video counter-measure generated",
                "output_path": output_path,
                "false_context_frame": analysis.get("false_context_frame"),
                "details": {
                    "frames_analyzed": len(frames),
                    "claim_context": claim_context,
                    "claim_date": claim_date
                }
            }
            # Attempt Cloudinary cleanup (best-effort) before responding
            await self._cloudinary_cleanup_prefix(config.CLOUDINARY_FOLDER or "frames")
            return result
            
        except Exception as e:
            return {
                "verified": False,
                "message": f"Error during video verification: {str(e)}",
                "details": {"error": str(e)}
            }

    async def _download_video(self, url: str) -> str:
        try:
            resp = requests.get(url, stream=True, timeout=30)
            resp.raise_for_status()
            content_type = (resp.headers.get("Content-Type") or "").lower()
            looks_like_video = ("video" in content_type) or url.lower().endswith((".mp4", ".mov", ".mkv", ".webm", ".m4v"))
            if not looks_like_video:
                raise RuntimeError(f"URL is not a direct video (content-type={content_type})")
            suffix = ".mp4"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            bytes_written = 0
            for chunk in resp.iter_content(chunk_size=1 << 14):
                if chunk:
                    tmp.write(chunk)
                    bytes_written += len(chunk)
            tmp.close()
            # Heuristic: reject tiny files that aren't valid containers
            if bytes_written < 200 * 1024:  # 200KB
                os.unlink(tmp.name)
                raise RuntimeError("Downloaded file too small to be a valid video")
            return tmp.name
        except Exception as e:
            raise RuntimeError(f"Failed to download video: {e}")

    async def _download_with_ytdlp(self, url: str) -> str:
        try:
            # Resolve yt-dlp binary
            ytdlp_bin = self._resolve_ytdlp_bin()
            tmp_dir = tempfile.mkdtemp()
            out_path = os.path.join(tmp_dir, "video.%(ext)s")
            cmd = [
                ytdlp_bin,
                "-f", "best[height<=720]/best[height<=480]/best",
                "--no-warnings",
                "--no-call-home",
                "--no-progress",
                "--restrict-filenames",
                "--socket-timeout", "30",
                "--retries", "3",
                "--fragment-retries", "3",
                "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "--extractor-retries", "3",
                "-o", out_path,
                url,
            ]
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            try:
                await asyncio.wait_for(proc.communicate(), timeout=config.STREAM_DOWNLOAD_TIMEOUT)
            except asyncio.TimeoutError:
                proc.kill()
                raise RuntimeError("yt-dlp timed out")
            if proc.returncode != 0:
                # capture stderr for diagnostics
                raise RuntimeError("yt-dlp failed (non-zero exit)")
            # Resolve resulting file (first mp4 in dir)
            for fname in os.listdir(tmp_dir):
                if fname.lower().endswith((".mp4", ".mkv", ".webm", ".mov")):
                    return os.path.join(tmp_dir, fname)
            raise RuntimeError("yt-dlp produced no playable file")
        except Exception as e:
            raise RuntimeError(f"yt-dlp error: {e}")

    def _resolve_ytdlp_bin(self) -> str:
        # Prefer configured path if executable, else try PATH
        cand = config.YTDLP_BIN or "yt-dlp"
        if os.path.isabs(cand) and os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
        from shutil import which
        found = which(cand) or which("yt-dlp")
        if not found:
            raise RuntimeError("yt-dlp not found on PATH; install yt-dlp or set YTDLP_BIN")
        return found
    
    def _is_youtube_url(self, url: str) -> bool:
        """
        Check if the URL is a YouTube URL
        
        Args:
            url: URL to check
            
        Returns:
            True if it's a YouTube URL, False otherwise
        """
        youtube_domains = [
            'youtube.com',
            'www.youtube.com',
            'youtu.be',
            'www.youtu.be',
            'm.youtube.com'
        ]
        
        url_lower = url.lower()
        return any(domain in url_lower for domain in youtube_domains)
    
    def _is_supported_platform(self, url: str) -> bool:
        """
        Check if the URL is from a platform supported by yt-dlp
        
        Args:
            url: URL to check
            
        Returns:
            True if it's a supported platform, False otherwise
        """
        supported_domains = [
            # Video platforms
            'instagram.com', 'www.instagram.com',
            'tiktok.com', 'www.tiktok.com', 'vm.tiktok.com',
            'twitter.com', 'x.com', 'www.twitter.com', 'www.x.com',
            'facebook.com', 'www.facebook.com', 'fb.watch',
            'vimeo.com', 'www.vimeo.com',
            'twitch.tv', 'www.twitch.tv',
            'dailymotion.com', 'www.dailymotion.com',
            'youtube.com', 'www.youtube.com', 'youtu.be', 'www.youtu.be',
            
            # Image platforms
            'imgur.com', 'www.imgur.com',
            'flickr.com', 'www.flickr.com',
            
            # Audio platforms
            'soundcloud.com', 'www.soundcloud.com',
            'mixcloud.com', 'www.mixcloud.com',
            
            # Alternative platforms
            'lbry.tv', 'odysee.com', 'www.odysee.com',
            'telegram.org', 't.me',
            'linkedin.com', 'www.linkedin.com',
            
            # Other platforms
            'streamable.com', 'www.streamable.com',
            'rumble.com', 'www.rumble.com',
            'bitchute.com', 'www.bitchute.com',
            'peertube.tv', 'www.peertube.tv'
        ]
        
        url_lower = url.lower()
        return any(domain in url_lower for domain in supported_domains)
    
    async def _verify_with_ytdlp(self, url: str, claim_context: str, claim_date: str) -> Dict[str, Any]:
        """
        Verify a video from supported platforms using yt-dlp + visual analysis
        
        Args:
            url: Video URL from supported platform
            claim_context: The claimed context of the video
            claim_date: The claimed date of the video
            
        Returns:
            Dictionary with verification results
        """
        try:
            print(f"🔍 DEBUG: Verifying video with yt-dlp: {url}")
            
            # Download video using yt-dlp
            video_path = await self._download_with_ytdlp(url)
            
            # Extract frames for visual verification
            frames = await self._extract_key_frames(video_path)
            
            if frames:
                # Perform visual analysis on frames
                visual_analysis = await self._analyze_frames_visually(frames, claim_context, claim_date)
                
                # Get platform info
                platform = self._get_platform_name(url)
                
                return {
                    'verified': visual_analysis.get('verified', True),
                    'message': f"✅ Video verified from {platform}: {visual_analysis.get('message', 'Visual analysis completed')}",
                    'details': {
                        'verification_method': 'ytdlp_plus_visual',
                        'platform': platform,
                        'url': url,
                        'claim_context': claim_context,
                        'claim_date': claim_date,
                        'visual_analysis': visual_analysis.get('details', {}),
                        'frames_analyzed': len(frames)
                    },
                    'reasoning': f"Video verified from {platform} using yt-dlp and visual analysis. {visual_analysis.get('reasoning', '')}",
                    'sources': [url]
                }
            else:
                # Fallback to basic verification if frames can't be extracted
                platform = self._get_platform_name(url)
                return {
                    'verified': True,
                    'message': f"✅ Video verified from {platform} (basic verification - frame extraction failed)",
                    'details': {
                        'verification_method': 'ytdlp_basic',
                        'platform': platform,
                        'url': url,
                        'claim_context': claim_context,
                        'claim_date': claim_date,
                        'limitation': 'Visual frame analysis unavailable'
                    },
                    'reasoning': f"Video verified from {platform} using yt-dlp. Visual analysis was not possible due to frame extraction issues.",
                    'sources': [url]
                }
                
        except Exception as e:
            platform = self._get_platform_name(url)
            return {
                'verified': False,
                'message': f'Error during {platform} video verification: {str(e)}',
                'details': {'error': str(e), 'platform': platform},
                'reasoning': f'An error occurred while verifying the {platform} video: {str(e)}',
                'sources': [url]
            }
    
    def _get_platform_name(self, url: str) -> str:
        """Get the platform name from URL"""
        url_lower = url.lower()
        
        if 'instagram.com' in url_lower:
            return 'Instagram'
        elif 'tiktok.com' in url_lower or 'vm.tiktok.com' in url_lower:
            return 'TikTok'
        elif 'twitter.com' in url_lower or 'x.com' in url_lower:
            return 'Twitter/X'
        elif 'facebook.com' in url_lower or 'fb.watch' in url_lower:
            return 'Facebook'
        elif 'vimeo.com' in url_lower:
            return 'Vimeo'
        elif 'twitch.tv' in url_lower:
            return 'Twitch'
        elif 'dailymotion.com' in url_lower:
            return 'DailyMotion'
        elif 'imgur.com' in url_lower:
            return 'Imgur'
        elif 'soundcloud.com' in url_lower:
            return 'SoundCloud'
        elif 'mixcloud.com' in url_lower:
            return 'Mixcloud'
        elif 'lbry.tv' in url_lower or 'odysee.com' in url_lower:
            return 'LBRY/Odysee'
        elif 'telegram.org' in url_lower or 't.me' in url_lower:
            return 'Telegram'
        elif 'linkedin.com' in url_lower:
            return 'LinkedIn'
        else:
            return 'Unknown Platform'
    
    async def _verify_youtube_video(self, url: str, claim_context: str, claim_date: str) -> Dict[str, Any]:
        """
        Verify a YouTube video using hybrid approach: API metadata + yt-dlp for visual analysis
        
        Args:
            url: YouTube URL
            claim_context: The claimed context of the video
            claim_date: The claimed date of the video
            
        Returns:
            Dictionary with verification results
        """
        try:
            # Step 1: Use YouTube Data API to verify the video exists and get metadata
            verification_result = self.youtube_api.verify_video_exists(url)
            
            if not verification_result.get('verified'):
                return {
                    'verified': False,
                    'message': f'YouTube video verification failed: {verification_result.get("message", "Unknown error")}',
                    'details': verification_result.get('details', {}),
                    'reasoning': f'The video could not be verified through YouTube Data API. {verification_result.get("message", "Unknown error")}',
                    'sources': [url]
                }
            
            # Step 2: Video exists, now try to download for visual analysis
            video_details = verification_result.get('details', {})
            
            try:
                # Attempt to download video for frame analysis
                print(f"🔍 DEBUG: Attempting to download video for visual analysis: {url}")
                video_path = await self._download_with_ytdlp(url)
                
                # Extract frames for visual verification
                frames = await self._extract_key_frames(video_path)
                
                if frames:
                    # Perform visual analysis on frames
                    visual_analysis = await self._analyze_frames_visually(frames, claim_context, claim_date)
                    
                    # Combine metadata + visual analysis
                    return {
                        'verified': visual_analysis.get('verified', True),
                        'message': f"✅ Video verified with visual analysis: '{video_details.get('title', 'Unknown Title')}' by {video_details.get('channel_title', 'Unknown Channel')}\n\n{visual_analysis.get('message', '')}",
                        'details': {
                            'verification_method': 'hybrid_youtube_api_plus_visual',
                            'video_id': video_details.get('video_id'),
                            'title': video_details.get('title'),
                            'channel_title': video_details.get('channel_title'),
                            'published_at': video_details.get('published_at'),
                            'duration': video_details.get('duration'),
                            'view_count': video_details.get('view_count'),
                            'thumbnail_url': video_details.get('thumbnail_url'),
                            'claim_context': claim_context,
                            'claim_date': claim_date,
                            'visual_analysis': visual_analysis.get('details', {}),
                            'frames_analyzed': len(frames)
                        },
                        'reasoning': f"Video verified through YouTube Data API and visual analysis. {visual_analysis.get('reasoning', '')}",
                        'sources': [url]
                    }
                else:
                    # Fallback to metadata-only verification
                    print(f"⚠️ DEBUG: Could not extract frames, falling back to metadata verification")
                    return self._create_metadata_only_response(video_details, claim_context, claim_date, url)
                    
            except Exception as download_error:
                # Fallback to metadata-only verification if download fails
                print(f"⚠️ DEBUG: Video download failed: {download_error}, falling back to metadata verification")
                return self._create_metadata_only_response(video_details, claim_context, claim_date, url)
            
        except Exception as e:
            return {
                'verified': False,
                'message': f'Error during YouTube video verification: {str(e)}',
                'details': {'error': str(e)},
                'reasoning': f'An error occurred while verifying the YouTube video: {str(e)}',
                'sources': [url]
            }
    
    def _create_metadata_only_response(self, video_details: Dict[str, Any], claim_context: str, claim_date: str, url: str) -> Dict[str, Any]:
        """Create a metadata-only verification response when visual analysis fails"""
        verification_message = f"✅ Video verified (metadata only): '{video_details.get('title', 'Unknown Title')}' by {video_details.get('channel_title', 'Unknown Channel')}"
        
        # Add context analysis if available
        if claim_context and claim_context.lower() != "the user wants to verify the content of the provided youtube video.":
            verification_message += f"\n\n📝 Claim Context: {claim_context}"
            verification_message += f"\n⚠️ Note: Visual content analysis unavailable - only metadata verification performed"
        
        if claim_date and claim_date.strip():
            verification_message += f"\n📅 Claimed Date: {claim_date}"
        
        verification_message += f"\n📊 Video Stats: {video_details.get('view_count', 'Unknown')} views, Published: {video_details.get('published_at', 'Unknown')}"
        
        return {
            'verified': True,
            'message': verification_message,
            'details': {
                'verification_method': 'youtube_data_api_metadata_only',
                'video_id': video_details.get('video_id'),
                'title': video_details.get('title'),
                'channel_title': video_details.get('channel_title'),
                'published_at': video_details.get('published_at'),
                'duration': video_details.get('duration'),
                'view_count': video_details.get('view_count'),
                'thumbnail_url': video_details.get('thumbnail_url'),
                'claim_context': claim_context,
                'claim_date': claim_date,
                'limitation': 'Visual content analysis unavailable'
            },
            'reasoning': f"Video verified through YouTube Data API metadata only. Visual content analysis was not possible due to download limitations.",
            'sources': [url]
        }
    
    async def _analyze_frames_visually(self, frames: List[Tuple[str, float]], claim_context: str, claim_date: str) -> Dict[str, Any]:
        """
        Analyze extracted frames for visual verification
        
        Args:
            frames: List of (frame_path, timestamp) tuples
            claim_context: The claimed context
            claim_date: The claimed date
            
        Returns:
            Dictionary with visual analysis results
        """
        try:
            # Analyze each frame using the image verifier
            frame_analyses = []
            
            for frame_path, timestamp in frames:
                try:
                    frame_result = await self.image_verifier.verify(
                        image_path=frame_path,
                        claim_context=f"{claim_context} (Frame at {timestamp}s)",
                        claim_date=claim_date
                    )
                    frame_analyses.append({
                        'timestamp': timestamp,
                        'result': frame_result
                    })
                except Exception as e:
                    print(f"⚠️ DEBUG: Frame analysis failed for {timestamp}s: {e}")
                    continue
            
            if not frame_analyses:
                return {
                    'verified': False,
                    'message': 'No frames could be analyzed',
                    'details': {'error': 'All frame analyses failed'},
                    'reasoning': 'Visual analysis failed for all extracted frames'
                }
            
            # Determine overall verification result
            verified_count = sum(1 for analysis in frame_analyses if analysis['result'].get('verified', False))
            total_frames = len(frame_analyses)
            
            if verified_count == 0:
                verification_status = False
                message = f"❌ Visual analysis found no supporting evidence in {total_frames} frames"
            elif verified_count == total_frames:
                verification_status = True
                message = f"✅ Visual analysis confirmed claim in all {total_frames} frames"
            else:
                verification_status = True  # Partial verification
                message = f"⚠️ Visual analysis partially confirmed claim in {verified_count}/{total_frames} frames"
            
            return {
                'verified': verification_status,
                'message': message,
                'details': {
                    'frames_analyzed': total_frames,
                    'verified_frames': verified_count,
                    'frame_results': frame_analyses
                },
                'reasoning': f"Analyzed {total_frames} video frames. {verified_count} frames supported the claim."
            }
            
        except Exception as e:
            return {
                'verified': False,
                'message': f'Visual analysis failed: {str(e)}',
                'details': {'error': str(e)},
                'reasoning': f'Error during visual frame analysis: {str(e)}'
            }
    
    async def _extract_key_frames(self, video_path: str) -> List[Tuple[str, float]]:
        """
        Extract key frames from video at regular intervals
        
        Args:
            video_path: Path to the video file
            
        Returns:
            List of tuples (frame_path, timestamp)
        """
        try:
            frames = []
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                print(f"Error: Could not open video file {video_path}")
                return []
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            frame_interval_frames = int(fps * self.frame_interval)
            
            frame_count = 0
            saved_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Save frame at regular intervals
                if frame_count % frame_interval_frames == 0:
                    timestamp = frame_count / fps
                    # Save frame into public/frames for local static serving
                    out_dir = os.path.join("public", "frames")
                    os.makedirs(out_dir, exist_ok=True)
                    frame_file = f"frame_{int(timestamp*1000)}.jpg"
                    frame_path = os.path.join(out_dir, frame_file)
                    cv2.imwrite(frame_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                    frames.append((frame_path, timestamp))
                    saved_count += 1
                    
                    # Limit number of frames to analyze
                    if saved_count >= 10:  # Max 10 frames
                        break
                
                frame_count += 1
            
            cap.release()
            return frames
            
        except Exception as e:
            print(f"Error extracting frames: {e}")
            return []
    
    async def _analyze_frames_with_vision(
        self,
        frames: List[Tuple[str, float]],
        claim_context: str,
        claim_date: str
    ) -> Dict[str, Any]:
        """
        Analyze video frames directly with Gemini Vision (first pass).
        Detects AI-generated/deepfake/manipulation in frames.
        
        Args:
            frames: List of (frame_path, timestamp) tuples
            claim_context: The claimed context
            claim_date: The claimed date
            
        Returns:
            Dictionary with preliminary vision analysis
        """
        try:
            if not self.image_verifier.gemini_model:
                return {
                    "overall_verdict": "uncertain",
                    "overall_summary": "Gemini Vision not available",
                    "frame_analyses": [],
                    "analysis_method": "vision_unavailable",
                }
            
            frame_analyses = []
            for frame_path, timestamp in frames:
                try:
                    # Use image verifier's vision analysis method
                    vision_result = await self.image_verifier._analyze_image_with_vision(
                        image_path=frame_path,
                        image_url=None,
                        claim_context=f"{claim_context} (Frame at {timestamp}s)",
                        claim_date=claim_date
                    )
                    frame_analyses.append({
                        "timestamp": timestamp,
                        "frame_path": frame_path,
                        "vision_analysis": vision_result,
                    })
                except Exception as e:
                    print(f"⚠️ Vision analysis failed for frame at {timestamp}s: {e}")
                    continue
            
            if not frame_analyses:
                return {
                    "overall_verdict": "uncertain",
                    "overall_summary": "No frames could be analyzed with vision",
                    "frame_analyses": [],
                    "analysis_method": "vision_no_frames",
                }
            
            # Aggregate vision results across frames
            false_count = sum(1 for fa in frame_analyses if fa["vision_analysis"].get("verdict") == "false")
            true_count = sum(1 for fa in frame_analyses if fa["vision_analysis"].get("verdict") == "true")
            uncertain_count = len(frame_analyses) - false_count - true_count
            
            # Determine overall verdict
            if false_count > true_count and false_count > uncertain_count:
                overall_verdict = "false"
                overall_summary = f"Vision analysis detected manipulation/AI-generated content in {false_count}/{len(frame_analyses)} frames"
            elif true_count > false_count and true_count > uncertain_count:
                overall_verdict = "true"
                overall_summary = f"Vision analysis found authentic content in {true_count}/{len(frame_analyses)} frames"
            else:
                overall_verdict = "uncertain"
                overall_summary = f"Vision analysis inconclusive: {true_count} true, {false_count} false, {uncertain_count} uncertain across {len(frame_analyses)} frames"
            
            return {
                "overall_verdict": overall_verdict,
                "overall_summary": overall_summary,
                "frame_analyses": frame_analyses,
                "false_count": false_count,
                "true_count": true_count,
                "uncertain_count": uncertain_count,
                "analysis_method": "gemini_vision",
            }
            
        except Exception as e:
            print(f"[vision] Error in frame vision analysis: {e}")
            return {
                "overall_verdict": "uncertain",
                "overall_summary": f"Error during vision analysis: {str(e)}",
                "frame_analyses": [],
                "analysis_method": "vision_error",
            }

    def _synthesize_video_analyses(
        self,
        preliminary_vision_analysis: Dict[str, Any],
        reverse_search_analysis: Dict[str, Any],
        frames: List[Tuple[str, float]],
        claim_context: str,
        claim_date: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Synthesize Gemini Vision analysis with reverse image search results.
        """
        try:
            if not self.image_verifier.gemini_model:
                return None
            
            prompt = f"""You are an expert video verification analyst. Combine direct frame analysis (Gemini Vision) with reverse image search evidence to produce a final verdict.

CLAIM: {claim_context}
CLAIM DATE: {claim_date}

DIRECT FRAME ANALYSIS (Gemini Vision):
{json.dumps(preliminary_vision_analysis or {}, indent=2, ensure_ascii=False)}

REVERSE IMAGE SEARCH ANALYSIS:
{json.dumps(reverse_search_analysis or {}, indent=2, ensure_ascii=False)}

TOTAL FRAMES ANALYZED: {len(frames)}

INSTRUCTIONS:
- Combine both analyses to make a final decision (true/false/uncertain)
- If vision analysis detects AI-generated/manipulated content in multiple frames, prioritize that
- If reverse image search finds contradictory evidence, factor that in
- Consider consistency across frames
- If evidence is thin, keep the tone cautious
- Provide clear, actionable messaging for the end user

Respond ONLY in this JSON format:
{{
  "overall_verdict": "true|false|uncertain",
  "overall_summary": "Concise user-facing summary combining both analyses",
  "confidence": "high|medium|low",
  "reasoning": "Brief reasoning trail you followed",
  "vision_findings": "Key findings from direct frame analysis",
  "search_findings": "Key findings from reverse image search"
}}"""
            
            response = self.image_verifier.gemini_model.generate_content(prompt)
            response_text = response.text.strip()
            
            if response_text.startswith("```json"):
                response_text = response_text.replace("```json", "").replace("```", "").strip()
            elif response_text.startswith("```"):
                response_text = response_text.replace("```", "").strip()
            
            final_analysis = json.loads(response_text)
            final_analysis.setdefault("overall_verdict", "uncertain")
            final_analysis.setdefault("overall_summary", "Unable to synthesize final verdict.")
            final_analysis.setdefault("confidence", "low")
            final_analysis["analysis_method"] = "hybrid_vision_and_search"
            
            # Preserve frame summaries and sources from reverse search
            final_analysis["frame_summaries"] = reverse_search_analysis.get("frame_summaries", [])
            final_analysis["consolidated_sources"] = reverse_search_analysis.get("consolidated_sources", [])
            final_analysis["preliminary_vision_analysis"] = preliminary_vision_analysis
            final_analysis["reverse_search_analysis"] = reverse_search_analysis
            
            return final_analysis
            
        except Exception as e:
            print(f"Video hybrid synthesis error: {e}")
            return None

    async def _analyze_frames(self, frames: List[Tuple[str, float]], 
                             claim_context: str, claim_date: str) -> Dict[str, Any]:
        """
        Analyze extracted frames for false context
        
        Args:
            frames: List of (frame_path, timestamp) tuples
            claim_context: The claimed context
            claim_date: The claimed date
            
        Returns:
            Aggregated analysis with overall verdict and optional false frame
        """
        frame_summaries: List[Dict[str, Any]] = []
        false_hit: Optional[Dict[str, Any]] = None
        true_hit: Optional[Dict[str, Any]] = None
        saw_false_validated = False
        saw_true_validated = False
        # 1) Per-frame: only gather evidence; defer verdict to a single final pass
        all_evidence: List[Dict[str, Any]] = []
        for frame_path, timestamp in frames:
            try:
                # Upload frame to Cloudinary if configured, else local static URL
                frame_url = None
                if config.CLOUDINARY_CLOUD_NAME and (config.CLOUDINARY_UPLOAD_PRESET or (config.CLOUDINARY_API_KEY and config.CLOUDINARY_API_SECRET)):
                    frame_url = await self._upload_frame_cloudinary(frame_path)
                if not frame_url:
                    # fallback local (note: SerpApi can't access localhost; cloudinary is preferred)
                    from urllib.parse import quote
                    rel = frame_path.replace(os.path.join("public", ''), "") if frame_path.startswith("public"+os.sep) else os.path.basename(frame_path)
                    frame_url = f"http://127.0.0.1:{config.SERVICE_PORT}/static/{quote(rel)}"
                print("[video] analyze_frame", {"ts": timestamp, "path": frame_path})
                # Gather evidence only for this frame
                ev = await self.image_verifier.gather_evidence(
                    image_path=None, image_url=frame_url, claim_context=claim_context
                )
                all_evidence.extend(ev or [])
                # Populate a placeholder entry per frame (no verdict yet)
                frame_entry = {
                    "timestamp": timestamp,
                    "verdict": None,
                    "summary": None,
                    "sources": None,
                    "frame_path": frame_path,
                    "validator": None,
                    "details": {"evidence": ev or []},
                }
                # Compute rule-based confidence (0..1)
                conf = 0.2
                reasons: List[str] = []
                checks = {}
                if frame_entry["verdict"] == "true":
                    if checks.get("relation_comention"):
                        conf += 0.3; reasons.append("relation_comention")
                if frame_entry["verdict"] == "false":
                    if not checks.get("relation_comention"):
                        conf += 0.25; reasons.append("no_relation_support")
                if checks.get("timeframe_citations") or checks.get("timeframe_match"):
                    conf += 0.15; reasons.append("timeframe_match")
                eos = checks.get("entity_overlap_score")
                try:
                    if eos is not None and float(eos) >= 0.7:
                        conf += 0.1; reasons.append("entity_overlap")
                except Exception:
                    pass
                # Penalize if sources dominated by low-priority domains
                low_priority_hits = 0
                total_sources = 0
                try:
                    from urllib.parse import urlparse
                    for s in (frame_entry.get("sources") or []):
                        total_sources += 1
                        net = urlparse((s.get("link") or "")).netloc
                        if net in config.LOW_PRIORITY_DOMAINS:
                            low_priority_hits += 1
                except Exception:
                    pass
                if total_sources > 0 and low_priority_hits / float(total_sources) >= 0.5:
                    conf -= 0.2; reasons.append("low_priority_sources")
                if conf < 0.0: conf = 0.0
                if conf > 1.0: conf = 1.0
                frame_entry["confidence"] = conf
                frame_entry["confidence_reasons"] = reasons
                print("[video] frame_result", {"ts": timestamp, "verdict": frame_entry["verdict"], "passed": (frame_entry.get("validator") or {}).get("passed")})
                # No per-frame debug when gathering evidence only
                frame_summaries.append(frame_entry)
                # No per-frame validator flags when gathering evidence only
                if false_hit is None:
                    false_hit = {
                        "timestamp": timestamp,
                        "frame_path": frame_path,
                        "evidence_image": None,
                        "details": {"evidence": ev or []},
                    }
                if true_hit is None:
                    true_hit = {
                        "timestamp": timestamp,
                        "frame_path": frame_path,
                        "details": {"evidence": ev or []},
                    }
                
            except Exception as e:
                print(f"Error analyzing frame {frame_path}: {e}")
                # Keep files even on error for debugging
        
        # 2) Single final pass: send aggregated evidence to image verifier's Gemini summarizer
        #    Reuse image verifier's structured summarizer for a consolidated verdict
        # Use the simple majority-based summarizer per product rule
        final_llm = self.image_verifier._summarize_with_gemini_majority(
            claim_context=claim_context,
            claim_date=claim_date,
            evidence=all_evidence[:24],  # cap to keep prompt manageable
        ) or {}
        final_verdict = (final_llm.get("verdict") or "uncertain").lower()
        # Prefer LLM clarification if present; else fallback to previous summary
        final_summary = final_llm.get("clarification") or final_llm.get("summary") or "Consolidated evidence analyzed."

        # Deterministic co-mention vote to override ambiguous LLM outcomes
        def _tokens(text: str) -> List[str]:
            import re
            return re.findall(r"[a-z0-9]{3,}", (text or "").lower())

        def _split_relation(claim: str) -> Tuple[List[str], List[str]]:
            # Heuristic: split on ' with ' to get subject vs object; fallback to all tokens as subject
            cl = (claim or "").strip()
            i = cl.lower().find(" with ")
            if i != -1:
                subj = cl[:i].strip()
                obj = cl[i+6:].strip().split(".")[0]
            else:
                subj = cl
                obj = ""
            return list(set(_tokens(subj))), list(set(_tokens(obj)))

        def _evidence_text(ev: Dict[str, Any]) -> str:
            return " ".join([t for t in [ev.get("title"), ev.get("snippet"), ev.get("source")] if t])

        subj_toks, obj_toks = _split_relation(claim_context)
        support = 0
        contra = 0
        for ev in all_evidence[:24]:
            txt_toks = set(_tokens(_evidence_text(ev)))
            if not txt_toks:
                continue
            subj_hit = bool(subj_toks and (set(subj_toks) & txt_toks))
            obj_hit = bool(obj_toks and (set(obj_toks) & txt_toks))
            if subj_hit and obj_hit:
                support += 1
            elif subj_hit and obj_toks:
                # mentions subject but not the claimed object → treat as contradiction to the claimed relation
                contra += 1

        # Apply override rules: prioritize clear majority; else keep LLM
        if support == 0 and contra > 0:
            final_verdict = "false"  # keep LLM clarification
        elif support > contra and (support - contra) >= 1:
            final_verdict = "true"   # keep LLM clarification
        elif contra > support and (contra - support) >= 1:
            final_verdict = "false"  # keep LLM clarification
        # else keep LLM's verdict/summary

        return {
            "overall_verdict": final_verdict,
            "overall_summary": final_summary,
            "frame_summaries": frame_summaries,
            "consolidated_sources": final_llm.get("top_sources") or self.image_verifier._top_sources(all_evidence, 3),
        }

    async def _upload_frame_cloudinary(self, frame_path: str) -> Optional[str]:
        try:
            import hashlib
            import requests
            cloud = config.CLOUDINARY_CLOUD_NAME
            folder = config.CLOUDINARY_FOLDER.strip('/')
            # Unsigned upload if preset provided
            if config.CLOUDINARY_UPLOAD_PRESET:
                url = f"https://api.cloudinary.com/v1_1/{cloud}/image/upload"
                with open(frame_path, 'rb') as f:
                    files = {"file": f}
                    data = {"upload_preset": config.CLOUDINARY_UPLOAD_PRESET, "folder": folder}
                    r = requests.post(url, files=files, data=data, timeout=30)
                r.raise_for_status()
                return r.json().get("secure_url")
            # Signed upload
            ts = str(int(time.time()))
            params_to_sign = {"timestamp": ts, "folder": folder}
            to_sign = "&".join([f"{k}={v}" for k, v in sorted(params_to_sign.items())]) + config.CLOUDINARY_API_SECRET
            signature = hashlib.sha1(to_sign.encode('utf-8')).hexdigest()
            url = f"https://api.cloudinary.com/v1_1/{cloud}/image/upload"
            with open(frame_path, 'rb') as f:
                files = {"file": f}
                data = {
                    "api_key": config.CLOUDINARY_API_KEY,
                    "timestamp": ts,
                    "signature": signature,
                    "folder": folder,
                }
                r = requests.post(url, files=files, data=data, timeout=30)
            r.raise_for_status()
            return r.json().get("secure_url")
        except Exception as e:
            print(f"Cloudinary upload failed: {e}")
            return None
    
    async def _generate_video_counter_measure(self, video_path: str, 
                                            false_context_frame: Dict[str, Any],
                                            claim_context: str, claim_date: str) -> str:
        """
        Generate a video counter-measure
        
        Args:
            video_path: Path to the original video
            false_context_frame: Information about the false context frame
            claim_context: The claimed context
            claim_date: The claimed date
            
        Returns:
            Path to the generated counter-measure video
        """
        try:
            # Create temporary directory for video processing
            temp_dir = tempfile.mkdtemp()
            
            # Generate video components
            title_clip = await self._create_title_clip(temp_dir, claim_context, claim_date)
            misleading_clip = await self._create_misleading_clip(
                video_path, false_context_frame["timestamp"], temp_dir
            )
            debunk_clip = await self._create_debunk_clip(
                temp_dir, false_context_frame, claim_context, claim_date
            )
            verdict_clip = await self._create_verdict_clip(temp_dir)
            
            # Concatenate all clips
            output_path = await self._concatenate_clips(
                [title_clip, misleading_clip, debunk_clip, verdict_clip],
                temp_dir
            )
            
            # Clean up temporary files
            self._cleanup_temp_files(temp_dir)
            
            # Attempt Cloudinary cleanup (best-effort) before responding
            await self._cloudinary_cleanup_prefix(config.CLOUDINARY_FOLDER or "frames")
            return output_path
            
        except Exception as e:
            print(f"Error generating video counter-measure: {e}")
            raise
    
    async def _create_title_clip(self, temp_dir: str, claim_context: str, claim_date: str) -> str:
        """Create title clip with claim information"""
        try:
            # Create title image
            img = Image.new('RGB', (800, 400), 'white')
            draw = ImageDraw.Draw(img)
            
            try:
                font_large = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 36)
                font_medium = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 24)
            except:
                font_large = ImageFont.load_default()
                font_medium = ImageFont.load_default()
            
            # Add title
            title = "FALSE CONTEXT DETECTED"
            title_bbox = draw.textbbox((0, 0), title, font=font_large)
            title_width = title_bbox[2] - title_bbox[0]
            title_x = (800 - title_width) // 2
            draw.text((title_x, 100), title, fill='red', font=font_large)
            
            # Add claim details
            claim_text = f"Claim: {claim_context}, {claim_date}"
            claim_bbox = draw.textbbox((0, 0), claim_text, font=font_medium)
            claim_width = claim_bbox[2] - claim_bbox[0]
            claim_x = (800 - claim_width) // 2
            draw.text((claim_x, 200), claim_text, fill='black', font=font_medium)
            
            # Save image
            title_img_path = os.path.join(temp_dir, "title.png")
            img.save(title_img_path)
            
            # Convert to video clip
            title_video_path = os.path.join(temp_dir, "title.mp4")
            await self._image_to_video(title_img_path, title_video_path, duration=3)
            
            return title_video_path
            
        except Exception as e:
            print(f"Error creating title clip: {e}")
            raise
    
    async def _create_misleading_clip(self, video_path: str, timestamp: float, temp_dir: str) -> str:
        """Create clip from original misleading video"""
        try:
            # Calculate frame numbers for 5-second clip
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            
            start_frame = int(timestamp * fps) - int(self.clip_duration / 2 * fps)
            start_frame = max(0, start_frame)
            
            # Extract clip using ffmpeg
            clip_path = os.path.join(temp_dir, "misleading_clip.mp4")
            
            start_time = max(0, timestamp - self.clip_duration / 2)
            
            cmd = [
                'ffmpeg', '-i', video_path,
                '-ss', str(start_time),
                '-t', str(self.clip_duration),
                '-c', 'copy',
                '-y', clip_path
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            await process.communicate()
            
            if process.returncode != 0:
                raise Exception("FFmpeg failed to create misleading clip")
            
            return clip_path
            
        except Exception as e:
            print(f"Error creating misleading clip: {e}")
            raise
    
    async def _create_debunk_clip(self, temp_dir: str, false_context_frame: Dict[str, Any],
                                 claim_context: str, claim_date: str) -> str:
        """Create debunk scene clip with side-by-side comparison"""
        try:
            # Create debunk image using image verifier's counter-measure
            debunk_img_path = await self.image_verifier._generate_counter_measure(
                false_context_frame["frame_path"],
                false_context_frame["evidence_image"],
                claim_context,
                claim_date
            )
            
            # Move to temp directory
            final_debunk_img = os.path.join(temp_dir, "debunk.png")
            os.rename(debunk_img_path, final_debunk_img)
            
            # Convert to video clip
            debunk_video_path = os.path.join(temp_dir, "debunk.mp4")
            await self._image_to_video(final_debunk_img, debunk_video_path, duration=5)
            
            return debunk_video_path
            
        except Exception as e:
            print(f"Error creating debunk clip: {e}")
            raise
    
    async def _create_verdict_clip(self, temp_dir: str) -> str:
        """Create verdict clip with conclusion"""
        try:
            # Create verdict image
            img = Image.new('RGB', (800, 400), 'white')
            draw = ImageDraw.Draw(img)
            
            try:
                font_large = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 36)
                font_medium = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 24)
            except:
                font_large = ImageFont.load_default()
                font_medium = ImageFont.load_default()
            
            # Add verdict
            verdict = "VERDICT: FALSE CONTEXT"
            verdict_bbox = draw.textbbox((0, 0), verdict, font=font_large)
            verdict_width = verdict_bbox[2] - verdict_bbox[0]
            verdict_x = (800 - verdict_width) // 2
            draw.text((verdict_x, 100), verdict, fill='red', font=font_large)
            
            # Add explanation
            explanation = "This content is being used in a false context"
            explanation_bbox = draw.textbbox((0, 0), explanation, font=font_medium)
            explanation_width = explanation_bbox[2] - explanation_bbox[0]
            explanation_x = (800 - explanation_width) // 2
            draw.text((explanation_x, 200), explanation, fill='black', font=font_medium)
            
            # Save image
            verdict_img_path = os.path.join(temp_dir, "verdict.png")
            img.save(verdict_img_path)
            
            # Convert to video clip
            verdict_video_path = os.path.join(temp_dir, "verdict.mp4")
            await self._image_to_video(verdict_img_path, verdict_video_path, duration=3)
            
            return verdict_video_path
            
        except Exception as e:
            print(f"Error creating verdict clip: {e}")
            raise
    
    async def _image_to_video(self, image_path: str, video_path: str, duration: int) -> None:
        """Convert image to video clip using FFmpeg"""
        try:
            cmd = [
                'ffmpeg', '-loop', '1',
                '-i', image_path,
                '-c:v', 'libx264',
                '-t', str(duration),
                '-pix_fmt', 'yuv420p',
                '-y', video_path
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            await process.communicate()
            
            if process.returncode != 0:
                raise Exception("FFmpeg failed to convert image to video")
                
        except Exception as e:
            print(f"Error converting image to video: {e}")
            raise
    
    async def _concatenate_clips(self, clip_paths: List[str], temp_dir: str) -> str:
        """Concatenate multiple video clips into one"""
        try:
            # Create file list for FFmpeg
            file_list_path = os.path.join(temp_dir, "clips.txt")
            with open(file_list_path, 'w') as f:
                for clip_path in clip_paths:
                    f.write(f"file '{clip_path}'\n")
            
            # Concatenate clips
            output_path = tempfile.mktemp(suffix=".mp4")
            
            cmd = [
                'ffmpeg', '-f', 'concat',
                '-safe', '0',
                '-i', file_list_path,
                '-c', 'copy',
                '-y', output_path
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            await process.communicate()
            
            if process.returncode != 0:
                raise Exception("FFmpeg failed to concatenate clips")
            
            return output_path
            
        except Exception as e:
            print(f"Error concatenating clips: {e}")
            raise
    
    def _cleanup_temp_files(self, temp_dir: str) -> None:
        """Clean up temporary files and directory"""
        try:
            import shutil
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"Error cleaning up temp files: {e}")

    async def _cloudinary_cleanup_prefix(self, prefix: str) -> None:
        try:
            if not (config.CLOUDINARY_CLOUD_NAME and (config.CLOUDINARY_API_KEY and config.CLOUDINARY_API_SECRET)):
                return
            # List and delete all resources under the folder prefix (rate-limited; best-effort)
            import requests
            from requests.auth import HTTPBasicAuth
            cloud = config.CLOUDINARY_CLOUD_NAME
            auth = HTTPBasicAuth(config.CLOUDINARY_API_KEY, config.CLOUDINARY_API_SECRET)
            list_url = f"https://api.cloudinary.com/v1_1/{cloud}/resources/image"
            params = {"prefix": prefix, "max_results": 100}
            r = requests.get(list_url, params=params, auth=auth, timeout=20)
            if r.status_code != 200:
                return
            data = r.json()
            public_ids = [res.get("public_id") for res in data.get("resources", []) if res.get("public_id")]
            if not public_ids:
                return
            del_url = f"https://api.cloudinary.com/v1_1/{cloud}/resources/image/delete_by_ids"
            requests.post(del_url, data={"public_ids": ",".join(public_ids)}, auth=auth, timeout=20)
        except Exception as e:
            print(f"Cloudinary cleanup failed: {e}")
