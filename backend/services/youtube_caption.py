# pip install yt-dlp

import yt_dlp
import os
import re
from pathlib import Path

def get_youtube_transcript_ytdlp(video_url, output_file="transcript.txt"):
    """
    Extract YouTube transcript using yt-dlp
    Works perfectly in India - yt-dlp handles all signature/blocking issues
    """
    
    print("[*] Starting transcript extraction with yt-dlp...")
    
    # Extract video ID for reference
    video_id_match = re.search(r'v=([^&]*)', video_url)
    video_id = video_id_match.group(1) if video_id_match else 'unknown'
    
    print(f"[+] Video ID: {video_id}")
    
    # Normalize URL to just the video (remove playlist parameters)
    normalized_url = f"https://www.youtube.com/watch?v={video_id}"
    print(f"[+] Normalized URL: {normalized_url}")
    
    try:
        # Create temp directory for subtitles
        temp_dir = "temp_subs"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Setup yt-dlp options
        ydl_opts = {
            'writeautomaticsub': True,      # Download auto-generated subtitles
            'subtitlesformat': 'vtt',       # Format (can also be 'json3', 'srt', 'ass')
            'skip_download': True,          # Only download subs, not video
            'noplaylist': True,             # Only download the video, not the playlist
            'outtmpl': os.path.join(temp_dir, '%(id)s'),  # Output template
            'quiet': False,                 # Show progress
            'no_warnings': False,
            'sub_langs': 'en',              # Only English subtitles
        }
        
        print("[*] Downloading subtitles...")
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(normalized_url, download=True)  # Use normalized URL
            
        print("[+] Subtitles downloaded successfully")
        
        # Find the subtitle file
        subtitle_file = None
        for file in os.listdir(temp_dir):
            if video_id in file and (file.endswith('.vtt') or file.endswith('.srt')):
                subtitle_file = os.path.join(temp_dir, file)
                print(f"[+] Found subtitle file: {file}")
                break
        
        if not subtitle_file or not os.path.exists(subtitle_file):
            print("[ERROR] Subtitle file not found")
            print(f"[DEBUG] Files in {temp_dir}: {os.listdir(temp_dir)}")
            return None
        
        # Read and parse the subtitle file
        print("[*] Parsing subtitle file...")
        
        transcript_lines = []
        
        if subtitle_file.endswith('.vtt'):
            # Parse VTT format
            with open(subtitle_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                # Skip headers, timestamps, and empty lines
                if line and not line.startswith('WEBVTT') and not '-->' in line and line:
                    transcript_lines.append(line)
        
        elif subtitle_file.endswith('.srt'):
            # Parse SRT format
            with open(subtitle_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                # Skip sequence numbers and timestamps
                if line and not line[0].isdigit() and not '-->' in line and line:
                    transcript_lines.append(line)
        
        if not transcript_lines:
            print("[ERROR] No text extracted from subtitle file")
            return None
        
        # Combine into full transcript
        full_text = "\n".join(transcript_lines)
        
        # Save to output file
        print(f"[*] Saving transcript to {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(full_text)
        
        # Cleanup temp directory
        import shutil
        shutil.rmtree(temp_dir)
        
        print(f"\n✓ SUCCESS!")
        print(f"  File: {output_file}")
        print(f"  Total characters: {len(full_text)}")
        print(f"  Total lines: {len(transcript_lines)}")
        
        return full_text
        
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        return None


# ==================== MAIN ====================

if __name__ == "__main__":
    
    print("=" * 70)
    print("YouTube Transcript Extractor - yt-dlp VERSION (WORKS IN INDIA!)")
    print("=" * 70)
    
    video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    
    print(f"\nTarget video: {video_url}\n")
    
    transcript = get_youtube_transcript_ytdlp(video_url)
    
    if transcript:
        print("\n" + "=" * 70)
        print("TRANSCRIPT PREVIEW (First 800 characters)")
        print("=" * 70)
        print(transcript[:800])
        print("\n...")
    else:
        print("\n[FAILED] Could not extract transcript")
