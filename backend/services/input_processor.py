import os
import re
import json
from typing import Dict, List, Optional, Union, Tuple
import google.generativeai as genai
import tempfile
from config import config

class InputProcessor:
    """
    Intelligent input processor that converts chatbot input into structured verification requests
    """
    
    def __init__(self):
        # Configure Gemini
        genai.configure(api_key=config.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(
            config.GEMINI_MODEL,
            generation_config=genai.types.GenerationConfig(
                temperature=config.GEMINI_TEMPERATURE,
                top_p=config.GEMINI_TOP_P,
                max_output_tokens=config.GEMINI_MAX_TOKENS
            )
        )
        
        self.system_prompt = """You are an intelligent input processor for a visual verification service. 
        
Your task is to analyze user input and extract:
1. Image/video/audio content (files, URLs, or descriptions)
2. Claim context (what the user is claiming)
3. Claim date (when the claim was made)
4. Type of verification needed (image, video, audio, or text)

Return a JSON response with this structure:
{
    "verification_type": "image" or "video" or "audio" or "text",
    "content": {
        "files": ["list of file paths if files provided"],
        "urls": ["list of image/video/audio URLs"],
        "descriptions": ["list of text descriptions"],
        "text": "the text claim to verify (if verification_type is text)"
    },
    "claim_context": "extracted or inferred claim context",
    "claim_date": "extracted or inferred date"
}

Rules:
- If multiple images/videos/audio files are mentioned, separate them clearly
- Extract URLs from text using regex patterns
- Infer context from surrounding text if not explicitly stated
- If no date is mentioned leave it blank
- Handle mixed content types appropriately"""

    async def process_input(
        self, 
        text_input: Optional[str] = None, 
        files: Optional[List] = None
    ) -> Dict:
        """
        Process chatbot input and return structured verification request
        """
        try:
            print(f"🔍 DEBUG: InputProcessor.process_input called")
            print(f"🔍 DEBUG: text_input = {text_input}")
            print(f"🔍 DEBUG: files = {files}")
            print(f"🔍 DEBUG: files type = {type(files)}")
            
            # Prepare input for LLM analysis
            print(f"🔍 DEBUG: Preparing input text for LLM analysis")
            input_text = self._prepare_input_text(text_input, files)
            print(f"🔍 DEBUG: Prepared input_text = {input_text}")
            
            # Get LLM analysis
            print(f"🔍 DEBUG: Calling LLM analysis")
            llm_response = await self._analyze_with_llm(input_text)
            print(f"🔍 DEBUG: LLM response = {llm_response}")
            
            # Parse and validate LLM response
            print(f"🔍 DEBUG: Parsing LLM response")
            parsed_response = self._parse_llm_response(llm_response)
            print(f"🔍 DEBUG: Parsed response = {parsed_response}")
            
            # Post-process and enhance the response
            print(f"🔍 DEBUG: Post-processing response")
            final_response = await self._post_process_response(parsed_response, files)

            # PATCH: If verification_type is 'video' but all files have audio extensions, reassign to 'audio'
            audio_exts = ['.mp3', '.wav', '.ogg', '.flac', '.m4a']
            content_files = final_response.get('content', {}).get('files', [])
            if (
                final_response.get('verification_type') == 'video' and
                content_files and
                all(any(f.lower().endswith(e) for e in audio_exts) for f in content_files)
            ):
                print(f"🔍 PATCH: Rewriting 'verification_type' from 'video' to 'audio' (all files are audio)")
                final_response['verification_type'] = 'audio'
            print(f"🔍 DEBUG: Final response = {final_response}")
            return final_response
            
        except Exception as e:
            print(f"❌ DEBUG: Exception in InputProcessor.process_input: {e}")
            print(f"❌ DEBUG: Exception type: {type(e).__name__}")
            import traceback
            print(f"❌ DEBUG: Traceback: {traceback.format_exc()}")
            return {
                "error": f"Failed to process input: {str(e)}",
                "verification_type": "unknown",
                "content": {"files": [], "urls": [], "descriptions": []},
                "claim_context": "Unknown context",
                "claim_date": "Unknown date",
            }

    def _prepare_input_text(self, text_input: Optional[str], files: Optional[List]) -> str:
        """Prepare input text for LLM analysis"""
        print(f"🔍 DEBUG: _prepare_input_text called with text_input={text_input}, files={files}")
        input_parts = []
        
        if text_input:
            input_parts.append(f"Text input: {text_input}")
            print(f"🔍 DEBUG: Added text input: {text_input}")
        
        if files:
            file_info = []
            for i, file in enumerate(files):
                file_info.append(f"File {i+1}: {file.filename} ({file.content_type})")
                print(f"🔍 DEBUG: Added file {i+1}: {file.filename} ({file.content_type})")
            input_parts.append(f"Files provided: {'; '.join(file_info)}")
        
        if not input_parts:
            input_parts.append("No text or files provided")
            print(f"🔍 DEBUG: No input parts, using default message")
        
        result = "\n".join(input_parts)
        print(f"🔍 DEBUG: Final prepared input text: {result}")
        return result

    async def _analyze_with_llm(self, input_text: str) -> str:
        """Use Gemini to analyze the input"""
        try:
            print(f"🔍 DEBUG: _analyze_with_llm called with input_text: {input_text}")
            prompt = f"{self.system_prompt}\n\nUser input: {input_text}"
            print(f"🔍 DEBUG: Generated prompt: {prompt}")
            response = self.model.generate_content(prompt)
            print(f"🔍 DEBUG: LLM response text: {response.text}")
            return response.text
        except Exception as e:
            print(f"❌ DEBUG: LLM analysis failed: {e}")
            print(f"🔍 DEBUG: Falling back to rule-based parsing")
            # Fallback to rule-based parsing if LLM fails
            return self._fallback_parsing(input_text)

    def _fallback_parsing(self, input_text: str) -> str:
        """Fallback parsing when LLM is unavailable"""
        print(f"🔍 DEBUG: _fallback_parsing called with input_text: {input_text}")
        
        # Extract URLs using regex
        url_pattern = r'https?://[^\s<>"]+|www\.[^\s<>"]+'
        urls = re.findall(url_pattern, input_text)
        print(f"🔍 DEBUG: Extracted URLs: {urls}")
        
        # Simple content type detection
        verification_type = "text"  # default for text-only queries
        
        # Check for video platform URLs first
        video_platforms = [
            'instagram.com/reels/', 'instagram.com/p/', 'instagram.com/tv/',
            'youtube.com/watch', 'youtu.be/', 'youtube.com/shorts/',
            'tiktok.com/', 'vm.tiktok.com/',
            'twitter.com/', 'x.com/', 't.co/',
            'facebook.com/', 'fb.watch/',
            'vimeo.com/', 'twitch.tv/', 'dailymotion.com/',
            'imgur.com/', 'soundcloud.com/', 'mixcloud.com/',
            'lbry.tv/', 'odysee.com/', 't.me/'
        ]
        
        # Check for image platform URLs
        image_platforms = [
            'instagram.com/p/', 'imgur.com/', 'flickr.com/',
            'pinterest.com/', 'unsplash.com/', 'pexels.com/'
        ]
        
        # Check for direct file extensions
        if any(ext in input_text.lower() for ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm', 'video']):
            verification_type = "video"
        elif any(ext in input_text.lower() for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp', 'image', 'photo', 'picture']):
            verification_type = "image"
        elif any(ext in input_text.lower() for ext in ['.mp3', '.wav', '.ogg', '.flac', '.m4a', 'audio']):
            verification_type = "audio"
        # Check for video platform URLs
        elif any(platform in input_text.lower() for platform in video_platforms):
            verification_type = "video"
        # Check for image platform URLs
        elif any(platform in input_text.lower() for platform in image_platforms):
            verification_type = "image"
            
        print(f"🔍 DEBUG: Detected verification_type: {verification_type}")
        
        # Extract date patterns
        date_pattern = r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}'
        dates = re.findall(date_pattern, input_text)
        claim_date = dates[0] if dates else "Unknown date"
        print(f"🔍 DEBUG: Extracted dates: {dates}, using: {claim_date}")
        
        # Clean up the input text for better processing
        clean_text = input_text.replace("Text input: ", "").strip()
        
        result = {
            "verification_type": verification_type,
            "content": {
                "files": [],
                "urls": urls,
                "descriptions": [clean_text],
                "text": clean_text if verification_type == "text" else None
            },
            "claim_context": clean_text,
            "claim_date": claim_date,
        }
        print(f"🔍 DEBUG: Fallback parsing result: {result}")
        return json.dumps(result)

    def _parse_llm_response(self, llm_response: str) -> Dict:
        """Parse and validate LLM response"""
        try:
            print(f"🔍 DEBUG: _parse_llm_response called with llm_response: {llm_response}")
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
            if json_match:
                print(f"🔍 DEBUG: Found JSON match: {json_match.group()}")
                parsed = json.loads(json_match.group())
                print(f"🔍 DEBUG: Parsed JSON: {parsed}")
            else:
                print(f"❌ DEBUG: No JSON found in response")
                raise ValueError("No JSON found in response")
            
            # Validate required fields
            required_fields = ["verification_type", "content", "claim_context", "claim_date"]
            for field in required_fields:
                if field not in parsed:
                    print(f"❌ DEBUG: Missing required field: {field}")
                    raise ValueError(f"Missing required field: {field}")
            
            print(f"🔍 DEBUG: Successfully parsed and validated response")
            return parsed
            
        except Exception as e:
            print(f"❌ DEBUG: Failed to parse LLM response: {e}")
            print(f"🔍 DEBUG: Returning safe defaults")
            # Return safe defaults if parsing fails
            return {
                "verification_type": "image",
                "content": {"files": [], "urls": [], "descriptions": []},
                "claim_context": "Unknown context",
                "claim_date": "Unknown date",
            }

    async def _post_process_response(self, parsed_response: Dict, files: Optional[List]) -> Dict:
        """Post-process the parsed response and add file information"""
        print(f"🔍 DEBUG: _post_process_response called with parsed_response: {parsed_response}, files: {files}")
        
        # Add actual file information if files were provided
        if files:
            print(f"🔍 DEBUG: Processing {len(files)} files")
            file_paths = []
            for i, file in enumerate(files):
                print(f"🔍 DEBUG: Saving file {i}: {file.filename}")
                # Save file temporarily and get path
                temp_path = await self._save_temp_file(file)
                if temp_path:
                    file_paths.append(temp_path)
                    print(f"🔍 DEBUG: Saved file {i} to: {temp_path}")
                else:
                    print(f"❌ DEBUG: Failed to save file {i}")
            
            parsed_response["content"]["files"] = file_paths
            print(f"🔍 DEBUG: Updated files list: {file_paths}")
        else:
            print(f"🔍 DEBUG: No files to process")
        
        print(f"🔍 DEBUG: Final post-processed response: {parsed_response}")
        return parsed_response

    async def _save_temp_file(self, file) -> Optional[str]:
        """Save uploaded file temporarily and return path"""
        try:
            print(f"🔍 DEBUG: _save_temp_file called for file: {file.filename}")
            # Create temp file
            import os
            suffix = os.path.splitext(file.filename)[1] if file.filename else ""
            print(f"🔍 DEBUG: Using suffix: {suffix}")
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                content = await file.read()
                print(f"🔍 DEBUG: Read {len(content)} bytes from file")
                temp_file.write(content)
                temp_path = temp_file.name
                print(f"🔍 DEBUG: Saved temp file to: {temp_path}")
                return temp_path
        except Exception as e:
            print(f"❌ DEBUG: Failed to save temp file: {e}")
            return None

    def cleanup_temp_files(self, file_paths: List[str]):
        """Clean up temporary files"""
        for path in file_paths:
            try:
                if os.path.exists(path):
                    os.unlink(path)
            except Exception as e:
                print(f"Failed to cleanup temp file {path}: {e}")
