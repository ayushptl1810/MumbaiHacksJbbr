from fastapi import FastAPI, File, UploadFile, HTTPException, Form, WebSocket, WebSocketDisconnect
from typing import Optional, List, Dict, Any
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import tempfile
from pathlib import Path
import asyncio
import logging
import json
import base64
import requests
import re

from services.image_verifier import ImageVerifier
from services.video_verifier import VideoVerifier
from services.input_processor import InputProcessor
from services.text_fact_checker import TextFactChecker
from services.educational_content_generator import EducationalContentGenerator
from services.mongodb_service import MongoDBService
from services.websocket_service import connection_manager, initialize_mongodb_change_stream, cleanup_mongodb_change_stream
from utils.file_utils import save_upload_file, cleanup_temp_files
from config import config
from services.deepfake_checker import detect_audio_deepfake

app = FastAPI(
    title="Visual Verification Service",
    description="A service to verify images/videos and generate visual counter-measures",
    version="1.0.0"
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add CORS middleware
# Note: When allow_credentials=True, you cannot use allow_origins=["*"]
# Must specify exact origins
# Chrome extensions make requests from background scripts which bypass CORS,
# but we include common origins for web frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static directory for local assets (e.g., extracted frames)
import os
os.makedirs("public/frames", exist_ok=True)
app.mount("/static", StaticFiles(directory="public"), name="static")
app.mount("/frames", StaticFiles(directory="public/frames"), name="frames")


# Initialize verifiers and input processor
image_verifier = ImageVerifier()
video_verifier = VideoVerifier()
input_processor = InputProcessor()
text_fact_checker = TextFactChecker()
educational_generator = EducationalContentGenerator()

# Initialize MongoDB service
mongodb_service = None
try:
    mongodb_service = MongoDBService()
except Exception as e:
    print(f"Warning: MongoDB service initialization failed: {e}")

# Initialize MongoDB change service (will be set in startup event)
mongodb_change_service = None

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global mongodb_change_service
    try:
        mongodb_change_service = await initialize_mongodb_change_stream()
        logger.info("✅ All services initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize services: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup services on shutdown"""
    try:
        await cleanup_mongodb_change_stream()
        logger.info("🧹 All services cleaned up successfully")
    except Exception as e:
        logger.error(f"❌ Error during cleanup: {e}")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await connection_manager.connect(websocket, {"connected_at": asyncio.get_event_loop().time()})
    logger.info(f"✅ WebSocket client connected. Total connections: {len(connection_manager.active_connections)}")
    
    try:
        # Send initial greeting to confirm connectivity
        await connection_manager.send_personal_message(
            json.dumps({"type": "hello", "message": "Connected to rumours stream"}),
            websocket
        )
        while True:
            try:
                # Wait for incoming messages with a timeout
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                
                # Echo back a response (optional)
                await connection_manager.send_personal_message(
                    json.dumps({"type": "pong", "message": "Connection active"}), 
                    websocket
                )
            except asyncio.TimeoutError:
                # Send a ping to keep connection alive
                await connection_manager.send_personal_message(
                    json.dumps({"type": "ping", "message": "Keep alive"}), 
                    websocket
                )
            except Exception as e:
                logger.error(f"❌ Error in WebSocket message handling: {e}")
                break
                
    except WebSocketDisconnect:
        logger.info("🔌 WebSocket client disconnected normally")
        connection_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"❌ WebSocket error: {e}")
        connection_manager.disconnect(websocket)

@app.get("/")
async def root():
    return {"message": "Visual Verification Service is running"}

@app.post("/verify/image")
async def verify_image(
    file: Optional[UploadFile] = File(None),
    image_url: Optional[str] = Form(None),
    claim_context: str = Form("Unknown context"),
    claim_date: str = Form("Unknown date")
):
    """
    Verify a single image and generate a visual counter-measure
    """
    try:
        # Save uploaded file temporarily
        temp_file_path = None
        if file is not None:
            temp_file_path = await save_upload_file(file)
        
        # Verify image
        result = await image_verifier.verify(
            image_path=temp_file_path,
            claim_context=claim_context,
            claim_date=claim_date,
            image_url=image_url
        )
        
        # Clean up temp file
        if temp_file_path:
            cleanup_temp_files([temp_file_path])
        
        return result
            
    except Exception as e:
        # Clean up on error
        if 'temp_file_path' in locals() and temp_file_path:
            cleanup_temp_files([temp_file_path])
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/verify/video")
async def verify_video(
    file: Optional[UploadFile] = File(None),
    video_url: Optional[str] = Form(None),
    claim_context: str = Form("Unknown context"),
    claim_date: str = Form("Unknown date")
):
    """
    Verify a video and generate a visual counter-measure video
    """
    try:
        # Save uploaded file temporarily
        temp_file_path = None
        if file is not None:
            temp_file_path = await save_upload_file(file)
        
        # Verify video
        result = await video_verifier.verify(
            video_path=temp_file_path,
            claim_context=claim_context,
            claim_date=claim_date,
            video_url=video_url
        )
        
        # Clean up temp file
        if temp_file_path:
            cleanup_temp_files([temp_file_path])
        
        return result
            
    except Exception as e:
        # Clean up on error
        if 'temp_file_path' in locals() and temp_file_path:
            cleanup_temp_files([temp_file_path])
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/verify/text")
async def verify_text(
    text_input: str = Form(...),
    claim_context: str = Form("Unknown context"),
    claim_date: str = Form("Unknown date")
):
    """
    Verify a textual claim using Google's Fact Check Tools API
    """
    try:
        # Verify text claim
        result = await text_fact_checker.verify(
            text_input=text_input,
            claim_context=claim_context,
            claim_date=claim_date
        )
        
        return result
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def _extract_media_from_url(url: str) -> Optional[Dict[str, Any]]:
    """
    Use yt-dlp to extract media from a URL and determine if it's an image or video.
    
    Returns:
        Dict with "type" ("image" or "video") and "path" (local file path), or None if fails
    """
    try:
        from shutil import which
        import subprocess
        import tempfile
        
        # Resolve yt-dlp binary
        ytdlp_bin = config.YTDLP_BIN or "yt-dlp"
        found = which(ytdlp_bin) or which("yt-dlp")
        if not found:
            print("[extract_media] yt-dlp not found")
            return None
        
        # Create temp directory
        temp_dir = tempfile.mkdtemp(prefix="media_extract_")
        
        # First, get info about the media
        info_cmd = [found, url, "--dump-json", "--no-playlist"]
        result = subprocess.run(
            info_cmd,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            print(f"[extract_media] yt-dlp info failed: {result.stderr}")
            return None
        
        info = json.loads(result.stdout)
        
        # Determine media type
        ext = info.get("ext", "").lower()
        is_video = ext in ["mp4", "webm", "mkv", "avi", "mov", "flv", "m4v"]
        is_image = ext in ["jpg", "jpeg", "png", "gif", "webp", "bmp"]
        
        if not is_video and not is_image:
            # Check formats to determine type
            formats = info.get("formats", [])
            has_video_codec = any(f.get("vcodec") != "none" for f in formats)
            has_audio_codec = any(f.get("acodec") != "none" for f in formats)
            
            if has_video_codec:
                is_video = True
            elif not has_audio_codec and not has_video_codec:
                # Likely an image
                is_image = True
        
        media_type = "video" if is_video else "image"
        
        # Download the media
        output_template = os.path.join(temp_dir, f"media.%(ext)s")
        download_cmd = [
            found,
            url,
            "-o", output_template,
            "--no-playlist",
        ]
        
        # For images, prefer best quality; for videos, get best format
        if is_image:
            download_cmd.extend(["--format", "best"])
        else:
            download_cmd.extend(["--format", "best[ext=mp4]/best"])
        
        result = subprocess.run(
            download_cmd,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode != 0:
            print(f"[extract_media] yt-dlp download failed: {result.stderr}")
            return None
        
        # Find the downloaded file
        downloaded_files = [f for f in os.listdir(temp_dir) if os.path.isfile(os.path.join(temp_dir, f))]
        if not downloaded_files:
            print("[extract_media] No file downloaded")
            return None
        
        media_path = os.path.join(temp_dir, downloaded_files[0])
        
        return {
            "type": media_type,
            "path": media_path,
            "temp_dir": temp_dir  # Keep for cleanup
        }
        
    except Exception as e:
        print(f"[extract_media] Error: {e}")
        import traceback
        print(traceback.format_exc())
        return None


@app.post("/chatbot/verify")
async def chatbot_verify(
    text_input: Optional[str] = Form(None),
    files: Optional[List[UploadFile]] = File(None)
):
    """
    Chatbot-friendly endpoint that intelligently processes input and routes to appropriate verification
    """
    try:
        print(f"🔍 DEBUG: Chatbot verify endpoint called")
        print(f"🔍 DEBUG: text_input = {text_input}")
        print(f"🔍 DEBUG: files = {files}")
        print(f"🔍 DEBUG: files type = {type(files)}")
        received_files_meta: List[Dict[str, Any]] = []
        if files:
            for i, file in enumerate(files):
                print(f"🔍 DEBUG: File {i}: filename={file.filename}, content_type={file.content_type}, size={file.size}")
                try:
                    received_files_meta.append({
                        "filename": getattr(file, "filename", None),
                        "content_type": getattr(file, "content_type", None),
                        "size": getattr(file, "size", None)
                    })
                except Exception:
                    received_files_meta.append({
                        "filename": getattr(file, "filename", None),
                        "content_type": getattr(file, "content_type", None),
                        "size": None
                    })
        
        # Process input with LLM
        print(f"🔍 DEBUG: Calling input_processor.process_input()")
        processed_input = await input_processor.process_input(
            text_input=text_input,
            files=files
        )
        print(f"🔍 DEBUG: processed_input = {processed_input}")
        
        if "error" in processed_input:
            print(f"❌ DEBUG: Error in processed_input: {processed_input['error']}")
            return {"error": processed_input["error"]}
        
        verification_type = processed_input["verification_type"]
        content = processed_input["content"]
        claim_context = processed_input["claim_context"]
        claim_date = processed_input["claim_date"]
        
        print(f"🔍 DEBUG: verification_type = {verification_type}")
        print(f"🔍 DEBUG: content = {content}")
        print(f"🔍 DEBUG: claim_context = {claim_context}")
        print(f"🔍 DEBUG: claim_date = {claim_date}")
        
        results = []
        temp_files_to_cleanup = []
        
        # Handle text-only verification
        if verification_type == "text" and content.get("text"):
            print(f"🔍 DEBUG: Processing text verification with text: {content['text']}")
            result = await text_fact_checker.verify(
                text_input=content["text"],
                claim_context=claim_context,
                claim_date=claim_date
            )
            result["source"] = "text_input"
            results.append(result)
            print(f"🔍 DEBUG: Text verification result: {result}")
        
        # Process files if any
        files_list = content.get("files", [])
        print(f"🔍 DEBUG: Processing {len(files_list)} files")
        input_processor_for_audio = input_processor
        for i, file_path in enumerate(files_list):
            print(f"🔍 DEBUG: Processing file {i}: {file_path}")
            temp_files_to_cleanup.append(file_path)
            
            if verification_type == "image":
                print(f"🔍 DEBUG: Calling image_verifier.verify for file")
                result = await image_verifier.verify(
                    image_path=file_path,
                    claim_context=claim_context,
                    claim_date=claim_date
                )
            elif verification_type == "audio":
                print(f"🔍 DEBUG: Calling detect_audio_deepfake for file (AUDIO)")
                deepfake = detect_audio_deepfake(file_path)
                # Use Gemini to frame a verdict
                try:
                    gemini_prompt = f"""
You are an assistant for audio authenticity analysis.
{('User question: ' + claim_context) if claim_context else ''}
The audio has been analyzed and the result is: {'deepfake' if deepfake else 'NOT deepfake'}.
Compose a clear, friendly, 1-2 line summary verdict for the user, tailored to the above context/result (do not answer with JSON or code, just a natural response).
Avoid repeating 'deepfake detection' technical language; be concise and direct.
Do NOT mention file names or file paths in your response.
"""
                    gemini_response = input_processor_for_audio.model.generate_content(gemini_prompt)
                    ai_message = None
                    if gemini_response and hasattr(gemini_response, 'text') and gemini_response.text:
                        response_text = gemini_response.text.strip()
                        # Case 1: JSON block
                        if response_text.startswith('{') or response_text.startswith('```json'):
                            rt = response_text.strip('` ')
                            # Remove leading/trailing markdown code block marks
                            rt = re.sub(r'^```json', '', rt, flags=re.I).strip()
                            rt = re.sub(r'^```', '', rt, flags=re.I).strip()
                            rt = re.sub(r'```$', '', rt, flags=re.I).strip()
                            try:
                                import json
                                json_obj = json.loads(rt)
                                ai_message = json_obj.get('message') or ''
                                if not ai_message and 'verdict' in json_obj:
                                    # fallback: concat verdict + any explanation
                                    ai_message = f"Verdict: {json_obj['verdict']}" + (f". {json_obj.get('reasoning','')}" if json_obj.get('reasoning') else '')
                            except Exception as excjson:
                                print(f"[audio Gemini JSON extract fail] {type(excjson).__name__}: {excjson}")
                                # Fallback to the text itself
                                ai_message = response_text
                        else:
                            ai_message = response_text
                except Exception as exc:
                    print(f"[gemini audio summary error] {type(exc).__name__}: {exc}")
                    ai_message = None
                if not ai_message:
                    ai_message = (
                        "This audio is likely AI-generated." if deepfake else "This audio appears authentic and human." )
                result = {
                    "verified": not deepfake,
                    "is_deepfake": deepfake,
                    "file": file_path,
                    "message": ai_message,
                    "source": "uploaded_file"
                }
            else:  # video
                print(f"🔍 DEBUG: Calling video_verifier.verify for file")
                result = await video_verifier.verify(
                    video_path=file_path,
                    claim_context=claim_context,
                    claim_date=claim_date
                )
            
            result["source"] = "uploaded_file"
            results.append(result)
            print(f"🔍 DEBUG: File verification result: {result}")
        
        # Process URLs if any
        urls_list = content.get("urls", [])
        print(f"🔍 DEBUG: Processing {len(urls_list)} URLs")
        for i, url in enumerate(urls_list):
            print(f"🔍 DEBUG: Processing URL {i}: {url}")
            
            # STEP 1: For social media URLs, use yt-dlp to fetch the actual media first
            # This determines the REAL media type, not just what the LLM guessed
            url_lower = url.lower()
            is_social_media = any(domain in url_lower for domain in [
                'twitter.com', 'x.com', 'instagram.com', 'tiktok.com', 
                'facebook.com', 'youtube.com', 'youtu.be'
            ])
            
            extracted_media = None
            if is_social_media:
                print(f"🔍 DEBUG: Detected social media URL, extracting media with yt-dlp: {url}")
                try:
                    # Use yt-dlp to extract media and determine actual type
                    extracted_media = await _extract_media_from_url(url)
                    if extracted_media:
                        actual_type = extracted_media.get("type")  # "image" or "video"
                        media_path = extracted_media.get("path")
                        temp_dir = extracted_media.get("temp_dir")
                        
                        print(f"🔍 DEBUG: yt-dlp extracted {actual_type} from URL: {media_path}")
                        
                        # Route based on ACTUAL media type, not LLM's guess
                        if actual_type == "image":
                            result = await image_verifier.verify(
                                image_path=media_path,
                                claim_context=claim_context,
                                claim_date=claim_date
                            )
                        else:  # video
                            result = await video_verifier.verify(
                                video_path=media_path,
                                claim_context=claim_context,
                                claim_date=claim_date
                            )
                        
                        result["source"] = "url"
                        results.append(result)
                        
                        # Add to cleanup list
                        if media_path:
                            temp_files_to_cleanup.append(media_path)
                        if temp_dir:
                            temp_files_to_cleanup.append(temp_dir)
                        
                        continue  # Skip the old routing logic below
                    else:
                        print(f"⚠️ DEBUG: yt-dlp extraction returned None, falling back to direct URL")
                except Exception as e:
                    print(f"⚠️ DEBUG: Failed to extract media from URL with yt-dlp: {e}, falling back to direct URL")
                    import traceback
                    print(traceback.format_exc())
                    # Fall through to old logic
            
            # STEP 2: Fallback to old routing (for direct image/video URLs or if yt-dlp fails)
            if verification_type == "image":
                print(f"🔍 DEBUG: Calling image_verifier.verify for URL")
                result = await image_verifier.verify(
                    image_url=url,
                    claim_context=claim_context,
                    claim_date=claim_date
                )
            else:  # video
                print(f"🔍 DEBUG: Calling video_verifier.verify for URL")
                result = await video_verifier.verify(
                    video_url=url,
                    claim_context=claim_context,
                    claim_date=claim_date
                )
            
            result["source"] = "url"
            results.append(result)
            print(f"🔍 DEBUG: URL verification result: {result}")
        
        # Clean up temp files
        if temp_files_to_cleanup:
            input_processor.cleanup_temp_files(temp_files_to_cleanup)
        
        print(f"🔍 DEBUG: Total results collected: {len(results)}")
        for i, result in enumerate(results):
            print(f"🔍 DEBUG: Result {i}: {result}")
        
        # Aggregate verdict before using anywhere
        overall = _aggregate_verdicts(results)

        # Collect message/summary fields
        candidates = []
        for r in results:
            msg = (r.get("message") or r.get("summary") or "").strip()
            if msg:
                candidates.append(msg)
        best_msg = max(candidates, key=len, default="")

        # --- REFINE OUTPUT ---
        # For audio, force clear user-facing message
        verdict_is_audio = verification_type == "audio"
        if verdict_is_audio and results:
            # For batch, show the message(s) generated by Gemini/LLM for each result, joined with spacing.
            audio_msgs = [x["message"] for x in results if "message" in x and x["message"]]
            final_message = "\n\n".join(audio_msgs)
        else:
            # Final message extraction for ALL types: if best_msg is a raw JSON or code block, try extracting the `message` field.
            if not verdict_is_audio:
                raw_final = (best_msg or "").strip()
                nonjson = bool(raw_final) and not (raw_final.startswith('{') or raw_final.startswith('```'))
                extracted_message = raw_final
                if not nonjson:
                    rt = raw_final.strip('` \n')
                    rt = re.sub(r'^```json', '', rt, flags=re.I).strip()
                    rt = re.sub(r'^```', '', rt, flags=re.I).strip()
                    rt = re.sub(r'```$', '', rt, flags=re.I).strip()
                    try:
                        import json
                        json_obj = json.loads(rt)
                        extracted_message = json_obj.get('message') or ''
                        if not extracted_message and 'verdict' in json_obj:
                            extracted_message = f"Verdict: {json_obj['verdict']}" + (f". {json_obj.get('reasoning','')}" if json_obj.get('reasoning') else '')
                    except Exception as excjson:
                        print(f"[text gemini JSON extract fail] {type(excjson).__name__}: {excjson}")
                        extracted_message = raw_final
                final_message = extracted_message
                # Remove typical claim verdict phrases from start if present
                verdict_prefixes = [
                    "this claim is true:", "this claim is false:", "this claim is uncertain:", "this claim has mixed evidence:", "the claim is true:", "the claim is false:", "the claim is uncertain:", "result:",
                ]
                for prefix in verdict_prefixes:
                    if final_message.strip().lower().startswith(prefix):
                        final_message = final_message.strip()[len(prefix):].strip()
                        break
                # For stray audio check message from earlier code
                if final_message.strip().startswith("Audio deepfake detection completed"):
                    # Should not leak this to user; use generic fallback
                    final_message = "Audio deepfake detection was performed."
            else:
                final_message = (best_msg or "")
        print(f"🔍 DEBUG: Final message: {final_message}")
        print(f"🔍 DEBUG: Final verdict: {overall}")
        
        response = {
            "message": final_message,
            "verdict": overall,
            "details": {
                "results": results,
                "verification_type": verification_type,
                "claim_context": claim_context,
                "claim_date": claim_date,
                "received_files_count": len(received_files_meta),
                "received_files": received_files_meta
            }
        }
        
        print(f"🔍 DEBUG: Final response: {response}")
        return response
            
    except Exception as e:
        print(f"❌ DEBUG: Exception in chatbot_verify: {e}")
        print(f"❌ DEBUG: Exception type: {type(e).__name__}")
        import traceback
        print(f"❌ DEBUG: Traceback: {traceback.format_exc()}")
        # Clean up any temp files on error
        if 'temp_files_to_cleanup' in locals():
            input_processor.cleanup_temp_files(temp_files_to_cleanup)
        raise HTTPException(status_code=500, detail=str(e))

def _aggregate_verdicts(results: List[Dict]) -> str:
    """Aggregate individual verification results into overall verdict.

    Supports image results (with 'verdict'), video results (with details.overall_verdict), 
    and text results (with 'verdict').
    """
    if not results:
        return "no_content"
    
    normalized: List[str] = []
    for r in results:
        # Prefer explicit boolean 'verified' if present
        if "verified" in r and isinstance(r.get("verified"), bool):
            v = "true" if r.get("verified") else "false"
        else:
            v = r.get("verdict")
        if not v:
            details = r.get("details") or {}
            v = details.get("overall_verdict")
        normalized.append((v or "unknown").lower())
    
    # If any false, overall is false
    if "false" in normalized:
        return "false"
    
    # If any uncertain, overall is uncertain
    if "uncertain" in normalized:
        return "uncertain"
    
    # If all true, overall is true
    if all(v == "true" for v in normalized):
        return "true"
    
    return "mixed"

@app.get("/mongodb/recent-posts")
async def get_recent_debunk_posts(limit: int = 5):
    """
    Get recent debunk posts from MongoDB
    
    Args:
        limit: Maximum number of posts to return (default: 5)
        
    Returns:
        List of recent debunk posts
    """
    try:
        print(f"🔍 DEBUG: Endpoint called with limit={limit}")
        print(f"🔍 DEBUG: MongoDB service available: {mongodb_service is not None}")
        
        if not mongodb_service:
            print("❌ DEBUG: MongoDB service is None!")
            raise HTTPException(
                status_code=503, 
                detail="MongoDB service is not available. Check MONGO_CONNECTION_STRING environment variable."
            )
        
        print("🔍 DEBUG: Calling mongodb_service.get_recent_posts()")
        posts = mongodb_service.get_recent_posts(limit)
        print(f"🔍 DEBUG: Service returned {len(posts)} posts")
        
        if posts:
            print(f"🔍 DEBUG: First post keys: {list(posts[0].keys())}")
            print(f"🔍 DEBUG: First post _id: {posts[0].get('_id')}")
        else:
            print("⚠️ DEBUG: No posts returned from service")
        
        result = {
            "success": True,
            "count": len(posts),
            "posts": posts
        }
        
        print(f"🔍 DEBUG: Returning result with {len(posts)} posts")
        return result
        
    except Exception as e:
        print(f"❌ DEBUG: Exception in endpoint: {e}")
        print(f"🔍 DEBUG: Exception type: {type(e).__name__}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/mongodb/search-similar")
async def search_similar_rumours(
    query: str,
    similarity_threshold: float = 0.6,
    limit: int = 5
):
    """
    Search for rumours similar to the query text
    
    Args:
        query: Search query text
        similarity_threshold: Minimum similarity score (0.0 to 1.0, default: 0.6)
        limit: Maximum number of results to return (default: 5)
        
    Returns:
        List of similar rumours with similarity scores
    """
    try:
        if not mongodb_service:
            raise HTTPException(
                status_code=503,
                detail="MongoDB service is not available. Check MONGO_CONNECTION_STRING environment variable."
            )
        
        if not query or not query.strip():
            return {
                "success": True,
                "count": 0,
                "results": []
            }
        
        # Validate threshold
        similarity_threshold = max(0.0, min(1.0, similarity_threshold))
        
        results = mongodb_service.search_similar_rumours(
            query=query,
            similarity_threshold=similarity_threshold,
            limit=limit
        )
        
        return {
            "success": True,
            "count": len(results),
            "query": query,
            "similarity_threshold": similarity_threshold,
            "results": results
        }
        
    except Exception as e:
        logger.error(f"❌ Error searching similar rumours: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "visual-verification"}


@app.post("/speech-to-text")
async def speech_to_text(
    audio: UploadFile = File(...),
    language_code: str = Form("en-US")
):
    """Proxy uploaded audio to Google Speech-to-Text and return transcript.

    Accepts WEBM/OPUS, OGG/OPUS, or WAV/LINEAR16. For browser recordings via
    MediaRecorder the typical format is WEBM/OPUS which is supported by Google.
    """
    try:
        if not config.GOOGLE_API_KEY:
            raise HTTPException(status_code=500, detail="GOOGLE_API_KEY not configured")

        # Read audio bytes and base64-encode for Google API
        audio_bytes = await audio.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Empty audio payload")

        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

        # Infer encoding for common browser uploads; default to WEBM_OPUS if unknown
        content_type = (audio.content_type or "").lower()
        if "wav" in content_type or "x-wav" in content_type or "linear16" in content_type:
            encoding = "LINEAR16"
        elif "ogg" in content_type:
            encoding = "OGG_OPUS"
        else:
            encoding = "WEBM_OPUS"

        # Build request to Google Speech-to-Text v1 REST API
        endpoint = f"https://speech.googleapis.com/v1/speech:recognize?key={config.GOOGLE_API_KEY}"
        payload = {
            "config": {
                "encoding": encoding,
                "languageCode": language_code,
                # Enable auto punctuation; leave other options default to keep generalized
                "enableAutomaticPunctuation": True
            },
            "audio": {"content": audio_b64}
        }

        resp = requests.post(endpoint, json=payload, timeout=30)
        if resp.status_code != 200:
            detail = resp.text
            raise HTTPException(status_code=resp.status_code, detail=detail)

        data = resp.json()
        # Extract the best transcript
        transcript = ""
        if isinstance(data, dict):
            results = data.get("results") or []
            if results:
                alts = results[0].get("alternatives") or []
                if alts:
                    transcript = (alts[0].get("transcript") or "").strip()

        return {"transcript": transcript, "raw": data}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Educational Content API Endpoints
@app.get("/educational/modules")
async def get_educational_modules():
    """Get list of available educational modules"""
    try:
        modules_data = await educational_generator.get_modules_list()
        return modules_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/educational/modules/{module_id}")
async def get_module_content(
    module_id: str,
    difficulty_level: str = "beginner"
):
    """Get educational content for a specific module"""
    try:
        content = await educational_generator.generate_module_content(
            module_id, difficulty_level
        )
        return content
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/educational/contextual-learning")
async def get_contextual_learning(verification_result: Dict[str, Any]):
    """Generate educational content based on verification result"""
    try:
        content = await educational_generator.generate_contextual_learning(
            verification_result
        )
        return content
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/educational/clear-cache")
async def clear_educational_cache():
    """Clear all educational content from Redis cache"""
    try:
        if educational_generator.redis_client:
            # Get all educational cache keys
            keys = educational_generator.redis_client.keys("educational:*")
            if keys:
                educational_generator.redis_client.delete(*keys)
                return {"message": f"Cleared {len(keys)} cache entries", "keys": keys}
            else:
                return {"message": "No cache entries found"}
        else:
            return {"message": "Redis not available"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/educational/cache-status")
async def get_cache_status():
    """Get status of educational content cache"""
    try:
        if educational_generator.redis_client:
            keys = educational_generator.redis_client.keys("educational:*")
            cache_info = {}
            for key in keys:
                ttl = educational_generator.redis_client.ttl(key)
                cache_info[key] = {
                    "ttl": ttl,
                    "exists": ttl > 0
                }
            return {
                "redis_connected": True,
                "total_keys": len(keys),
                "cache_info": cache_info
            }
        else:
            return {"redis_connected": False, "message": "Redis not available"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from pydantic import BaseModel


# ---------- Auth endpoints (minimal implementation) ----------


class LoginRequest(BaseModel):
    email: str
    password: str

class SignupRequest(BaseModel):
    email: str
    password: str

class UserResponse(BaseModel):
    email: str
    id: Optional[str] = None

# Simple in-memory user store (replace with database in production)
users_db = {}

@app.post("/auth/signup")
async def signup(request: SignupRequest):
    """Sign up a new user"""
    if request.email in users_db:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # In production, hash the password
    users_db[request.email] = {
        "email": request.email,
        "password": request.password,  # Should be hashed
        "id": str(len(users_db) + 1)
    }
    
    return {
        "message": "User created successfully",
        "token": "mock_token_" + request.email,  # In production, use JWT
        "user": {"email": request.email, "id": users_db[request.email]["id"]}
    }

@app.post("/auth/login")
async def login(request: LoginRequest):
    """Login user"""
    if request.email not in users_db:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    user = users_db[request.email]
    if user["password"] != request.password:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    return {
        "message": "Login successful",
        "token": "mock_token_" + request.email,  # In production, use JWT
        "user": {"email": request.email, "id": user["id"]}
    }

@app.get("/auth/me")
async def get_current_user():
    """Get current user (requires authentication in production)"""
    # In production, verify JWT token from Authorization header
    return {
        "email": "demo@example.com",
        "id": "1"
    }


# ---------- Chat history endpoints ----------


class ChatSessionUpsert(BaseModel):
    session_id: Optional[str] = None
    title: Optional[str] = None
    user_id: Optional[str] = None
    anonymous_id: Optional[str] = None
    last_verdict: Optional[str] = None
    last_summary: Optional[str] = None


class ChatTurn(BaseModel):
    role: str
    content: str
    created_at: Optional[Any] = None  # Can be datetime, string, or None
    verdict: Optional[str] = None
    confidence: Optional[float] = None
    sources: Optional[Dict[str, Any]] = None
    attachments: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None


class ChatMessagesAppend(BaseModel):
    session_id: str
    user_id: Optional[str] = None
    anonymous_id: Optional[str] = None
    messages: List[ChatTurn]


@app.get("/chat/sessions")
async def list_chat_sessions(
    user_id: Optional[str] = None,
    anonymous_id: Optional[str] = None,
):
    """Return chat sessions for logged-in users only.
    
    Anonymous users will receive an empty list since their sessions are not persisted.
    """
    try:
        if not mongodb_service:
            raise HTTPException(status_code=503, detail="MongoDB service not available")

        # Only return sessions for logged-in users
        if not user_id:
            logger.info(f"⏭️ No user_id provided, returning empty sessions list")
            return {"sessions": []}

        logger.info(f"🔍 Loading chat sessions: user_id={user_id}")
        sessions = mongodb_service.get_chat_sessions(
            user_id=user_id,
            anonymous_id=None,  # Don't query by anonymous_id anymore
        )
        logger.info(f"✅ Found {len(sessions)} chat sessions")
        return {"sessions": sessions}
    except Exception as e:
        logger.error(f"❌ Error loading chat sessions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to load chat sessions: {str(e)}")


@app.post("/chat/sessions")
async def upsert_chat_session(payload: ChatSessionUpsert):
    """Create or update a chat session.

    Only saves sessions for logged-in users (user_id required).
    Anonymous sessions are not persisted to MongoDB but a session_id is still returned for UI purposes.
    """
    try:
        if not mongodb_service:
            raise HTTPException(status_code=503, detail="MongoDB service not available")

        data = payload.dict(exclude_unset=True)
        user_id = data.get("user_id")
        anonymous_id = data.get("anonymous_id")
        
        # Only persist sessions for logged-in users
        if not user_id:
            # Still return a session_id for UI purposes, but don't persist
            import uuid
            session_id = data.get("session_id") or str(uuid.uuid4())
            logger.info(f"⏭️ Skipping session persistence for anonymous user (session_id={session_id})")
            return {
                "session_id": session_id,
                "title": data.get("title", "New Chat"),
                "user_id": None,
                "anonymous_id": anonymous_id,
                "created_at": None,
                "updated_at": None,
                "persisted": False,
            }

        logger.info(f"🔍 Upserting chat session: {data}")

        # Optionally migrate anonymous history on first login
        if user_id and anonymous_id:
            try:
                migrated = mongodb_service.migrate_anonymous_sessions(
                    anonymous_id=anonymous_id, user_id=user_id
                )
                logger.info(f"✅ Migrated {migrated} anonymous sessions to user {user_id}")
            except Exception as exc:
                logger.error(f"Failed to migrate anonymous sessions: {exc}")

        session_doc = mongodb_service.upsert_chat_session(data)
        logger.info(f"✅ Created/updated session: {session_doc.get('session_id')}")
        return session_doc
    except Exception as e:
        logger.error(f"❌ Error upserting chat session: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to create/update chat session: {str(e)}")


@app.get("/chat/messages/{session_id}")
async def get_chat_messages(session_id: str):
    """Return all messages for a given chat session."""
    if not mongodb_service:
        raise HTTPException(status_code=503, detail="MongoDB service not available")

    messages = mongodb_service.get_chat_messages(session_id=session_id)
    return {"session_id": session_id, "messages": messages}


@app.post("/chat/messages")
async def append_chat_messages(payload: ChatMessagesAppend):
    """Append one or more messages to a chat session.
    
    Only saves messages for logged-in users (user_id required).
    Anonymous messages are not persisted to MongoDB.
    """
    if not mongodb_service:
        raise HTTPException(status_code=503, detail="MongoDB service not available")

    data = payload.dict()
    user_id = data.get("user_id")
    
    # Only persist messages for logged-in users
    if not user_id:
        logger.info(f"⏭️ Skipping message persistence for anonymous user (session_id={data['session_id']})")
        return {"inserted": 0, "message": "Messages not persisted for anonymous users"}

    inserted = mongodb_service.append_chat_messages(
        session_id=data["session_id"],
        messages=[m for m in data["messages"]],
        user_id=user_id,
        anonymous_id=data.get("anonymous_id"),
    )
    logger.info(f"✅ Persisted {inserted} messages for user {user_id}")
    return {"inserted": inserted}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=config.SERVICE_PORT)