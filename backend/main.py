from fastapi import FastAPI, File, UploadFile, HTTPException, Form, WebSocket, WebSocketDisconnect, Request
from typing import Optional, List, Dict, Any
from fastapi.responses import FileResponse, JSONResponse
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
from services.razorpay_service import RazorpayService
import razorpay.errors
from utils.file_utils import save_upload_file, cleanup_temp_files
from config import config
from services.deepfake_checker import detect_audio_deepfake
from services.youtube_caption import get_youtube_transcript_ytdlp
import google.generativeai as genai

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


# ---------- Tier configuration ----------

# Public-facing tiers used across the product
NORMALIZED_TIERS = ("Free", "Plus", "Pro")

# Map stored subscription_tier / plan_name values to normalized tiers.
# This keeps backward compatibility with any existing users whose tier
# might still be stored as \"Pro\" or \"Enterprise\".
SUBSCRIPTION_TIER_MAPPING = {
    "free": "Free",
    "plus": "Plus",
    "pro": "Plus",          # legacy Pro maps to Plus
    "enterprise": "Pro",    # highest tier maps to Pro
}

# Central limits per tier so they can be tuned in one place.
# These values are intentionally conservative to protect API costs.
TIER_LIMITS = {
    "Free": {
        "daily_verifications": 5,
        "monthly_verifications": 25,
        "max_chat_sessions": 1,
        "max_messages_per_session": 10,
    },
    "Plus": {
        "daily_verifications": 10,
        "monthly_verifications": 50,
        "max_chat_sessions": 5,
        "max_messages_per_session": 50,
    },
    "Pro": {
        "daily_verifications": 25,
        "monthly_verifications": 200,
        "max_chat_sessions": 20,
        "max_messages_per_session": 200,
    },
}


def get_normalized_tier(raw_tier: str | None) -> str:
    """
    Normalize any stored subscription_tier / plan_name to one of
    the public-facing tiers: Free, Plus, Pro.
    """
    if not raw_tier:
        return "Free"
    key = str(raw_tier).strip().lower()
    return SUBSCRIPTION_TIER_MAPPING.get(key, "Free")


def get_tier_limits(raw_tier: str | None) -> dict:
    """
    Return the limits dict for a given stored tier value.
    """
    normalized = get_normalized_tier(raw_tier)
    return TIER_LIMITS.get(normalized, TIER_LIMITS["Free"])


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

# Initialize Razorpay service
razorpay_service = None
try:
    razorpay_service = RazorpayService()
except Exception as e:
    print(f"Warning: Razorpay service initialization failed: {e}")

# Initialize MongoDB change service (will be set in startup event)
mongodb_change_service = None

async def initialize_subscription_plans():
    """Initialize subscription plans in Razorpay if they don't exist"""
    if not razorpay_service or not razorpay_service.client:
        logger.warning("⚠️ Razorpay service not available. Skipping plan initialization.")
        return
    
    # First, test Razorpay connection by trying to fetch account details or make a simple API call
    try:
        # Try to verify credentials work by attempting a simple operation
        # We'll skip listing plans if it fails and just try to create
        logger.info("🔍 Testing Razorpay API connection...")
    except Exception as e:
        logger.error(f"❌ Razorpay API connection test failed: {e}")
        logger.warning("⚠️ Skipping plan initialization due to API connection issues")
        return
    
    try:
        # Try to list existing plans, but don't fail if it errors
        existing_plan_names = set()
        try:
            existing_plans = razorpay_service.list_plans(count=100)
            if existing_plans and existing_plans.get("items"):
                existing_plan_names = {
                    p.get("item", {}).get("name") 
                    for p in existing_plans.get("items", [])
                    if p.get("item", {}).get("name")
                }
                logger.info(f"📋 Found {len(existing_plan_names)} existing plans")
        except Exception as list_error:
            error_msg = str(list_error).lower()
            if "not found" in error_msg or "404" in error_msg:
                logger.info("ℹ️ No existing plans found (this is normal for new accounts)")
            else:
                logger.warning(f"⚠️ Could not list existing plans: {list_error}")
            # Continue anyway - we'll try to create plans and handle duplicates
        
        plans_to_create = [
            {
                "name": "Plan 1",
                "amount": 100,  # 1 INR in paise
                "currency": "INR",
                "interval": 1,
                "period": "monthly",
                "description": "Plan 1 - Monthly Subscription (1 INR)"
            },
            {
                "name": "Plan 2",
                "amount": 200,  # 2 INR in paise
                "currency": "INR",
                "interval": 1,
                "period": "monthly",
                "description": "Plan 2 - Monthly Subscription (2 INR)"
            },
            {
                "name": "Plan 3",
                "amount": 300,  # 3 INR in paise
                "currency": "INR",
                "interval": 1,
                "period": "monthly",
                "description": "Plan 3 - Monthly Subscription (3 INR)"
            }
        ]
        
        created_count = 0
        skipped_count = 0
        error_count = 0
        
        for plan_data in plans_to_create:
            plan_name = plan_data["name"]
            
            # Check if plan already exists
            if plan_name in existing_plan_names:
                logger.info(f"⏭️ Plan {plan_name} already exists, skipping")
                skipped_count += 1
                continue
            
            try:
                logger.info(f"🔄 Creating plan: {plan_name}...")
                plan = razorpay_service.create_plan(**plan_data)
                logger.info(f"✅ Created subscription plan: {plan_name} (ID: {plan.get('id')})")
                created_count += 1
            except razorpay.errors.BadRequestError as e:
                error_msg = str(e).lower()
                # Check if error is due to plan already existing (duplicate)
                if "already exists" in error_msg or "duplicate" in error_msg:
                    logger.info(f"⏭️ Plan {plan_name} already exists (detected during creation), skipping")
                    skipped_count += 1
                else:
                    logger.error(f"❌ BadRequestError creating plan {plan_name}: {e}")
                    error_count += 1
            except Exception as e:
                error_msg = str(e).lower()
                # Check if error is due to plan already existing (duplicate)
                if "already exists" in error_msg or "duplicate" in error_msg:
                    logger.info(f"⏭️ Plan {plan_name} already exists (detected during creation), skipping")
                    skipped_count += 1
                elif "not found" in error_msg or "404" in error_msg:
                    logger.error(f"❌ API endpoint not found for plan {plan_name}. Check Razorpay credentials and API access.")
                    logger.error(f"   Error details: {e}")
                    error_count += 1
                else:
                    logger.error(f"❌ Failed to create plan {plan_name}: {e}")
                    error_count += 1
        
        if created_count > 0:
            logger.info(f"✅ Successfully created {created_count} subscription plans")
        if skipped_count > 0:
            logger.info(f"⏭️ Skipped {skipped_count} plans (already exist)")
        if error_count > 0:
            logger.warning(f"⚠️ {error_count} plans failed to create. Check Razorpay credentials and API permissions.")
        if created_count == 0 and skipped_count == 0 and error_count > 0:
            logger.error("❌ All plan creation attempts failed. Please verify:")
            logger.error("   1. RAZORPAY_ID and RAZORPAY_KEY are correct")
            logger.error("   2. API keys have subscription/plan creation permissions")
            logger.error("   3. Razorpay account has subscriptions feature enabled")
            
    except Exception as e:
        logger.error(f"❌ Failed to initialize subscription plans: {e}")
        import traceback
        logger.error(traceback.format_exc())

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global mongodb_change_service
    try:
        mongodb_change_service = await initialize_mongodb_change_stream()
        # Initialize subscription plans
        await initialize_subscription_plans()
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


def _is_youtube_url(url: str) -> bool:
    """Check if URL is a YouTube URL"""
    url_lower = url.lower()
    youtube_domains = ['youtube.com', 'youtu.be', 'www.youtube.com', 'www.youtu.be', 'm.youtube.com']
    return any(domain in url_lower for domain in youtube_domains)


async def _generate_claims_summary(claim_results: List[Dict[str, Any]], gemini_model) -> str:
    """Generate a comprehensive summary of all claim verification results using Gemini"""
    try:
        # Prepare claims data for Gemini
        claims_data = []
        for i, result in enumerate(claim_results, 1):
            claims_data.append({
                "number": i,
                "claim": result.get("claim_text", ""),
                "verdict": result.get("verdict", "uncertain"),
                "explanation": result.get("message", "No explanation available")
            })
        
        prompt = f"""You are a fact-checking summary writer. Based on the following verified claims from a YouTube video, create a comprehensive, user-friendly summary.

CLAIM VERIFICATION RESULTS:
{json.dumps(claims_data, indent=2)}

Your task is to create a clear, concise summary that:
1. Lists each claim with its verdict (TRUE/FALSE/MIXED/UNCERTAIN)
2. Explains WHY each claim is true or false in simple terms
3. Highlights the most important findings
4. Provides an overall assessment of the video's factual accuracy

Format your response as a well-structured summary that is easy to read. Use clear sections and bullet points where appropriate.

IMPORTANT: 
- Be concise but thorough
- Explain the reasoning for each verdict
- Focus on the most significant false or misleading claims
- Keep the tone professional and informative
- Do NOT use markdown formatting, just plain text with clear structure

Return ONLY the summary text, no JSON or code blocks."""
        
        response = gemini_model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Clean up response if needed
        if response_text.startswith('```'):
            response_text = re.sub(r'^```[a-z]*\n?', '', response_text, flags=re.IGNORECASE)
            response_text = re.sub(r'```$', '', response_text, flags=re.IGNORECASE).strip()
        
        print(f"✅ Generated comprehensive summary")
        return response_text
        
    except Exception as e:
        print(f"❌ Error generating summary with Gemini: {e}")
        import traceback
        print(traceback.format_exc())
        # Fallback to simple concatenation
        summary_parts = []
        summary_parts.append(f"Analyzed {len(claim_results)} controversial claim(s) from the video transcript:\n")
        
        for i, result in enumerate(claim_results, 1):
            claim_text = result.get("claim_text", "")
            verdict = result.get("verdict", "uncertain")
            message = result.get("message", "No explanation available")
            
            claim_display = claim_text[:150] + "..." if len(claim_text) > 150 else claim_text
            
            verdict_label = {
                "true": "✅ TRUE",
                "false": "❌ FALSE",
                "mixed": "⚠️ MIXED",
                "uncertain": "❓ UNCERTAIN",
                "error": "⚠️ ERROR"
            }.get(verdict, "❓ UNCERTAIN")
            
            summary_parts.append(f"\n{i}. {verdict_label}: {claim_display}")
            summary_parts.append(f"   Explanation: {message}")
        
        return "\n".join(summary_parts)


async def _extract_claims_from_captions(captions: str, gemini_model) -> List[str]:
    """Extract top 5 controversial claims from video captions using Gemini"""
    try:
        prompt = f"""You are a fact-checking assistant. Analyze the following video transcript and extract the TOP 5 MOST CONTROVERSIAL and verifiable claims that were mentioned in the video.

VIDEO TRANSCRIPT:
{captions}

Your task is to identify the 5 MOST controversial, factual claims that can be verified. Prioritize:
- Claims about events, statistics, or facts that are controversial or disputed
- Claims about people, organizations, or institutions that are potentially misleading
- Claims that are specific enough to be fact-checked and are likely to be false or disputed
- Claims that have significant impact or are widely discussed

Ignore:
- General opinions or subjective statements
- Questions or hypothetical scenarios
- Vague statements without specific claims
- Small talk or filler content

IMPORTANT: Return EXACTLY 5 claims (or fewer if the video doesn't contain 5 verifiable controversial claims). Rank them by controversy/importance.

Return ONLY a JSON object in this exact format:
{{
    "claims": [
        "Claim 1 text here (most controversial)",
        "Claim 2 text here",
        "Claim 3 text here",
        "Claim 4 text here",
        "Claim 5 text here"
    ]
}}

Return ONLY the JSON object, no other text or explanation."""
        
        response = gemini_model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Clean up response if needed
        if response_text.startswith('```json'):
            response_text = response_text.replace('```json', '').replace('```', '').strip()
        elif response_text.startswith('```'):
            response_text = response_text.replace('```', '').strip()
        
        # Parse JSON response
        parsed = json.loads(response_text)
        claims = parsed.get("claims", [])
        
        # Filter out empty claims and limit to 5
        claims = [c.strip() for c in claims if c and c.strip()][:5]
        
        print(f"✅ Extracted {len(claims)} claims from video captions")
        return claims
        
    except Exception as e:
        print(f"❌ Error extracting claims from captions: {e}")
        import traceback
        print(traceback.format_exc())
        return []


async def _verify_youtube_video(url: str, claim_context: str, claim_date: str) -> Dict[str, Any]:
    """Verify a YouTube video by extracting captions, extracting claims, and verifying each claim"""
    import tempfile
    import asyncio
    
    try:
        print(f"🎥 Starting YouTube video verification for: {url}")
        
        # Step 1: Extract captions
        print(f"📝 Extracting captions from YouTube video...")
        # Create a temporary file for the transcript output
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
            temp_output_file = temp_file.name
        
        # Run the synchronous function in an executor to avoid blocking
        loop = asyncio.get_event_loop()
        captions = await loop.run_in_executor(
            None, 
            get_youtube_transcript_ytdlp, 
            url, 
            temp_output_file
        )
        
        # Clean up the temporary output file if it was created
        try:
            if os.path.exists(temp_output_file):
                os.unlink(temp_output_file)
        except Exception as cleanup_error:
            print(f"⚠️ Warning: Could not clean up temp file {temp_output_file}: {cleanup_error}")
        
        if not captions:
            return {
                "verified": False,
                "verdict": "error",
                "message": "Could not extract captions from the YouTube video. The video may not have captions available.",
                "details": {
                    "video_url": url,
                    "error": "Caption extraction failed"
                },
                "source": "youtube_url"
            }
        
        print(f"✅ Extracted {len(captions)} characters of captions")
        
        # Step 2: Extract claims using Gemini
        print(f"🔍 Extracting controversial claims from captions...")
        genai.configure(api_key=config.GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel(config.GEMINI_MODEL)
        
        claims = await _extract_claims_from_captions(captions, gemini_model)
        
        if not claims:
            return {
                "verified": False,
                "verdict": "uncertain",
                "message": "No verifiable claims were found in the video transcript. The video may contain only opinions, questions, or non-factual content.",
                "details": {
                    "video_url": url,
                    "captions_length": len(captions),
                    "claims_extracted": 0
                },
                "source": "youtube_url"
            }
        
        print(f"✅ Extracted {len(claims)} claims, starting verification...")
        
        # Step 3: Verify each claim
        claim_results = []
        for i, claim in enumerate(claims, 1):
            print(f"🔍 Verifying claim {i}/{len(claims)}: {claim[:100]}...")
            try:
                verification_result = await text_fact_checker.verify(
                    text_input=claim,
                    claim_context=f"Claim from YouTube video: {url}",
                    claim_date=claim_date
                )
                verification_result["claim_text"] = claim
                verification_result["claim_index"] = i
                claim_results.append(verification_result)
            except Exception as e:
                print(f"❌ Error verifying claim {i}: {e}")
                claim_results.append({
                    "claim_text": claim,
                    "claim_index": i,
                    "verified": False,
                    "verdict": "error",
                    "message": f"Error during verification: {str(e)}"
                })
        
        # Step 4: Combine results
        print(f"📊 Combining {len(claim_results)} claim verification results...")
        
        # Aggregate verdicts
        verdicts = [r.get("verdict", "uncertain") for r in claim_results]
        true_count = verdicts.count("true")
        false_count = verdicts.count("false")
        uncertain_count = verdicts.count("uncertain")
        mixed_count = verdicts.count("mixed")
        error_count = verdicts.count("error")
        
        # Determine overall verdict
        if false_count > 0:
            overall_verdict = "false"
            verified = False
        elif true_count > 0 and false_count == 0:
            overall_verdict = "true"
            verified = True
        elif mixed_count > 0:
            overall_verdict = "mixed"
            verified = False
        elif uncertain_count > 0:
            overall_verdict = "uncertain"
            verified = False
        else:
            overall_verdict = "error"
            verified = False
        
        # Step 5: Generate comprehensive summary using Gemini
        print(f"📝 Generating comprehensive summary with Gemini...")
        combined_message = await _generate_claims_summary(claim_results, gemini_model)
        
        return {
            "verified": verified,
            "verdict": overall_verdict,
            "message": combined_message,
            "details": {
                "video_url": url,
                "captions_length": len(captions),
                "total_claims": len(claims),
                "claims_verified": true_count,
                "claims_false": false_count,
                "claims_mixed": mixed_count,
                "claims_uncertain": uncertain_count,
                "claims_error": error_count,
                "claim_results": claim_results
            },
            "source": "youtube_url"
        }
        
    except Exception as e:
        print(f"❌ Error verifying YouTube video: {e}")
        import traceback
        print(traceback.format_exc())
        return {
            "verified": False,
            "verdict": "error",
            "message": f"Error processing YouTube video: {str(e)}",
            "details": {
                "video_url": url,
                "error": str(e)
            },
            "source": "youtube_url"
        }


@app.post("/chatbot/verify")
async def chatbot_verify(
    request: Request,
    text_input: Optional[str] = Form(None),
    files: Optional[List[UploadFile]] = File(None),
    anonymous_id: Optional[str] = Form(None),
    user_id: Optional[str] = Form(None),
):
    """
    Chatbot-friendly endpoint that intelligently processes input and routes to appropriate verification
    """
    try:
        print(f"🔍 DEBUG: Chatbot verify endpoint called")
        print(f"🔍 DEBUG: text_input = {text_input}")
        print(f"🔍 DEBUG: files = {files}")
        print(f"🔍 DEBUG: files type = {type(files)}")
        print(f"🔍 DEBUG: anonymous_id = {anonymous_id}")
        print(f"🔍 DEBUG: user_id = {user_id}")

        # Determine logical user key and tier for rate limiting
        user_doc = None
        raw_tier = "Free"
        if user_id and mongodb_service:
            try:
                user_doc = mongodb_service.get_user_by_id(user_id)
            except Exception as e:
                logger.warning(
                    f"⚠️ Failed to load user {user_id} for tier resolution: {e}"
                )

        if user_doc:
            raw_tier = user_doc.get("subscription_tier") or "Free"
        else:
            raw_tier = "Free"

        limits = get_tier_limits(raw_tier)
        key_host = getattr(request.client, "host", "unknown")
        key = user_id or anonymous_id or f"ip:{key_host}"

        if mongodb_service:
            usage_info = mongodb_service.increment_usage_and_check_limits(
                key=key,
                feature="verification",
                daily_limit=limits.get("daily_verifications"),
                monthly_limit=limits.get("monthly_verifications"),
            )
        else:
            usage_info = {
                "allowed": True,
                "tier_limits": {
                    "daily": limits.get("daily_verifications"),
                    "monthly": limits.get("monthly_verifications"),
                },
            }

        if not usage_info.get("allowed", True):
            normalized_tier = get_normalized_tier(raw_tier)
            return JSONResponse(
                status_code=429,
                content={
                    "error": "verification_limit_reached",
                    "tier": normalized_tier,
                    "key": key,
                    "limits": usage_info.get("tier_limits"),
                    "usage": {
                        "daily": usage_info.get("daily"),
                        "monthly": usage_info.get("monthly"),
                    },
                },
            )
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
            
            # STEP 0: Check if this is a YouTube URL - handle specially
            if _is_youtube_url(url):
                print(f"🎥 DEBUG: Detected YouTube URL, using caption-based verification: {url}")
                try:
                    result = await _verify_youtube_video(url, claim_context, claim_date)
                    results.append(result)
                    print(f"🔍 DEBUG: YouTube verification result: {result}")
                    continue  # Skip the rest of the URL processing
                except Exception as e:
                    print(f"❌ DEBUG: YouTube verification failed: {e}")
                    import traceback
                    print(traceback.format_exc())
                    # Fall through to regular video processing as fallback
            
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


# Educational Content API Endpoints - Now fetching from MongoDB weekly_posts
@app.get("/educational/modules")
async def get_educational_modules():
    """Get list of available educational modules from MongoDB weekly_posts"""
    try:
        if not mongodb_service:
            raise HTTPException(status_code=503, detail="MongoDB service not available")
        
        modules_list = mongodb_service.get_educational_modules_list()
        response_data = {
            "modules": modules_list,
            "total": len(modules_list)
        }
        # Return with no-cache headers to prevent stale cache in production
        return JSONResponse(
            content=response_data,
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get educational modules: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/educational/modules/{module_id}")
async def get_module_content(
    module_id: str,
    difficulty_level: str = "beginner"  # Kept for backward compatibility but not used
):
    """Get educational content for a specific module from MongoDB weekly_posts"""
    try:
        if not mongodb_service:
            raise HTTPException(status_code=503, detail="MongoDB service not available")
        
        content = mongodb_service.get_educational_module_by_id(module_id)
        if not content:
            raise HTTPException(status_code=404, detail=f"Module '{module_id}' not found")
        
        # Return with no-cache headers to prevent stale cache in production
        return JSONResponse(
            content=content,
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get module content: {e}")
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
    """
    Clear all educational content from Redis cache.
    
    Note: The /educational/modules endpoints now use no-cache headers
    to prevent browser/CDN caching. This endpoint is mainly for clearing
    any legacy Redis cache entries.
    """
    try:
        if educational_generator.redis_client:
            # Get all educational cache keys
            keys = educational_generator.redis_client.keys("educational:*")
            if keys:
                educational_generator.redis_client.delete(*keys)
                return {
                    "message": f"Cleared {len(keys)} cache entries",
                    "keys": keys,
                    "note": "Educational endpoints use no-cache headers to prevent stale data"
                }
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
    name: str
    email: str
    password: str
    phone_number: Optional[str] = None
    age: Optional[int] = None
    domain_preferences: Optional[List[str]] = []

class UserResponse(BaseModel):
    email: str
    id: Optional[str] = None

@app.post("/auth/signup")
async def signup(request: SignupRequest):
    """Sign up a new user"""
    if not mongodb_service:
        raise HTTPException(status_code=503, detail="MongoDB service not available")
    
    try:
        # Hash password (in production, use bcrypt or similar)
        import hashlib
        password_hash = hashlib.sha256(request.password.encode()).hexdigest()
        
        user_data = {
            "name": request.name,
            "email": request.email,
            "password": password_hash,
            "phone_number": request.phone_number,
            "age": request.age,
            "domain_preferences": request.domain_preferences or [],
            "created_at": None,  # Will be set by MongoDB service
            "updated_at": None,
        }
        
        user = mongodb_service.create_user(user_data)
        
        # Generate token (in production, use JWT)
        token = f"mock_token_{request.email}"
        
        return {
            "message": "User created successfully",
            "token": token,
            "user": {
                "name": user.get("name"),
                "email": user["email"],
                "id": user["id"],
                "phone_number": user.get("phone_number"),
                "age": user.get("age"),
                "domain_preferences": user.get("domain_preferences", [])
            }
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Signup error: {e}")
        raise HTTPException(status_code=500, detail="Failed to create user")

@app.post("/auth/login")
async def login(request: LoginRequest):
    """Login user"""
    if not mongodb_service:
        raise HTTPException(status_code=503, detail="MongoDB service not available")
    
    try:
        user = mongodb_service.get_user_by_email(request.email)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid email or password")
        
        # Verify password (in production, use bcrypt or similar)
        import hashlib
        password_hash = hashlib.sha256(request.password.encode()).hexdigest()
        
        if user["password"] != password_hash:
            raise HTTPException(status_code=401, detail="Invalid email or password")
        
        # Generate token (in production, use JWT)
        token = f"mock_token_{request.email}"
        
        return {
            "message": "Login successful",
            "token": token,
            "user": {
                "name": user.get("name"),
                "email": user["email"],
                "id": user["id"],
                "phone_number": user.get("phone_number"),
                "age": user.get("age"),
                "domain_preferences": user.get("domain_preferences", [])
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(status_code=500, detail="Failed to login")

@app.get("/auth/me")
async def get_current_user(request: Request):
    """Get current user (requires authentication in production)"""
    if not mongodb_service:
        raise HTTPException(status_code=503, detail="MongoDB service not available")
    
    # In production, verify JWT token from Authorization header
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    token = auth_header.replace("Bearer ", "")
    
    # Extract email from token (in production, decode JWT)
    if not token.startswith("mock_token_"):
        raise HTTPException(status_code=401, detail="Invalid token")
    
    email = token.replace("mock_token_", "")
    
    try:
        user = mongodb_service.get_user_by_email(email)
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        
        # Get subscription tier from user document (preferred) or check subscription
        subscription_tier = user.get("subscription_tier", "Free")
        
        # If not in user doc, check active subscription
        if subscription_tier == "Free" and user.get("id"):
            subscription = mongodb_service.get_user_subscription(user_id=user["id"], status="active")
            if subscription:
                subscription_tier = subscription.get("plan_name", "Free")
                # Update user document with subscription tier
                mongodb_service.update_user_subscription_tier(user["id"], subscription_tier)
        
        return {
            "name": user.get("name"),
            "email": user["email"],
            "id": user["id"],
            "phone_number": user.get("phone_number"),
            "age": user.get("age"),
            "domain_preferences": user.get("domain_preferences", []),
            "subscription_tier": subscription_tier
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get user error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get user")


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


# ---------- Subscription endpoints ----------


class CreatePlanRequest(BaseModel):
    name: str
    amount: int  # Amount in paise (smallest currency unit)
    currency: str = "INR"
    interval: int = 1
    period: str = "monthly"  # daily, weekly, monthly, yearly
    description: Optional[str] = None


class CreateSubscriptionRequest(BaseModel):
    plan_id: str
    user_id: str
    customer_notify: int = 1
    total_count: Optional[int] = None
    notes: Optional[Dict[str, str]] = None


class CancelSubscriptionRequest(BaseModel):
    subscription_id: str
    cancel_at_cycle_end: bool = False


@app.post("/subscriptions/plans")
async def create_subscription_plan(request: CreatePlanRequest):
    """Create a subscription plan in Razorpay (admin/one-time setup)"""
    try:
        if not razorpay_service or not razorpay_service.client:
            raise HTTPException(
                status_code=503,
                detail="Razorpay service not available. Check RAZORPAY_ID and RAZORPAY_KEY."
            )
        
        plan = razorpay_service.create_plan(
            name=request.name,
            amount=request.amount,
            currency=request.currency,
            interval=request.interval,
            period=request.period,
            description=request.description
        )
        
        return {
            "success": True,
            "plan": plan
        }
    except Exception as e:
        logger.error(f"❌ Failed to create subscription plan: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/subscriptions/plans")
async def list_subscription_plans(count: int = 10, skip: int = 0):
    """List available subscription plans"""
    try:
        if not razorpay_service or not razorpay_service.client:
            raise HTTPException(
                status_code=503,
                detail="Razorpay service not available. Check RAZORPAY_ID and RAZORPAY_KEY."
            )
        
        plans = razorpay_service.list_plans(count=count, skip=skip)
        return {
            "success": True,
            "plans": plans
        }
    except Exception as e:
        logger.error(f"❌ Failed to list subscription plans: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/subscriptions/config")
async def get_subscription_config():
    """Get Razorpay public configuration (Key ID) for frontend"""
    try:
        if not config.RAZORPAY_ID:
            raise HTTPException(
                status_code=503,
                detail="Razorpay not configured"
            )
        
        return {
            "success": True,
            "razorpay_key_id": config.RAZORPAY_ID
        }
    except Exception as e:
        logger.error(f"❌ Failed to get subscription config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/subscriptions/create")
async def create_subscription(request: CreateSubscriptionRequest):
    """Create a subscription for a user"""
    try:
        if not razorpay_service or not razorpay_service.client:
            raise HTTPException(
                status_code=503,
                detail="Razorpay service not available. Check RAZORPAY_ID and RAZORPAY_KEY."
            )
        
        if not mongodb_service:
            raise HTTPException(
                status_code=503,
                detail="MongoDB service not available"
            )
        
        # Create subscription in Razorpay
        subscription = razorpay_service.create_subscription(
            plan_id=request.plan_id,
            customer_notify=request.customer_notify,
            total_count=request.total_count,
            notes=request.notes
        )
        
        # Get plan details
        plan = razorpay_service.get_plan(request.plan_id)
        
        # Extract plan name - try multiple possible locations
        plan_name = "Pro"  # Default
        if plan:
            # Try different possible locations for plan name
            plan_name_raw = (
                plan.get("item", {}).get("name") or
                plan.get("name") or
                request.notes.get("plan_name") if request.notes else None or
                "Pro"
            )
            # Normalize plan name
            plan_name_raw_lower = plan_name_raw.lower()
            if "pro" in plan_name_raw_lower:
                plan_name = "Pro"
            elif "enterprise" in plan_name_raw_lower:
                plan_name = "Enterprise"
            else:
                plan_name = plan_name_raw
        
        # Store subscription in MongoDB
        from datetime import datetime
        subscription_data = {
            "user_id": request.user_id,
            "razorpay_subscription_id": subscription.get("id"),
            "razorpay_plan_id": request.plan_id,
            "plan_name": plan_name,
            "status": subscription.get("status", "created"),
            "amount": plan.get("item", {}).get("amount", 0) if plan else 0,
            "currency": plan.get("item", {}).get("currency", "INR") if plan else "INR",
            "current_start": subscription.get("current_start"),
            "current_end": subscription.get("current_end"),
            "next_billing_at": subscription.get("end_at"),
            "created_at": datetime.utcnow(),
            "razorpay_data": subscription  # Store full Razorpay response
        }
        
        mongodb_service.upsert_subscription(subscription_data)
        
        # Update user's subscription tier immediately if status is active
        # Otherwise, it will be updated via webhook when payment is completed
        if subscription.get("status") == "active":
            mongodb_service.update_user_subscription_tier(request.user_id, plan_name)
            logger.info(f"✅ Updated user {request.user_id} subscription tier to {plan_name}")
        else:
            logger.info(f"⏳ Subscription created with status '{subscription.get('status')}'. User tier will be updated when subscription is activated via webhook.")
        
        return {
            "success": True,
            "subscription_id": subscription.get("id"),
            "short_url": subscription.get("short_url"),
            "subscription": subscription
        }
    except Exception as e:
        logger.error(f"❌ Failed to create subscription: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/subscriptions/status")
async def get_subscription_status(user_id: Optional[str] = None):
    """Get user's subscription status"""
    try:
        if not mongodb_service:
            raise HTTPException(
                status_code=503,
                detail="MongoDB service not available"
            )
        
        if not user_id:
            return {
                "success": True,
                "subscription": None,
                "message": "No user_id provided"
            }
        
        subscription = mongodb_service.get_user_subscription(user_id=user_id)
        
        if subscription:
            # Optionally fetch latest data from Razorpay
            if razorpay_service and razorpay_service.client:
                try:
                    razorpay_sub = razorpay_service.get_subscription(
                        subscription.get("razorpay_subscription_id")
                    )
                    # Update status if changed
                    if razorpay_sub.get("status") != subscription.get("status"):
                        mongodb_service.update_subscription_status(
                            subscription.get("razorpay_subscription_id"),
                            razorpay_sub.get("status"),
                            {
                                "current_start": razorpay_sub.get("current_start"),
                                "current_end": razorpay_sub.get("current_end"),
                                "next_billing_at": razorpay_sub.get("end_at")
                            }
                        )
                        subscription["status"] = razorpay_sub.get("status")
                except Exception as e:
                    logger.warning(f"Failed to sync with Razorpay: {e}")
        
        return {
            "success": True,
            "subscription": subscription
        }
    except Exception as e:
        logger.error(f"❌ Failed to get subscription status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/subscriptions/cancel")
async def cancel_subscription(request: CancelSubscriptionRequest):
    """Cancel user's subscription"""
    try:
        if not razorpay_service or not razorpay_service.client:
            raise HTTPException(
                status_code=503,
                detail="Razorpay service not available. Check RAZORPAY_ID and RAZORPAY_KEY."
            )
        
        if not mongodb_service:
            raise HTTPException(
                status_code=503,
                detail="MongoDB service not available"
            )
        
        # Cancel subscription in Razorpay
        subscription = razorpay_service.cancel_subscription(
            subscription_id=request.subscription_id,
            cancel_at_cycle_end=request.cancel_at_cycle_end
        )
        
        # Update status in MongoDB
        mongodb_service.update_subscription_status(
            request.subscription_id,
            subscription.get("status", "cancelled"),
            {
                "current_start": subscription.get("current_start"),
                "current_end": subscription.get("current_end"),
                "next_billing_at": subscription.get("end_at")
            }
        )
        
        return {
            "success": True,
            "subscription": subscription
        }
    except Exception as e:
        logger.error(f"❌ Failed to cancel subscription: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/webhooks/razorpay")
async def razorpay_webhook(request: Request):
    """Handle Razorpay webhook events"""
    try:
        if not razorpay_service:
            raise HTTPException(
                status_code=503,
                detail="Razorpay service not available"
            )
        
        if not mongodb_service:
            raise HTTPException(
                status_code=503,
                detail="MongoDB service not available"
            )
        
        # Get raw body for signature verification
        body = await request.body()
        body_str = body.decode('utf-8')
        
        # Get signature from header
        signature = request.headers.get("X-Razorpay-Signature", "")
        
        # Verify webhook signature
        if not razorpay_service.verify_webhook_signature(body_str, signature):
            logger.warning("⚠️ Invalid webhook signature")
            raise HTTPException(status_code=400, detail="Invalid webhook signature")
        
        # Parse webhook payload from body string
        webhook_data = json.loads(body_str)
        event = webhook_data.get("event")
        payload = webhook_data.get("payload", {})
        
        logger.info(f"📥 Received Razorpay webhook: {event}")
        
        # Handle different webhook events
        if event == "subscription.activated":
            subscription = payload.get("subscription", {}).get("entity", {})
            subscription_id = subscription.get("id")
            
            if subscription_id:
                # Get subscription from DB to get user_id and plan_name
                sub_doc = mongodb_service.get_subscription_by_razorpay_id(subscription_id)
                if sub_doc:
                    user_id = sub_doc.get("user_id")
                    plan_name = sub_doc.get("plan_name", "Pro")
                    
                    logger.info(f"📥 Processing subscription.activated for user {user_id}, plan {plan_name}")
                    
                    mongodb_service.update_subscription_status(
                        subscription_id,
                        "active",
                        {
                            "current_start": subscription.get("current_start"),
                            "current_end": subscription.get("current_end"),
                            "next_billing_at": subscription.get("end_at")
                        }
                    )
                    
                    # Update user's subscription tier
                    if user_id:
                        success = mongodb_service.update_user_subscription_tier(user_id, plan_name)
                        if success:
                            logger.info(f"✅ Successfully updated user {user_id} tier to {plan_name} via webhook")
                        else:
                            logger.error(f"❌ Failed to update user {user_id} tier to {plan_name}")
                else:
                    logger.warning(f"⚠️ Subscription {subscription_id} not found in database")
        
        elif event == "subscription.charged":
            subscription = payload.get("subscription", {}).get("entity", {})
            payment = payload.get("payment", {}).get("entity", {})
            subscription_id = subscription.get("id")
            
            if subscription_id:
                # Get subscription from DB to get user_id and plan_name
                sub_doc = mongodb_service.get_subscription_by_razorpay_id(subscription_id)
                if sub_doc:
                    user_id = sub_doc.get("user_id")
                    plan_name = sub_doc.get("plan_name", "Pro")
                    
                    logger.info(f"📥 Processing subscription.charged for user {user_id}, plan {plan_name}")
                    
                    # Update subscription with payment info
                    update_data = {
                        "current_start": subscription.get("current_start"),
                        "current_end": subscription.get("current_end"),
                        "next_billing_at": subscription.get("end_at"),
                        "last_payment_id": payment.get("id"),
                        "last_payment_amount": payment.get("amount"),
                        "last_payment_date": payment.get("created_at")
                    }
                    mongodb_service.update_subscription_status(
                        subscription_id,
                        subscription.get("status", "active"),
                        update_data
                    )
                    
                    # Update user's subscription tier when payment is charged
                    if user_id and subscription.get("status") == "active":
                        success = mongodb_service.update_user_subscription_tier(user_id, plan_name)
                        if success:
                            logger.info(f"✅ Successfully updated user {user_id} tier to {plan_name} via subscription.charged webhook")
                        else:
                            logger.error(f"❌ Failed to update user {user_id} tier to {plan_name}")
                else:
                    logger.warning(f"⚠️ Subscription {subscription_id} not found in database for subscription.charged event")
        
        elif event == "subscription.cancelled":
            subscription = payload.get("subscription", {}).get("entity", {})
            subscription_id = subscription.get("id")
            
            if subscription_id:
                # Get subscription from DB to get user_id
                sub_doc = mongodb_service.get_subscription_by_razorpay_id(subscription_id)
                if sub_doc:
                    user_id = sub_doc.get("user_id")
                    
                    mongodb_service.update_subscription_status(
                        subscription_id,
                        "cancelled",
                        {
                            "current_start": subscription.get("current_start"),
                            "current_end": subscription.get("current_end"),
                            "next_billing_at": subscription.get("end_at")
                        }
                    )
                    
                    # Update user's subscription tier to Free
                    if user_id:
                        mongodb_service.update_user_subscription_tier(user_id, "Free")
        
        elif event == "payment.failed":
            payment = payload.get("payment", {}).get("entity", {})
            subscription_id = payment.get("subscription_id")
            
            if subscription_id:
                # Update subscription to reflect failed payment
                subscription = razorpay_service.get_subscription(subscription_id)
                mongodb_service.update_subscription_status(
                    subscription_id,
                    subscription.get("status", "pending"),
                    {
                        "last_payment_failed": True,
                        "last_payment_failure_reason": payment.get("error_description")
                    }
                )
        
        return {"success": True, "message": "Webhook processed"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to process webhook: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=config.SERVICE_PORT)