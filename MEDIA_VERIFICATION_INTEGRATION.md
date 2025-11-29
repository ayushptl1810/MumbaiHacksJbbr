# Media Verification Integration

## Overview

The claim verifier now includes **intelligent media verification** as an add-on to enhance fact-checking accuracy. When a claim involves images or videos, the system automatically detects and verifies the media content, providing additional context to the main LLM evaluator.

## How It Works

### 1. **Automatic Media Detection**

The system automatically detects if a claim involves media content by:

- **URL Analysis**: Checks for image/video URLs in the post data
- **Text Analysis**: Looks for keywords like "image", "video", "photo", "footage", etc.
- **Content Source**: Analyzes the content_source field for media indicators

```python
# Detected media types
- Images: .jpg, .png, .gif, imgur.com, i.redd.it, etc.
- Videos: YouTube, TikTok, Instagram, Twitter/X, .mp4, etc.
```

### 2. **Media Verification Process**

When media is detected:

```
┌─────────────────────────────────────────────────────────────┐
│                    CLAIM VERIFICATION FLOW                   │
└─────────────────────────────────────────────────────────────┘

1. Text Claim Received
   ├─> Detect Media (images/videos)
   │
   ├─> IF MEDIA DETECTED:
   │   ├─> Run Image Verifier (for images)
   │   │   ├─> Gemini Vision Analysis (AI-generated detection)
   │   │   ├─> Reverse Image Search
   │   │   └─> Returns: verdict, summary, confidence
   │   │
   │   └─> Run Video Verifier (for videos)  
   │       ├─> Frame Extraction
   │       ├─> Frame Analysis (per frame)
   │       └─> Returns: verdict, analysis, details
   │
   └─> Run Text Fact-Checking (always)
       ├─> Google Custom Search
       ├─> LLM Analysis
       └─> Include media analysis as ADDITIONAL CONTEXT

2. Final Verdict
   └─> Combines fact-checking + media analysis
```

### 3. **Enhanced LLM Prompt**

When media verification is available, the main fact-checking LLM receives enriched context:

```
CLAIM TO VERIFY: "This image shows event X from 2020"

ADDITIONAL CONTEXT - IMAGE VERIFICATION:
The claim involves image content. An independent image verification analysis was performed:
- Verdict: false
- Confidence: high
- Analysis: "Image is from 2015, not 2020. Reverse image search shows
  this photo was taken at a different event in London, not New York."

This image analysis provides additional context but should be considered 
alongside the fact-checking sources below.

FACT-CHECKING SOURCES:
1. Snopes: [analysis]
2. PolitiFact: [analysis]
...
```

## Features

### ✅ **Lazy Loading**
Media verifiers are only loaded when needed, reducing memory footprint.

### ✅ **Non-Blocking**
- Media verification **does not block** text fact-checking
- If media verification fails, text verification continues normally
- Works independently - doesn't require media URLs

### ✅ **Add-On Architecture**
- **Optional enhancement**: System works without media verifiers installed
- **Graceful degradation**: Falls back to text-only verification if media verifiers unavailable
- **Independent verification**: Media analysis provided as additional context, not as replacement

### ✅ **Comprehensive Media Support**

**Image Verification:**
- AI-generated/deepfake detection (Gemini Vision)
- Manipulation artifact detection
- Reverse image search
- Context verification
- Source credibility analysis

**Video Verification:**
- Frame-by-frame analysis
- YouTube/social media support
- Visual inconsistency detection
- Temporal context verification

## Usage

### Automatic (Recommended)

The system automatically detects and verifies media. No code changes needed!

```python
# Just pass your claim data as usual
verifier = ClaimVerifierAgent()
result = await verifier.verify_content(content_data)

# If media is detected, result will include:
# result['verified_claims'][0]['verification']['media_analysis'] = {
#     'type': 'image' or 'video',
#     'summary': 'Analysis summary',
#     'verdict': 'true/false/uncertain',
#     'confidence': 'high/medium/low'
# }
```

### Manual Media Inclusion

You can explicitly provide media URLs:

```python
claim_data = {
    'text_input': 'This image shows...',
    'claim_context': 'Full context text',
    'url': 'https://example.com/image.jpg',  # Will be detected
    'image_url': 'https://i.imgur.com/abc.jpg',  # Explicit image
    'video_url': 'https://youtube.com/watch?v=xyz'  # Explicit video
}

result = await fact_checker.verify(**claim_data)
```

## Example Output

### Without Media Analysis
```json
{
  "verified": true,
  "verdict": "false",
  "message": "Claim is false according to fact-checkers",
  "confidence": "high",
  "sources": {...}
}
```

### With Media Analysis
```json
{
  "verified": true,
  "verdict": "false",
  "message": "Claim is false according to fact-checkers and media analysis",
  "confidence": "high",
  "sources": {...},
  "media_analysis": {
    "type": "image",
    "summary": "Image is from 2015 event in London, not 2020 in New York as claimed",
    "verdict": "false",
    "confidence": "high"
  }
}
```

## Configuration

### Environment Variables

```bash
# Required for media verification
SERP_API_KEY=your_serpapi_key_here  # For reverse image search

# Optional - Cloudinary for video frame hosting
CLOUDINARY_CLOUD_NAME=your_cloud_name
CLOUDINARY_API_KEY=your_api_key
CLOUDINARY_API_SECRET=your_api_secret
```

### Dependencies

Media verification requires:
```bash
pip install Pillow opencv-python requests google-search-results cloudinary
```

## Benefits

1. **Higher Accuracy**: Detects manipulated images/videos that text fact-checking might miss
2. **Context-Aware**: Provides visual context to complement textual analysis
3. **Deepfake Detection**: Uses Gemini Vision to detect AI-generated content
4. **Source Verification**: Reverse image search finds original sources
5. **Temporal Verification**: Detects old content presented as recent

## Limitations

- Media verifiers require additional API keys (SerpAPI)
- Video verification may be slower (frame extraction required)
- Large videos may be downsampled for efficiency
- Requires internet access for reverse image search

## Future Enhancements

- [ ] Audio verification for voice deepfakes
- [ ] OCR text extraction from images
- [ ] Facial recognition for identity verification
- [ ] Geolocation verification from image metadata
- [ ] Real-time video stream analysis
- [ ] Multi-language support for media content

## Troubleshooting

### Media Verification Not Running

**Issue**: Media not being verified even though URLs present

**Solution**:
1. Check if image_verifier.py and video_verifier.py exist
2. Verify SERP_API_KEY is set in environment
3. Check logs for media detection: `📸 Media detected: image`

### Verifier Import Errors

**Issue**: `ImportError: No module named 'image_verifier'`

**Solution**:
```bash
# Ensure files exist in project root
ls image_verifier.py video_verifier.py

# Check imports are accessible
python -c "from image_verifier import ImageVerifier"
```

### Slow Verification

**Issue**: Claims taking too long to verify

**Solution**:
- Media verification runs in parallel with text verification
- For videos, reduce frame extraction interval
- Use Cloudinary for faster frame uploads
- Consider caching media verification results

## Technical Details

### Media Detection Algorithm

```python
def _detect_media_in_claim(claim_context, claim_data):
    """
    Priority order:
    1. Direct URL fields (url, image_url, video_url)
    2. Content source indicators (scraped, image)
    3. Text keyword analysis (image, video, photo, etc.)
    4. URL extraction from text content
    """
```

### Verification Flow

```python
# Simplified pseudocode
if media_detected:
    media_result = await verify_media(media_url, claim_text)
    # Media result becomes ADDITIONAL CONTEXT
    
text_result = await verify_text(claim, search_results)
# Text verification sees media analysis in prompt

final_verdict = combine(text_result, media_context)
```

## Support

For issues or questions:
- Check logs for media detection warnings
- Verify API keys are correctly configured
- Ensure dependencies are installed
- Test with a simple image/video URL first

---

**Integration Status**: ✅ Active
**Last Updated**: November 2024
**Version**: 1.0.0
