# Orchestrator Agent System

## Overview

The Orchestrator Agent coordinates the complete misinformation detection pipeline by integrating:

1. **Trend Scanner Agent** - Finds trending posts and generates claims/summaries
2. **Claim Verifier Agent** - Fact-checks the extracted claims  
3. **Orchestrator Agent** - Coordinates the workflow and combines results

## Architecture Flow

```
Reddit Posts → Trend Scanner → Claims Extraction → Claim Verifier → Final Results
                     ↓
                 Gemini AI                      ↓
              (Summarization)              Google Search + AI
                                         (Fact Checking)
```

## Pipeline Steps

### 1. Trend Scanner Phase
- Scans Reddit subreddits for trending posts
- Sends all posts to Gemini AI for batch processing
- Generates structured claims and summaries
- Returns JSON with posts data

### 2. Claims Extraction Phase  
- Orchestrator extracts verifiable claims from trend results
- Converts posts data to claim verification format
- Filters out non-verifiable content

### 3. Claim Verification Phase
- Sends claims to Claim Verifier Agent
- Uses Google Custom Search + Gemini AI for fact-checking
- Returns verification results with verdicts

### 4. Results Combination Phase
- Combines trend data with verification results
- Creates final structured output
- Saves comprehensive results

## Usage

### Run Complete Pipeline
```bash
python run_pipeline.py --mode full
```

### Run Individual Components
```bash
# Trend scanning only
python run_pipeline.py --mode trend-only

# Direct orchestrator
python orchestrator_agent.py

# Direct trend scanner
python trend_scanner_agent.py

# Direct claim verifier
python claim_verifier_agent.py --operation verify-claim --claim "Your claim here"
```

## Output Format

The orchestrator produces a final JSON with this structure:

```json
{
  "timestamp": "2024-01-15T10:30:00",
  "total_posts": 3,
  "posts": [
    {
      "claim": "Government plans to ban social media platforms",
      "summary": "Comprehensive AI-generated summary combining post content and external sources...",
      "platform": "reddit", 
      "Post_link": "https://reddit.com/r/conspiracy/comments/abc123",
      "verification": {
        "verified": true,
        "verdict": "false",
        "message": "This claim has been debunked by multiple fact-checkers",
        "details": {
          "confidence": "high",
          "sources_found": 5
        }
      },
      "post_index": 0
    }
  ]
}
```

## Configuration

### Required Environment Variables
```env
# Reddit API
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret

# Google/Gemini AI  
GEMINI_API_KEY=your_gemini_api_key

# Fact-checking (for claim verifier)
GOOGLE_FACT_CHECK_API_KEY=your_google_search_api_key
GOOGLE_FACT_CHECK_CX=your_custom_search_engine_id
```

### Target Subreddits
Edit `TARGET_SUBREDDITS` in `trend_scanner_agent.py`:
```python
TARGET_SUBREDDITS = [
    'NoFilterNews',
    'conspiracy',
    'worldnews',
    'politics'  # Add more as needed
]
```

## File Structure

```
├── orchestrator_agent.py          # Main orchestrator logic
├── run_pipeline.py                # Simple launcher script
├── trend_scanner_agent.py         # Trend scanning with Gemini summarization
├── claim_verifier_agent.py        # Claim verification system
├── claim_verifier/                # Claim verifier package
│   ├── agents.py                  # Verification agents
│   ├── tools.py                   # TextFactChecker
│   └── config.py                  # Configuration
├── trend_scanner/                 # Trend scanner package  
│   ├── google_agents.py           # Google Agents orchestration
│   └── tools.py                   # Reddit scanning tools
├── orchestrator_results/          # Orchestrator output files
├── claim_verification_results/    # Claim verifier output files
└── requirements.txt               # Dependencies
```

## Session Management

Each orchestrator run creates:
- Unique session ID with timestamp
- Comprehensive result files in `orchestrator_results/`
- Individual component logs
- Combined trend + verification data

## Error Handling

The orchestrator includes robust error handling:
- **Trend Scanner Fails**: Returns empty results, continues gracefully
- **No Claims Found**: Skips verification, returns trend data only  
- **Claim Verifier Fails**: Returns trend data with "not_verified" status
- **API Failures**: Falls back to basic processing where possible

## Monitoring

Logs are written to:
- `orchestrator.log` - Main orchestrator operations
- `claim_verifier.log` - Claim verification details
- `trend_scanner.log.txt` - Trend scanning operations

## Performance

- **Batch Processing**: Single Gemini API call for all posts
- **Efficient Verification**: Optimized claim extraction and verification
- **Minimal API Calls**: Strategic batching reduces costs
- **Parallel Processing**: Where possible, operations run concurrently

## Integration

The orchestrator is designed to be:
- **API-friendly**: Can be called programmatically  
- **Webhook-ready**: Easy to integrate with external systems
- **Scalable**: Can handle varying numbers of posts
- **Extensible**: Easy to add new verification sources or scanners