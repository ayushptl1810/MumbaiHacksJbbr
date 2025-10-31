# Explanation Agent

The Explanation Agent converts fact-check verification results into structured, publishable debunk posts for the Aegis Feed.

## Features

- **Structured Post Creation**: Converts verification results into well-formatted debunk posts
- **AI-Powered Content Generation**: Uses Gemini AI to create clear, engaging explanations
- **Source Categorization**: Automatically categorizes sources into confirmation vs. misinformation sources
- **Confidence Scoring**: Converts verification confidence into percentage scores
- **Batch Processing**: Handles multiple verification results efficiently
- **JSON Output**: Produces structured JSON posts ready for publication

## Output Structure

Each debunk post includes:

```json
{
  "post_id": "aegis_post_20250921_120000",
  "timestamp": "2025-09-21T12:00:00",
  "claim": {
    "text": "The original claim being fact-checked",
    "verdict": "true|false|mixed|uncertain", 
    "verified": true/false
  },
  "post_content": {
    "heading": "Clear, engaging headline about the fact-check",
    "body": "Detailed explanation with evidence and context",
    "summary": "Concise one-sentence summary"
  },
  "sources": {
    "misinformation_sources": [...],
    "confirmation_sources": [...],
    "total_sources": 5
  },
  "confidence_percentage": 85.0,
  "metadata": {
    "agent_version": "1.0",
    "processing_method": "gemini_ai"
  }
}
```

## Usage

### Individual Post Creation

```python
from explanation_agent.agents import ExplanationAgent

agent = ExplanationAgent()

# Verification result from claim verifier
verification_result = {
    "verdict": "false",
    "verified": False,
    "confidence": "high",
    "reasoning": "Detailed analysis...",
    "message": "Summary message...",
    "sources": {...},
    "claim_text": "The claim to fact-check"
}

# Create debunk post
post = agent.create_debunk_post(verification_result)
```

### Batch Processing

```python
# Process multiple verification results
verification_results = [result1, result2, result3]
posts = agent.batch_create_posts(verification_results)
```

## Integration with Verification Pipeline

The Explanation Agent integrates seamlessly with the claim verification output:

1. **Claim Verifier** produces verification results
2. **Explanation Agent** converts results to structured posts
3. **Aegis Feed** publishes the debunk posts

## Configuration

Configure in `config.py`:

- **Gemini API settings**
- **Output directory** for saved posts
- **Content length limits**
- **Trusted source domains**
- **Confidence thresholds**

## Testing

Run the test script:

```bash
python explanation_agent/test_explanation_agent.py
```

This will create sample debunk posts and save them to the `aegis_feed_posts` directory.

## Files

- `agents.py` - Main ExplanationAgent class
- `config.py` - Configuration settings
- `test_explanation_agent.py` - Testing script
- `../aegis_feed_posts/` - Output directory for posts