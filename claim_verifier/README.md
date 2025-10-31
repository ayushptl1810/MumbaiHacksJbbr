# Claim Verifier Agent System

A comprehensive fact-checking and claim verification system built with Google Agents SDK, Custom Search API, and Gemini AI. This system provides both individual fact-checking capabilities and orchestrated workflows for processing multiple content sources.

## System Architecture

### Core Components

1. **TextFactChecker**: Individual fact-checking engine with Google Custom Search integration
2. **ClaimVerifierAgent**: Specialized agents for different verification tasks
3. **ClaimVerifierOrchestrator**: Workflow orchestrator managing multiple agents
4. **ClaimVerifierAgent (Main)**: Command-line interface and integration point

### Agent Roles

- **Claim Extraction Specialist**: Extracts verifiable claims from content
- **Fact Verification Specialist**: Verifies claims against reliable sources  
- **Priority Assessment Specialist**: Prioritizes claims based on risk and impact
- **Report Generation Specialist**: Creates comprehensive fact-checking reports

## Features

- **Multi-Agent Workflow**: Specialized agents working together for comprehensive verification
- **Smart Search**: Google Custom Search API with fact-checking websites
- **AI Analysis**: Gemini 2.5 Flash for intelligent claim analysis and reasoning
- **Relevance Scoring**: TF-IDF similarity matching for accurate source matching
- **Priority Assessment**: Risk-based claim prioritization system
- **Comprehensive Reporting**: Detailed verification reports with evidence
- **Integration Ready**: Designed for seamless integration with trend scanning systems
- **Session Management**: Persistent session tracking and result storage

## Setup

### 1. Install Dependencies

```bash
pip install google-generativeai scikit-learn requests python-dotenv
```

### 2. Environment Variables

Create a `.env` file with the following variables:

```env
# Google Custom Search API (required)
GOOGLE_FACT_CHECK_API_KEY=your_google_api_key
GOOGLE_FACT_CHECK_CX=your_custom_search_engine_id

# Gemini API (required)
GEMINI_API_KEY=your_gemini_api_key
```

### 3. Google Custom Search Setup

1. **Create a Custom Search Engine**:
   - Visit: https://cse.google.com/cse/
   - Click "Add" to create a new search engine
   - Configure to search fact-checking sites:
     - snopes.com
     - politifact.com
     - factcheck.org
     - reuters.com/fact-check
     - apnews.com/hub/ap-fact-check
     - And other reliable fact-checking sources

2. **Get your Search Engine ID (cx)**:
   - In your custom search engine settings
   - Copy the "Search engine ID"

3. **Enable Custom Search API**:
   - Visit: https://console.developers.google.com/
   - Enable "Custom Search API"
   - Create API credentials

## Usage

### Command-Line Interface

The claim verifier agent provides a command-line interface for various operations:

```bash
# Verify claims from a file
python claim_verifier_agent.py --operation verify-file --file content.json

# Quick verification of a single claim
python claim_verifier_agent.py --operation verify-claim --claim "Vaccines contain microchips"

# Verify Reddit posts from trend scanner
python claim_verifier_agent.py --operation verify-reddit --reddit-file reddit_posts.json

# Get session summary
python claim_verifier_agent.py --operation session-summary
```

### Programmatic Usage

#### Basic Fact-Checking

```python
import asyncio
from claim_verifier import TextFactChecker

async def check_claim():
    fact_checker = TextFactChecker()
    
    result = await fact_checker.verify(
        text_input="COVID-19 vaccines contain microchips",
        claim_context="Social media post",
        claim_date="2024-01-15"
    )
    
    print(f"Verdict: {result['verdict']}")
    print(f"Verified: {result['verified']}")
    print(f"Message: {result['message']}")

asyncio.run(check_claim())
```

#### Agent-Based Workflow

```python
import asyncio
from claim_verifier_agent import ClaimVerifierAgent

async def agent_workflow():
    # Initialize agent
    agent = ClaimVerifierAgent()
    await agent.initialize()
    
    # Verify content from various sources
    content_data = [
        {
            "title": "Breaking News: Miracle Cure Found",
            "content": "Scientists claim new drug cures all diseases instantly",
            "source": "UnverifiedNews.com"
        }
    ]
    
    result = await agent.verify_content_list(content_data)
    print(f"Success: {result['success']}")
    print(f"Claims verified: {result['summary']['claims_verified']}")

asyncio.run(agent_workflow())
```

#### Multi-Agent Orchestration

```python
import asyncio
from claim_verifier import ClaimVerifierOrchestrator

async def orchestrated_verification():
    # Initialize orchestrator
    orchestrator = ClaimVerifierOrchestrator()
    
    # Process multiple content items with full workflow
    content_data = [
        {"title": "News Article 1", "content": "Content with claims..."},
        {"title": "Social Media Post", "content": "Viral claim about..."}
    ]
    
    result = await orchestrator.verify_content(content_data)
    
    # Access workflow results
    for step_result in result['workflow_results']:
        print(f"Agent: {step_result['agent_role']}")
        print(f"Task: {step_result['task']}")
        print(f"Success: {not step_result['has_error']}")

asyncio.run(orchestrated_verification())
```

### Integration with Trend Scanner

```python
import asyncio
from claim_verifier_agent import ClaimVerifierAgent

async def integrate_with_trend_scanner(reddit_posts):
    """Integrate claim verification with trend scanner"""
    
    agent = ClaimVerifierAgent()
    await agent.initialize()
    
    # Verify Reddit posts detected by trend scanner
    result = await agent.verify_reddit_posts(reddit_posts)
    
    if result['success']:
        summary = result['summary']
        high_priority = summary['high_priority_claims']
        
        if high_priority > 0:
            print(f"⚠️ {high_priority} high-priority claims detected!")
            print("Flagging for manual review...")
            
        return result
    
    return None

# Example usage in trend scanner
# verification_result = await integrate_with_trend_scanner(detected_posts)
```

### Advanced Usage

```python
import asyncio
from claim_verifier import TextFactChecker

async def detailed_fact_check():
    fact_checker = TextFactChecker()
    
    # Check a complex claim
    result = await fact_checker.verify(
        text_input="Scientists have proven that climate change is a natural cycle unrelated to human activity",
        claim_context="Online article",
        claim_date="2024-03-20"
    )
    
    # Access detailed analysis
    analysis = result['details']['analysis']
    print(f"Confidence: {analysis['confidence']}")
    print(f"Reasoning: {analysis['reasoning']}")
    print(f"Sources found: {len(result['details']['fact_checks'])}")
    
    # Review fact-check sources
    for source in result['details']['fact_checks'][:3]:
        print(f"- {source['title']}")
        print(f"  {source['link']}")

asyncio.run(detailed_fact_check())
```

## Response Format

```json
{
    "verified": true/false,
    "verdict": "true|false|mixed|uncertain|no_content|error",
    "message": "Human-readable explanation",
    "details": {
        "claim_text": "Original claim text",
        "claim_context": "Context provided",
        "claim_date": "Date provided",
        "fact_checks": [
            {
                "title": "Fact-check article title",
                "snippet": "Relevant excerpt",
                "link": "URL to full article"
            }
        ],
        "analysis": {
            "verdict": "true|false|mixed|uncertain",
            "verified": true/false,
            "confidence": "high|medium|low",
            "reasoning": "AI analysis reasoning",
            "relevant_results_count": 5,
            "analysis_method": "gemini|fallback"
        }
    }
}
```

## Agent System Architecture

### File Structure

```
claim_verifier/
├── __init__.py              # Package initialization
├── tools.py                 # TextFactChecker core implementation
├── config.py                # Configuration management
├── agents.py                # Agent classes and orchestrator
├── example.py               # Usage examples
└── README.md                # Documentation

claim_verifier_agent.py      # Main CLI interface
claim_verifier_integration_example.py  # Integration examples
claim_verification_results/  # Output directory (auto-created)
├── file_verification_*.json
├── quick_verification_*.json
└── reddit_posts_verification_*.json
```

### Agent Workflow Output

The agent system produces comprehensive output files with the following structure:

```json
{
    "success": true,
    "message": "Successfully verified 3 claims from 2 content items",
    "workflow_results": [
        {
            "agent_role": "Claim Extraction Specialist",
            "task": "extract_claims",
            "result": {
                "extracted_claims": [...],
                "total_claims": 3,
                "sources_processed": 2
            },
            "has_error": false,
            "timestamp": "2024-01-15T10:30:00"
        },
        {
            "agent_role": "Fact Verification Specialist", 
            "task": "verify_claims",
            "result": {
                "verified_claims": [...],
                "total_verified": 3,
                "success_rate": 1.0
            },
            "has_error": false,
            "timestamp": "2024-01-15T10:32:00"
        }
    ],
    "final_report": {
        "report": "Comprehensive fact-checking report...",
        "summary_stats": {
            "total_claims": 3,
            "verdict_distribution": {
                "true": 1,
                "false": 1,
                "mixed": 0,
                "uncertain": 1,
                "error": 0
            },
            "priority_distribution": {
                "high": 1,
                "medium": 1,
                "low": 1
            }
        },
        "high_priority_claims": [...],
        "false_claims": [...]
    },
    "summary": {
        "content_items_processed": 2,
        "claims_extracted": 3,
        "claims_verified": 3,
        "high_priority_claims": 1
    },
    "session_metadata": {
        "session_id": "cv_session_20240115_103000",
        "operation": "content_list_verification",
        "saved_at": "2024-01-15T10:35:00",
        "agent_version": "1.0.0"
    }
}
```

### Claim Object Structure

Each verified claim contains:

```json
{
    "claim_text": "Specific factual assertion",
    "context": "Surrounding context from source",
    "reason_to_check": "Why this claim needs verification",
    "priority": "high|medium|low",
    "source_content": {...},
    "extraction_timestamp": "2024-01-15T10:30:00",
    "verification": {
        "verified": true,
        "verdict": "false",
        "message": "This claim has been debunked by multiple sources",
        "details": {
            "fact_checks": [...],
            "search_results": [...],
            "search_queries": [...],
            "analysis": {
                "verdict": "false",
                "verified": true,
                "confidence": "high",
                "reasoning": "Multiple fact-checkers confirm this is false",
                "relevant_results_count": 5,
                "analysis_method": "gemini"
            }
        }
    },
    "priority_score": 85.0,
    "priority_level": "high",
    "prioritization_timestamp": "2024-01-15T10:33:00"
}
```

## Trend Scanner Integration

The claim verifier system is designed for seamless integration with trend scanning:

```python
from claim_verifier import TextFactChecker
from trend_scanner import TrendScannerOrchestrator

async def enhanced_scanning():
    # Initialize components
    orchestrator = TrendScannerOrchestrator(reddit_config)
    fact_checker = TextFactChecker()
    
    # Scan for trending posts
    results = orchestrator.scan_trending_content()
    
    # Fact-check high-risk posts
    for post in results['trending_posts']:
        if post['risk_level'] == 'HIGH':
            verification = await fact_checker.verify(
                text_input=post['title'],
                claim_context=f"Reddit post from r/{post['subreddit']}",
                claim_date=post.get('created_date', 'Unknown')
            )
            
            post['fact_check'] = verification
            print(f"Post: {post['title']}")
            print(f"Fact-check verdict: {verification['verdict']}")
```

## Configuration

The system uses the following configuration (see `config.py`):

- `GEMINI_MODEL`: "gemini-2.5-flash" (configurable)
- `MAX_SEARCH_RESULTS`: 10 (configurable)
- `RELEVANCE_THRESHOLD`: 0.05 (configurable)
- `MAX_ALTERNATIVE_QUERIES`: 2 (configurable)

## Error Handling

The system includes robust error handling:

- **API Failures**: Graceful degradation with informative error messages
- **No Results**: Intelligent query reformulation using LLM
- **Analysis Failures**: Fallback to simple rule-based analysis
- **Rate Limiting**: Built-in timeout and retry mechanisms

## Limitations

- Requires Google Custom Search API (paid service after free quota)
- Limited to fact-checking sources indexed by Google
- English language optimized (can be extended for other languages)
- Dependent on quality of fact-checking sources in search results

## Testing

Run the example to test your setup:

```bash
python claim_verifier/example.py
```

This will test the fact-checker with sample claims and show detailed output.