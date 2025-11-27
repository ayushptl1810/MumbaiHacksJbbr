"""
Tools for the trend_scanner agent.

This module provides a clean import interface for Reddit and Threads scanning tools.
The actual implementations are in the tools/ subfolder for better organization.
"""

# Import from tools subfolder
from .tools import (
    GoogleTool,
    RedditScanTool,
    RedditScanInput,
    RedditScanOutput,
    ThreadsScanTool,
    ThreadsScanInput,
    ThreadsScanOutput,
    RISK_ASSESSMENT_PROMPT
)

# Export all public classes
__all__ = [
    'GoogleTool',
    'RedditScanTool',
    'RedditScanInput',
    'RedditScanOutput',
    'ThreadsScanTool',
    'ThreadsScanInput',
    'ThreadsScanOutput',
    'RISK_ASSESSMENT_PROMPT'
]


# Legacy prompt definitions for backward compatibility
SCAN_TASK_PROMPT = """
Perform comprehensive scanning of r/{subreddit} for trending posts that could contain misinformation.

Your enhanced analysis should include:
1. Posts with high velocity (rapid upvote growth)
2. Content analysis of Reddit posts AND linked external content
3. Detection of suspicious claims, unsourced assertions, or misleading information
4. Evaluation of source credibility for linked content
5. Assessment of emotional manipulation techniques
6. Identification of conspiracy theories or pseudoscience

Subreddit: {subreddit}
Scan parameters: limit=20, sort_type=new

Pay special attention to:
- Posts linking to external articles or sources
- Health/medical misinformation
- Political conspiracy theories
- Pseudoscientific claims
- Sensational headlines with unverified content

Return detailed information about trending posts with full content analysis.
"""

ASSESSMENT_TASK_PROMPT = """
Perform comprehensive analysis of all trending posts found across all subreddits, including full content analysis of scraped external sources.

Scan results from all subreddits:
{scan_results}

Your comprehensive analysis should:
1. Rank posts by potential misinformation risk using both metadata and content analysis
2. Identify the most urgent posts needing immediate verification
3. Categorize posts by misinformation type (health, political, scientific, etc.)
4. Highlight posts with the highest viral potential and risk combination
5. Assess the credibility and bias of external sources that were scraped
6. Identify coordinated misinformation campaigns or similar narratives across posts

Priority factors:
- HIGH risk posts with scraped external content containing false claims
- Posts with high velocity in controversial subreddits
- Content contradicting scientific consensus without proper evidence
- Sensational claims designed to provoke emotional responses
- Unsourced medical or health advice
- Political conspiracy theories gaining rapid traction

For each high-priority post, provide:
- Risk level justification based on content analysis
- Key claims that need fact-checking
- Source credibility assessment
- Viral potential score
- Recommended verification priority
"""