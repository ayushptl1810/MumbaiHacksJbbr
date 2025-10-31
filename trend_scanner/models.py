from dataclasses import dataclass
from typing import Optional, List, Dict


@dataclass
class TrendingPost:
    post_id: str
    title: str
    content: str
    author: str
    subreddit: str
    url: str
    score: int
    upvote_ratio: float
    num_comments: int
    created_utc: float
    velocity_score: float
    engagement_rate: float
    detected_at: str
    permalink: str
    risk_level: str
    scraped_content: Optional[str] = None
    content_source: str = "reddit"


@dataclass
class BatchPostData:
    """Data structure for batch processing posts"""
    post_id: str
    title: str
    content: str
    scraped_content: Optional[str]
    subreddit: str
    score: int
    upvote_ratio: float
    num_comments: int
    age_hours: float
    author: str
    has_external_content: bool


@dataclass
class BatchRiskAssessment:
    """Result of batch risk assessment"""
    post_id: str
    risk_level: str
    confidence: float = 0.0
    reasoning: Optional[str] = None


@dataclass
class VelocityMetric:
    initial_score: int
    current_score: int
    initial_time: float
    current_time: float
    velocity: float
