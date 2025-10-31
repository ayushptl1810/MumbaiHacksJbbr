"""Claim Verifier Package - Comprehensive fact-checking and claim verification system"""

from .tools import TextFactChecker
from .config import config
from .agents import GoogleAgent, GoogleAgentsOrchestrator, ClaimVerifierOrchestrator

__version__ = "2.0.0"
__author__ = "MumbaiHacks Team"

__all__ = ['TextFactChecker', 'config', 'GoogleAgent', 'GoogleAgentsOrchestrator', 'ClaimVerifierOrchestrator']