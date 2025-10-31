"""Claim Verifier Agent - Main orchestration script for fact-checking operations"""

import os
import sys
import json
import logging
import asyncio
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from claim_verifier.agents import ClaimVerifierOrchestrator
from claim_verifier.config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('claim_verifier.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class ClaimVerifierAgent:
    """Main Claim Verifier Agent for processing fact-checking requests"""
    
    def __init__(self):
        self.orchestrator = None
        self.session_id = f"cv_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.results_dir = "claim_verification_results"
        os.makedirs(self.results_dir, exist_ok=True)
        
        logger.info(f"Claim Verifier Agent initialized - Session: {self.session_id}")
    
    async def initialize(self):
        """Initialize the orchestrator and agents"""
        try:
            logger.info("Initializing Claim Verifier Orchestrator...")
            self.orchestrator = ClaimVerifierOrchestrator()
            logger.info("Claim Verifier Agent ready for operations")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Claim Verifier Agent: {e}")
            return False
    
    async def verify_from_file(self, file_path: str, file_format: str = "auto") -> Dict[str, Any]:
        """Verify claims from a file containing content data"""
        try:
            logger.info(f"Loading content from file: {file_path}")
            
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Determine file format
            if file_format == "auto":
                if file_path.endswith('.json'):
                    file_format = "json"
                elif file_path.endswith('.txt'):
                    file_format = "txt"
                else:
                    file_format = "json"  # default
            
            # Load content based on format
            if file_format == "json":
                with open(file_path, 'r', encoding='utf-8') as f:
                    content_data = json.load(f)
                
                # Ensure it's a list
                if isinstance(content_data, dict):
                    content_data = [content_data]
            
            elif file_format == "txt":
                with open(file_path, 'r', encoding='utf-8') as f:
                    text_content = f.read()
                
                # Convert text to content format
                content_data = [{
                    'title': f"Content from {os.path.basename(file_path)}",
                    'content': text_content,
                    'source': file_path,
                    'timestamp': datetime.now().isoformat()
                }]
            
            else:
                raise ValueError(f"Unsupported file format: {file_format}")
            
            logger.info(f"Loaded {len(content_data)} content items")
            
            # Run verification
            result = await self.orchestrator.verify_content(content_data)
            
            # Save results
            result_file = self._save_results(result, f"file_verification_{os.path.basename(file_path)}")
            result['result_file'] = result_file
            
            return result
            
        except Exception as e:
            logger.error(f"File verification failed: {e}")
            return {
                'success': False,
                'message': f'File verification failed: {str(e)}',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def verify_content_list(self, content_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Verify claims from a list of content items"""
        try:
            logger.info(f"Verifying content list with {len(content_items)} items")
            
            result = await self.orchestrator.verify_content(content_items)
            
            # Save results
            result_file = self._save_results(result, "content_list_verification")
            result['result_file'] = result_file
            
            return result
            
        except Exception as e:
            logger.error(f"Content list verification failed: {e}")
            return {
                'success': False,
                'message': f'Content list verification failed: {str(e)}',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def quick_verify_claim(self, claim_text: str, context: str = "Direct input") -> Dict[str, Any]:
        """Quickly verify a single claim"""
        try:
            logger.info(f"Quick verifying claim: {claim_text[:100]}...")
            
            result = await self.orchestrator.quick_verify(claim_text, context)
            
            # Save results
            result_file = self._save_results(result, "quick_verification")
            result['result_file'] = result_file
            
            return result
            
        except Exception as e:
            logger.error(f"Quick verification failed: {e}")
            return {
                'success': False,
                'message': f'Quick verification failed: {str(e)}',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def verify_reddit_posts(self, reddit_posts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Verify claims from Reddit posts (integration with trend scanner)"""
        try:
            logger.info(f"Verifying Reddit posts: {len(reddit_posts)} posts")
            
            # Convert Reddit posts to content format
            content_data = []
            for post in reddit_posts:
                content_item = {
                    'title': post.get('title', ''),
                    'content': post.get('selftext', '') or post.get('content', ''),
                    'source': f"Reddit - r/{post.get('subreddit', 'unknown')}",
                    'url': post.get('url', ''),
                    'author': post.get('author', 'unknown'),
                    'score': post.get('score', 0),
                    'timestamp': post.get('created_utc', datetime.now().isoformat()),
                    'reddit_metadata': {
                        'id': post.get('id', ''),
                        'permalink': post.get('permalink', ''),
                        'num_comments': post.get('num_comments', 0),
                        'upvote_ratio': post.get('upvote_ratio', 0)
                    }
                }
                content_data.append(content_item)
            
            result = await self.orchestrator.verify_content(content_data)
            
            # Add Reddit-specific metadata
            result['reddit_integration'] = {
                'posts_processed': len(reddit_posts),
                'source': 'trend_scanner_integration'
            }
            
            # Save results
            result_file = self._save_results(result, "reddit_posts_verification")
            result['result_file'] = result_file
            
            return result
            
        except Exception as e:
            logger.error(f"Reddit posts verification failed: {e}")
            return {
                'success': False,
                'message': f'Reddit posts verification failed: {str(e)}',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _save_results(self, results: Dict[str, Any], operation_name: str) -> str:
        """Save verification results to file"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{operation_name}_{timestamp}.json"
            filepath = os.path.join(self.results_dir, filename)
            
            # Add session metadata
            results['session_metadata'] = {
                'session_id': self.session_id,
                'operation': operation_name,
                'saved_at': datetime.now().isoformat(),
                'agent_version': '1.0.0'
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Results saved to: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            return ""
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current session"""
        try:
            # List all result files for this session
            session_files = []
            if os.path.exists(self.results_dir):
                for filename in os.listdir(self.results_dir):
                    if filename.endswith('.json'):
                        filepath = os.path.join(self.results_dir, filename)
                        try:
                            with open(filepath, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                if data.get('session_metadata', {}).get('session_id') == self.session_id:
                                    session_files.append({
                                        'filename': filename,
                                        'filepath': filepath,
                                        'operation': data.get('session_metadata', {}).get('operation', 'unknown'),
                                        'timestamp': data.get('session_metadata', {}).get('saved_at', 'unknown'),
                                        'success': data.get('success', False)
                                    })
                        except:
                            continue
            
            return {
                'session_id': self.session_id,
                'operations_count': len(session_files),
                'session_files': session_files,
                'orchestrator_initialized': self.orchestrator is not None,
                'results_directory': self.results_dir
            }
            
        except Exception as e:
            logger.error(f"Failed to get session summary: {e}")
            return {
                'session_id': self.session_id,
                'error': str(e)
            }


async def main():
    """Main entry point for claim verifier agent"""
    parser = argparse.ArgumentParser(description='Claim Verifier Agent - Fact-checking operations')
    parser.add_argument('--operation', choices=['verify-file', 'verify-claim', 'verify-reddit', 'session-summary'], 
                       required=True, help='Operation to perform')
    parser.add_argument('--file', help='File path for file verification')
    parser.add_argument('--format', choices=['json', 'txt', 'auto'], default='auto', 
                       help='File format (auto-detect by default)')
    parser.add_argument('--claim', help='Claim text for quick verification')
    parser.add_argument('--context', default='Direct input', help='Context for claim verification')
    parser.add_argument('--reddit-file', help='JSON file containing Reddit posts data')
    
    args = parser.parse_args()
    
    # Initialize agent
    agent = ClaimVerifierAgent()
    
    if not await agent.initialize():
        logger.error("Failed to initialize agent")
        return 1
    
    try:
        if args.operation == 'verify-file':
            if not args.file:
                logger.error("--file argument required for verify-file operation")
                return 1
            
            result = await agent.verify_from_file(args.file, args.format)
            print(json.dumps(result, indent=2))
        
        elif args.operation == 'verify-claim':
            if not args.claim:
                logger.error("--claim argument required for verify-claim operation")
                return 1
            
            result = await agent.quick_verify_claim(args.claim, args.context)
            print(json.dumps(result, indent=2))
        
        elif args.operation == 'verify-reddit':
            if not args.reddit_file:
                logger.error("--reddit-file argument required for verify-reddit operation")
                return 1
            
            # Load Reddit posts data
            with open(args.reddit_file, 'r', encoding='utf-8') as f:
                reddit_posts = json.load(f)
            
            result = await agent.verify_reddit_posts(reddit_posts)
            print(json.dumps(result, indent=2))
        
        elif args.operation == 'session-summary':
            summary = agent.get_session_summary()
            print(json.dumps(summary, indent=2))
        
        return 0
        
    except Exception as e:
        logger.error(f"Operation failed: {e}")
        print(json.dumps({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }, indent=2))
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))