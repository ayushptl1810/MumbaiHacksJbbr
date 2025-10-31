"""Test script for Explanation Agent"""

import os
import sys
import json
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from explanation_agent.agents import ExplanationAgent


def test_explanation_agent():
    """Test the Explanation Agent with sample verification results"""
    
    # Sample verification results (similar to what claim verifier outputs)
    sample_verification_results = [
        {
            "verified": False,
            "verdict": "false",
            "message": "Multiple sources confirm this claim is false. The event described did not occur as stated.",
            "confidence": "high",
            "reasoning": "Cross-referenced with multiple fact-checking organizations including Snopes and PolitiFact. No credible evidence supports the claim. Original sources contradict the assertion.",
            "sources": {
                "links": [
                    "https://www.snopes.com/fact-check/example-false-claim/",
                    "https://www.politifact.com/factchecks/false-claim-rating/",
                    "https://www.factcheck.org/example-debunk/"
                ],
                "titles": [
                    "Snopes: False Claim About Event Debunked",
                    "PolitiFact: Claim Rates False on Truth-O-Meter",
                    "FactCheck.org: No Evidence for Viral Claim"
                ],
                "count": 3
            },
            "claim_text": "A viral social media post claims that a major political figure made controversial statements at a private event",
            "verification_date": datetime.now().isoformat()
        },
        {
            "verified": True,
            "verdict": "partially_verified",
            "message": "The core claim has some truth but important context is missing. The situation is more nuanced than presented.",
            "confidence": 0.75,
            "reasoning": "While the basic facts are accurate, the claim oversimplifies a complex situation. Additional context from multiple sources shows the full picture is more balanced.",
            "sources": {
                "links": [
                    "https://www.reuters.com/fact-check/partial-truth/",
                    "https://www.bbc.com/news/context-matters",
                    "https://apnews.com/nuanced-story"
                ],
                "titles": [
                    "Reuters: Fact Check Reveals Partial Truth",
                    "BBC: Context Matters for Viral Claim",
                    "AP News: The Full Story Behind the Headlines"
                ],
                "count": 3
            },
            "claim_text": "Social media users share statistics about recent policy changes without full context",
            "verification_date": datetime.now().isoformat()
        }
    ]
    
    try:
        print("🚀 Initializing Explanation Agent...")
        agent = ExplanationAgent()
        
        print(f"📝 Creating debunk posts for {len(sample_verification_results)} verification results...")
        
        # Test individual post creation
        for i, result in enumerate(sample_verification_results, 1):
            print(f"\n--- Creating Post {i} ---")
            post = agent.create_debunk_post(result)
            
            print(f"✅ Post ID: {post.get('post_id')}")
            print(f"📰 Heading: {post.get('post_content', {}).get('heading', 'N/A')}")
            print(f"🎯 Verdict: {post.get('claim', {}).get('verdict', 'N/A')}")
            print(f"📊 Confidence: {post.get('confidence_percentage', 0)}%")
            print(f"📁 Saved to: {post.get('saved_to', 'N/A')}")
            
            # Show sources breakdown
            sources = post.get('sources', {})
            print(f"🔍 Confirmation Sources: {len(sources.get('confirmation_sources', []))}")
            print(f"⚠️  Misinformation Sources: {len(sources.get('misinformation_sources', []))}")
        
        # Test batch processing
        print(f"\n--- Testing Batch Processing ---")
        batch_posts = agent.batch_create_posts(sample_verification_results)
        print(f"✅ Batch processing completed: {len(batch_posts)} posts created")
        
        # Display sample output
        if batch_posts:
            sample_post = batch_posts[0]
            print(f"\n--- Sample Post Structure ---")
            print(json.dumps({
                "post_id": sample_post.get('post_id'),
                "claim": sample_post.get('claim'),
                "post_content": sample_post.get('post_content'),
                "confidence_percentage": sample_post.get('confidence_percentage'),
                "sources_summary": {
                    "total": sample_post.get('sources', {}).get('total_sources'),
                    "confirmation": len(sample_post.get('sources', {}).get('confirmation_sources', [])),
                    "misinformation": len(sample_post.get('sources', {}).get('misinformation_sources', []))
                }
            }, indent=2))
        
        print(f"\n🎉 Explanation Agent test completed successfully!")
        print(f"📁 Posts saved to: {os.path.join(os.path.dirname(__file__), '..', 'aegis_feed_posts')}")
        
    except Exception as e:
        print(f"❌ Error testing Explanation Agent: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_explanation_agent()