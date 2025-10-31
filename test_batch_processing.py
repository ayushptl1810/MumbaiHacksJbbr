#!/usr/bin/env python3
"""
Test script for batch processing in ClaimVerifier and ExplanationAgent
Tests both individual and batch processing to validate performance improvements
"""

import sys
import os
import json
import time
import asyncio
from datetime import datetime
from typing import List, Dict, Any

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from claim_verifier.agents import ClaimVerifierOrchestrator
from explanation_agent.agents import ExplanationAgent

class BatchProcessingTester:
    """Test suite for batch processing functionality"""
    
    def __init__(self):
        self.claim_verifier = None
        self.explanation_agent = None
        self.test_results = {
            'claim_verifier': {},
            'explanation_agent': {},
            'performance_comparison': {}
        }
    
    async def setup(self):
        """Initialize the agents for testing"""
        print("🔧 Setting up test environment...")
        
        try:
            # Initialize ClaimVerifier
            print("   Initializing ClaimVerifier...")
            self.claim_verifier = ClaimVerifierOrchestrator()
            
            # Initialize ExplanationAgent
            print("   Initializing ExplanationAgent...")
            self.explanation_agent = ExplanationAgent()
            
            print("✅ Test environment setup complete")
            return True
            
        except Exception as e:
            print(f"❌ Setup failed: {e}")
            return False
    
    def create_test_claims(self, count: int) -> List[Dict[str, Any]]:
        """Create test claims for verification"""
        test_claims = [
            {
                'title': 'Claim: Vaccines cause autism in children',
                'content': 'A social media post claims that vaccines are linked to autism development in children.',
                'source': 'https://reddit.com/r/conspiracy/post1',
                'platform': 'reddit',
                'timestamp': datetime.now().isoformat(),
                'claim_metadata': {
                    'post_index': 0,
                    'extracted_claim': 'Vaccines cause autism in children'
                }
            },
            {
                'title': 'Claim: Climate change is not real',
                'content': 'User claims that climate change is a hoax created by governments.',
                'source': 'https://reddit.com/r/skeptic/post2',
                'platform': 'reddit',
                'timestamp': datetime.now().isoformat(),
                'claim_metadata': {
                    'post_index': 1,
                    'extracted_claim': 'Climate change is not real'
                }
            },
            {
                'title': 'Claim: 5G towers spread COVID-19',
                'content': 'Post alleging that 5G cellular towers are responsible for spreading coronavirus.',
                'source': 'https://reddit.com/r/conspiracy/post3',
                'platform': 'reddit',
                'timestamp': datetime.now().isoformat(),
                'claim_metadata': {
                    'post_index': 2,
                    'extracted_claim': '5G towers spread COVID-19'
                }
            },
            {
                'title': 'Claim: Earth is flat',
                'content': 'Conspiracy theory claiming the Earth is flat and space agencies are lying.',
                'source': 'https://reddit.com/r/flatearth/post4',
                'platform': 'reddit',
                'timestamp': datetime.now().isoformat(),
                'claim_metadata': {
                    'post_index': 3,
                    'extracted_claim': 'Earth is flat'
                }
            },
            {
                'title': 'Claim: Moon landing was fake',
                'content': 'Allegation that the 1969 moon landing was staged by NASA.',
                'source': 'https://reddit.com/r/conspiracy/post5',
                'platform': 'reddit',
                'timestamp': datetime.now().isoformat(),
                'claim_metadata': {
                    'post_index': 4,
                    'extracted_claim': 'Moon landing was fake'
                }
            },
            {
                'title': 'Claim: COVID vaccines contain microchips',
                'content': 'False claim that COVID-19 vaccines contain tracking microchips.',
                'source': 'https://reddit.com/r/conspiracy/post6',
                'platform': 'reddit',
                'timestamp': datetime.now().isoformat(),
                'claim_metadata': {
                    'post_index': 5,
                    'extracted_claim': 'COVID vaccines contain microchips'
                }
            },
            {
                'title': 'Claim: Drinking bleach cures diseases',
                'content': 'Dangerous misinformation about bleach as a medical treatment.',
                'source': 'https://reddit.com/r/health/post7',
                'platform': 'reddit',
                'timestamp': datetime.now().isoformat(),
                'claim_metadata': {
                    'post_index': 6,
                    'extracted_claim': 'Drinking bleach cures diseases'
                }
            },
            {
                'title': 'Claim: Chemtrails control weather',
                'content': 'Conspiracy theory about aircraft condensation trails being used for weather control.',
                'source': 'https://reddit.com/r/conspiracy/post8',
                'platform': 'reddit',
                'timestamp': datetime.now().isoformat(),
                'claim_metadata': {
                    'post_index': 7,
                    'extracted_claim': 'Chemtrails control weather'
                }
            },
            {
                'title': 'Claim: Birds are not real',
                'content': 'Satirical conspiracy theory claiming birds are government drones.',
                'source': 'https://reddit.com/r/birdsarentreal/post9',
                'platform': 'reddit',
                'timestamp': datetime.now().isoformat(),
                'claim_metadata': {
                    'post_index': 8,
                    'extracted_claim': 'Birds are not real'
                }
            },
            {
                'title': 'Claim: Vitamin C prevents all diseases',
                'content': 'Exaggerated claim about vitamin C being a cure-all supplement.',
                'source': 'https://reddit.com/r/health/post10',
                'platform': 'reddit',
                'timestamp': datetime.now().isoformat(),
                'claim_metadata': {
                    'post_index': 9,
                    'extracted_claim': 'Vitamin C prevents all diseases'
                }
            }
        ]
        
        # Extend the list to the requested count by cycling through
        extended_claims = []
        for i in range(count):
            base_claim = test_claims[i % len(test_claims)].copy()
            base_claim['title'] = f"Claim {i+1}: {base_claim['title'][7:]}"
            base_claim['claim_metadata']['post_index'] = i
            extended_claims.append(base_claim)
        
        return extended_claims
    
    async def test_claim_verifier_batch(self, claim_count: int = 20):
        """Test ClaimVerifier batch processing"""
        print(f"\n🧪 Testing ClaimVerifier Batch Processing ({claim_count} claims)")
        print("=" * 60)
        
        test_claims = self.create_test_claims(claim_count)
        
        try:
            start_time = time.time()
            
            # Test batch verification
            verification_result = await self.claim_verifier.verify_content(test_claims)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Analyze results
            success = verification_result.get('success', False)
            verified_claims = verification_result.get('verified_claims', [])
            summary = verification_result.get('summary', {})
            
            print(f"✅ Batch verification completed in {processing_time:.2f} seconds")
            print(f"   Success: {success}")
            print(f"   Total claims processed: {len(verified_claims)}")
            print(f"   Successfully verified: {summary.get('successfully_verified', 0)}")
            print(f"   Verification errors: {summary.get('verification_errors', 0)}")
            print(f"   Batch size used: {summary.get('batch_size_used', 'N/A')}")
            print(f"   Total batches: {summary.get('total_batches', 'N/A')}")
            
            # Store test results
            self.test_results['claim_verifier'] = {
                'success': success,
                'processing_time': processing_time,
                'total_claims': len(verified_claims),
                'successfully_verified': summary.get('successfully_verified', 0),
                'verification_errors': summary.get('verification_errors', 0),
                'batch_size_used': summary.get('batch_size_used', 15),
                'total_batches': summary.get('total_batches', 1),
                'claims_per_second': len(verified_claims) / processing_time if processing_time > 0 else 0
            }
            
            # Show sample results
            print("\n📊 Sample Verification Results:")
            for i, claim in enumerate(verified_claims[:3]):
                verification = claim.get('verification', {})
                print(f"   {i+1}. {claim.get('claim_text', 'Unknown')[:50]}...")
                print(f"      Verdict: {verification.get('verdict', 'unknown')}")
                print(f"      Confidence: {verification.get('confidence', 'unknown')}")
                print(f"      Verified: {verification.get('verified', False)}")
            
            if len(verified_claims) > 3:
                print(f"   ... and {len(verified_claims) - 3} more claims")
            
            return verified_claims
            
        except Exception as e:
            print(f"❌ ClaimVerifier batch test failed: {e}")
            self.test_results['claim_verifier'] = {
                'success': False,
                'error': str(e)
            }
            return []
    
    def create_mock_verification_results(self, count: int) -> List[Dict[str, Any]]:
        """Create mock verification results for ExplanationAgent testing"""
        mock_results = []
        
        base_results = [
            {
                'claim_text': 'Vaccines cause autism in children',
                'verdict': 'false',
                'verified': False,
                'confidence': 'high',
                'reasoning': 'Multiple peer-reviewed studies have shown no link between vaccines and autism.',
                'message': 'This claim is false based on scientific evidence',
                'sources': {
                    'links': ['https://cdc.gov/vaccines', 'https://who.int/vaccines'],
                    'titles': ['CDC Vaccine Safety', 'WHO Vaccine Information'],
                    'count': 2
                },
                'verification_date': datetime.now().isoformat()
            },
            {
                'claim_text': 'Climate change is not real',
                'verdict': 'false',
                'verified': False,
                'confidence': 'high',
                'reasoning': 'Scientific consensus shows human activities are driving climate change.',
                'message': 'This claim contradicts overwhelming scientific evidence',
                'sources': {
                    'links': ['https://nasa.gov/climate', 'https://ipcc.ch/reports'],
                    'titles': ['NASA Climate Evidence', 'IPCC Climate Reports'],
                    'count': 2
                },
                'verification_date': datetime.now().isoformat()
            },
            {
                'claim_text': '5G towers spread COVID-19',
                'verdict': 'false',
                'verified': False,
                'confidence': 'high',
                'reasoning': 'Viruses cannot spread through radio waves or mobile networks.',
                'message': 'This claim is scientifically impossible',
                'sources': {
                    'links': ['https://who.int/emergencies/diseases/novel-coronavirus-2019/advice-for-public/myth-busters'],
                    'titles': ['WHO COVID-19 Myth Busters'],
                    'count': 1
                },
                'verification_date': datetime.now().isoformat()
            }
        ]
        
        # Extend to requested count
        for i in range(count):
            base_result = base_results[i % len(base_results)].copy()
            base_result['claim_text'] = f"Test claim {i+1}: {base_result['claim_text']}"
            mock_results.append(base_result)
        
        return mock_results
    
    async def test_explanation_agent_batch(self, post_count: int = 15):
        """Test ExplanationAgent batch processing"""
        print(f"\n🧪 Testing ExplanationAgent Batch Processing ({post_count} posts)")
        print("=" * 60)
        
        mock_verification_results = self.create_mock_verification_results(post_count)
        
        try:
            start_time = time.time()
            
            # Test batch explanation generation
            explanation_result = self.explanation_agent.batch_create_posts(mock_verification_results)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Analyze results
            success = explanation_result.get('success', False)
            debunk_posts = explanation_result.get('debunk_posts', [])
            batch_stats = explanation_result.get('batch_statistics', {})
            
            print(f"✅ Batch explanation generation completed in {processing_time:.2f} seconds")
            print(f"   Success: {success}")
            print(f"   Total posts created: {len(debunk_posts)}")
            print(f"   Successful posts: {batch_stats.get('successful_posts', 0)}")
            print(f"   Error posts: {batch_stats.get('error_posts', 0)}")
            print(f"   Batch size used: {batch_stats.get('batch_size_used', 'N/A')}")
            print(f"   Total batches: {batch_stats.get('total_batches', 'N/A')}")
            
            # Store test results
            self.test_results['explanation_agent'] = {
                'success': success,
                'processing_time': processing_time,
                'total_posts': len(debunk_posts),
                'successful_posts': batch_stats.get('successful_posts', 0),
                'error_posts': batch_stats.get('error_posts', 0),
                'batch_size_used': batch_stats.get('batch_size_used', 10),
                'total_batches': batch_stats.get('total_batches', 1),
                'posts_per_second': len(debunk_posts) / processing_time if processing_time > 0 else 0
            }
            
            # Show sample results
            print("\n📄 Sample Debunk Posts:")
            for i, post in enumerate(debunk_posts[:3]):
                post_content = post.get('post_content', {})
                print(f"   {i+1}. {post.get('post_id', 'Unknown ID')}")
                print(f"      Heading: {post_content.get('heading', 'No heading')[:50]}...")
                print(f"      Confidence: {post.get('confidence_percentage', 0)}%")
                print(f"      Sources: {post.get('sources', {}).get('total_sources', 0)}")
                print(f"      Saved to: {os.path.basename(post.get('saved_to', 'Not saved'))}")
            
            if len(debunk_posts) > 3:
                print(f"   ... and {len(debunk_posts) - 3} more posts")
            
            return debunk_posts
            
        except Exception as e:
            print(f"❌ ExplanationAgent batch test failed: {e}")
            self.test_results['explanation_agent'] = {
                'success': False,
                'error': str(e)
            }
            return []
    
    def analyze_performance(self):
        """Analyze and compare performance results"""
        print(f"\n📊 Performance Analysis")
        print("=" * 60)
        
        cv_results = self.test_results.get('claim_verifier', {})
        ea_results = self.test_results.get('explanation_agent', {})
        
        if cv_results.get('success', False):
            print(f"🔍 ClaimVerifier Performance:")
            print(f"   Processing time: {cv_results.get('processing_time', 0):.2f} seconds")
            print(f"   Claims per second: {cv_results.get('claims_per_second', 0):.2f}")
            print(f"   Batch size: {cv_results.get('batch_size_used', 'N/A')}")
            print(f"   Success rate: {cv_results.get('successfully_verified', 0)}/{cv_results.get('total_claims', 0)} ({(cv_results.get('successfully_verified', 0)/max(cv_results.get('total_claims', 1), 1)*100):.1f}%)")
        
        if ea_results.get('success', False):
            print(f"\n📝 ExplanationAgent Performance:")
            print(f"   Processing time: {ea_results.get('processing_time', 0):.2f} seconds")
            print(f"   Posts per second: {ea_results.get('posts_per_second', 0):.2f}")
            print(f"   Batch size: {ea_results.get('batch_size_used', 'N/A')}")
            print(f"   Success rate: {ea_results.get('successful_posts', 0)}/{ea_results.get('total_posts', 0)} ({(ea_results.get('successful_posts', 0)/max(ea_results.get('total_posts', 1), 1)*100):.1f}%)")
        
        # Estimate API call reduction
        if cv_results.get('success', False):
            total_claims = cv_results.get('total_claims', 0)
            batch_size = cv_results.get('batch_size_used', 15)
            batches = cv_results.get('total_batches', 1)
            
            individual_api_calls = total_claims * 2  # Rough estimate: 2 calls per claim
            batch_api_calls = batches + (total_claims)  # 1 analysis call per batch + search calls
            
            print(f"\n💰 Estimated API Call Reduction (ClaimVerifier):")
            print(f"   Individual processing: ~{individual_api_calls} API calls")
            print(f"   Batch processing: ~{batch_api_calls} API calls")
            print(f"   Reduction: {(1 - batch_api_calls/max(individual_api_calls, 1))*100:.1f}%")
        
        if ea_results.get('success', False):
            total_posts = ea_results.get('total_posts', 0)
            batch_size = ea_results.get('batch_size_used', 10)
            batches = ea_results.get('total_batches', 1)
            
            individual_api_calls = total_posts * 2  # Rough estimate: 2 calls per post
            batch_api_calls = batches * 2  # 2 calls per batch (content + source)
            
            print(f"\n💰 Estimated API Call Reduction (ExplanationAgent):")
            print(f"   Individual processing: ~{individual_api_calls} API calls")
            print(f"   Batch processing: ~{batch_api_calls} API calls")
            print(f"   Reduction: {(1 - batch_api_calls/max(individual_api_calls, 1))*100:.1f}%")
    
    def save_test_results(self):
        """Save test results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"batch_processing_test_results_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(self.test_results, f, indent=2)
            print(f"\n💾 Test results saved to: {filename}")
        except Exception as e:
            print(f"\n❌ Failed to save test results: {e}")
    
    async def run_all_tests(self):
        """Run all batch processing tests"""
        print("🚀 Starting Batch Processing Tests")
        print("=" * 60)
        
        # Setup
        if not await self.setup():
            print("❌ Setup failed, aborting tests")
            return
        
        # Test ClaimVerifier batch processing
        verified_claims = await self.test_claim_verifier_batch(20)  # Test with 20 claims (batch size 15)
        
        # Test ExplanationAgent batch processing  
        debunk_posts = await self.test_explanation_agent_batch(15)  # Test with 15 posts (batch size 10)
        
        # Analyze performance
        self.analyze_performance()
        
        # Save results
        self.save_test_results()
        
        print(f"\n🎉 Batch processing tests completed!")
        print("=" * 60)

async def main():
    """Main test function"""
    tester = BatchProcessingTester()
    await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())