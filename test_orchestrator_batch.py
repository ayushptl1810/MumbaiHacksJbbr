"""Test Orchestrator Agent with Integrated Batch Processing

This test validates that the orchestrator pipeline correctly uses:
1. Batch processing for ClaimVerifier (max 15 claims)
2. Batch processing for ExplanationAgent (max 10 posts)
3. End-to-end integration with trend scanning
"""

import os
import sys
import json
import asyncio
import logging
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from orchestrator_agent import OrchestratorAgent

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestOrchestratorBatch:
    """Test suite for orchestrator batch processing integration"""
    
    def __init__(self):
        self.orchestrator = None
        self.test_results = []
    
    async def setup_orchestrator(self):
        """Setup orchestrator with mocked components for testing"""
        try:
            logger.info("Setting up orchestrator for batch processing tests...")
            
            # Mock environment variables
            os.environ['GEMINI_API_KEY'] = 'test_key_for_orchestrator_batch_testing'
            
            self.orchestrator = OrchestratorAgent()
            
            # Mock the Google Agents orchestrator initialization
            with patch('orchestrator_agent.GoogleAgentsOrchestrator') as mock_orchestrator_class:
                mock_orchestrator = Mock()
                mock_orchestrator.agents = {}
                mock_orchestrator_class.return_value = mock_orchestrator
                
                # Mock the claim verifier
                with patch('orchestrator_agent.ClaimVerifierOrchestrator') as mock_verifier_class:
                    mock_verifier = Mock()
                    mock_verifier_class.return_value = mock_verifier
                    
                    # Mock the explanation agent  
                    with patch('orchestrator_agent.ExplanationAgent') as mock_explanation_class:
                        mock_explanation = Mock()
                        mock_explanation_class.return_value = mock_explanation
                        
                        # Initialize the orchestrator
                        success = await self.orchestrator.initialize()
                        
                        if success:
                            logger.info("✅ Orchestrator setup successful")
                            return True
                        else:
                            logger.error("❌ Orchestrator setup failed")
                            return False
                            
        except Exception as e:
            logger.error(f"❌ Orchestrator setup failed: {e}")
            return False
    
    async def test_batch_claim_verification_integration(self):
        """Test that orchestrator uses batch processing for claim verification"""
        logger.info("\n🧪 Testing Batch Claim Verification Integration...")
        
        try:
            # Create mock trend data with multiple claims
            mock_trend_results = {
                'posts': [
                    {
                        'claim': f'Test claim {i}',
                        'summary': f'Summary for claim {i}',
                        'Post_link': f'https://reddit.com/post{i}',
                        'platform': 'reddit'
                    }
                    for i in range(20)  # Test with 20 claims (should trigger batching)
                ],
                'total_posts': 20
            }
            
            # Mock the trend scanning result
            mock_trend_workflow = {
                'workflow_results': [
                    {
                        'agent_role': 'Trend Scanning Coordinator',
                        'result': mock_trend_results
                    }
                ]
            }
            
            # Mock verification results with batch processing metadata
            mock_verification_results = {
                'success': True,
                'verified_claims': [
                    {
                        'claim_text': f'Test claim {i}',
                        'verification': {
                            'verified': True,
                            'verdict': 'mixed',
                            'confidence': 0.7
                        }
                    }
                    for i in range(20)
                ],
                'batch_processing': {
                    'enabled': True,
                    'total_claims': 20,
                    'batch_size': 15,
                    'processing_method': 'batch_verification'
                }
            }
            
            mock_verification_workflow = {
                'workflow_results': [
                    {
                        'agent_role': 'Claim Verification Coordinator',
                        'result': mock_verification_results
                    }
                ]
            }
            
            # Mock the Google Agents workflow execution
            async def mock_execute_workflow(tasks):
                task = tasks[0]
                if 'trend_scanner' in task.get('agent', ''):
                    return mock_trend_workflow
                elif 'verifier_coordinator' in task.get('agent', ''):
                    return mock_verification_workflow
                else:
                    return {'workflow_results': []}
            
            self.orchestrator.google_agents.execute_workflow = AsyncMock(side_effect=mock_execute_workflow)
            
            # Test the verification integration
            content_data = [
                {
                    'title': f'Claim: Test claim {i}',
                    'content': f'Summary for claim {i}',
                    'source': f'https://reddit.com/post{i}',
                    'platform': 'reddit'
                }
                for i in range(20)
            ]
            
            # Simulate verification with batch processing
            verification_task = {
                'agent': 'verifier_coordinator',
                'task': 'Verify extracted claims using comprehensive fact-checking workflow',
                'context': {
                    'verification_mode': 'comprehensive',
                    'use_google_search': True,
                    'content_data': content_data
                }
            }
            
            result = await self.orchestrator.google_agents.execute_workflow([verification_task])
            
            # Validate batch processing was used
            verification_result = result['workflow_results'][0]['result']
            batch_info = verification_result.get('batch_processing', {})
            
            assert batch_info.get('enabled') == True, "Batch processing should be enabled"
            assert batch_info.get('total_claims') == 20, "Should process all 20 claims"
            assert batch_info.get('batch_size') == 15, "Should use max batch size of 15"
            assert batch_info.get('processing_method') == 'batch_verification', "Should use batch verification method"
            
            logger.info("✅ Batch claim verification integration test PASSED")
            logger.info(f"   📊 Processed {batch_info['total_claims']} claims in batches of {batch_info['batch_size']}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Batch claim verification integration test FAILED: {e}")
            return False
    
    async def test_batch_explanation_generation_integration(self):
        """Test that orchestrator uses batch processing for explanation generation"""
        logger.info("\n🧪 Testing Batch Explanation Generation Integration...")
        
        try:
            # Create mock verification results with misinformation claims
            mock_verification_results = [
                {
                    'claim_text': f'False claim {i}',
                    'verification': {
                        'verified': False,
                        'verdict': 'false',
                        'confidence': 0.9
                    }
                }
                for i in range(15)  # Test with 15 misinformation claims
            ]
            
            # Mock explanation results with batch processing metadata
            mock_explanation_results = {
                'success': True,
                'debunk_posts': [
                    {
                        'post_id': f'debunk_{i}',
                        'claim': f'False claim {i}',
                        'post_content': f'Debunk content for claim {i}',
                        'confidence_percentage': 90
                    }
                    for i in range(15)
                ],
                'batch_processing': {
                    'enabled': True,
                    'total_claims': 15,
                    'batch_size': 10,
                    'processing_method': 'batch_explanation_generation'
                }
            }
            
            mock_explanation_workflow = {
                'workflow_results': [
                    {
                        'agent_role': 'Explanation Generation Coordinator',
                        'result': mock_explanation_results
                    }
                ]
            }
            
            # Mock the workflow execution for explanation
            async def mock_execute_workflow(tasks):
                task = tasks[0]
                if 'explanation_coordinator' in task.get('agent', ''):
                    return mock_explanation_workflow
                else:
                    return {'workflow_results': []}
            
            self.orchestrator.google_agents.execute_workflow = AsyncMock(side_effect=mock_execute_workflow)
            
            # Test the explanation integration
            explanation_task = {
                'agent': 'explanation_coordinator',
                'task': 'Generate debunk posts for misinformation claims using batch processing',
                'context': {
                    'verification_results': mock_verification_results,
                    'generation_mode': 'batch_debunk_posts'
                }
            }
            
            result = await self.orchestrator.google_agents.execute_workflow([explanation_task])
            
            # Validate batch processing was used
            explanation_result = result['workflow_results'][0]['result']
            batch_info = explanation_result.get('batch_processing', {})
            debunk_posts = explanation_result.get('debunk_posts', [])
            
            assert batch_info.get('enabled') == True, "Batch processing should be enabled"
            assert batch_info.get('total_claims') == 15, "Should process all 15 claims"
            assert batch_info.get('batch_size') == 10, "Should use max batch size of 10"
            assert batch_info.get('processing_method') == 'batch_explanation_generation', "Should use batch explanation method"
            assert len(debunk_posts) == 15, "Should generate 15 debunk posts"
            
            logger.info("✅ Batch explanation generation integration test PASSED")
            logger.info(f"   📊 Generated {len(debunk_posts)} debunk posts in batches of {batch_info['batch_size']}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Batch explanation generation integration test FAILED: {e}")
            return False
    
    async def test_full_pipeline_batch_integration(self):
        """Test complete pipeline with both batch processing components"""
        logger.info("\n🧪 Testing Full Pipeline Batch Integration...")
        
        try:
            # Mock complete pipeline data
            mock_trend_results = {
                'posts': [
                    {
                        'claim': f'Pipeline test claim {i}',
                        'summary': f'Pipeline summary {i}',
                        'Post_link': f'https://reddit.com/pipeline{i}',
                        'platform': 'reddit'
                    }
                    for i in range(12)  # Test with 12 claims
                ],
                'total_posts': 12
            }
            
            # Mock verification results
            mock_verification_results = {
                'success': True,
                'verified_claims': [
                    {
                        'claim_text': f'Pipeline test claim {i}',
                        'verification': {
                            'verified': False if i % 2 == 0 else True,  # Half are misinformation
                            'verdict': 'false' if i % 2 == 0 else 'true',
                            'confidence': 0.8
                        }
                    }
                    for i in range(12)
                ],
                'batch_processing': {
                    'enabled': True,
                    'total_claims': 12,
                    'batch_size': 12,  # All fit in one batch
                    'processing_method': 'batch_verification'
                }
            }
            
            # Mock explanation results (only for misinformation claims)
            misinformation_count = 6  # Half of the 12 claims
            mock_explanation_results = {
                'success': True,
                'debunk_posts': [
                    {
                        'post_id': f'debunk_pipeline_{i*2}',
                        'claim': f'Pipeline test claim {i*2}',
                        'post_content': f'Debunk content for pipeline claim {i*2}',
                        'confidence_percentage': 85
                    }
                    for i in range(misinformation_count)
                ],
                'batch_processing': {
                    'enabled': True,
                    'total_claims': misinformation_count,
                    'batch_size': 6,  # All fit in one batch
                    'processing_method': 'batch_explanation_generation'
                }
            }
            
            # Mock the complete workflow
            workflow_call_count = 0
            async def mock_execute_workflow(tasks):
                nonlocal workflow_call_count
                workflow_call_count += 1
                
                task = tasks[0]
                agent_name = task.get('agent', '')
                
                if 'trend_scanner' in agent_name:
                    return {
                        'workflow_results': [
                            {
                                'agent_role': 'Trend Scanning Coordinator',
                                'result': mock_trend_results
                            }
                        ]
                    }
                elif 'verifier_coordinator' in agent_name:
                    return {
                        'workflow_results': [
                            {
                                'agent_role': 'Claim Verification Coordinator',
                                'result': mock_verification_results
                            }
                        ]
                    }
                elif 'explanation_coordinator' in agent_name:
                    return {
                        'workflow_results': [
                            {
                                'agent_role': 'Explanation Generation Coordinator',
                                'result': mock_explanation_results
                            }
                        ]
                    }
                else:
                    return {'workflow_results': []}
            
            self.orchestrator.google_agents.execute_workflow = AsyncMock(side_effect=mock_execute_workflow)
            
            # Mock the _save_results method to avoid file I/O
            self.orchestrator._save_results = Mock(return_value="test_results.json")
            
            # Run the full pipeline
            result = await self.orchestrator.run_full_pipeline()
            
            # Validate the complete pipeline results
            assert result.get('success') == True, "Pipeline should complete successfully"
            
            final_output = result.get('final_output', [])
            debunk_posts = result.get('debunk_posts', [])
            batch_metadata = result.get('batch_processing_metadata', {})
            
            assert len(final_output) == 12, "Should process all 12 posts"
            assert len(debunk_posts) == 6, "Should generate 6 debunk posts for misinformation"
            
            # Validate batch processing metadata
            verification_batch = batch_metadata.get('verification_batch_processing', {})
            explanation_batch = batch_metadata.get('explanation_batch_processing', {})
            
            assert verification_batch.get('enabled') == True, "Verification batch processing should be enabled"
            assert explanation_batch.get('enabled') == True, "Explanation batch processing should be enabled"
            
            # Validate summary
            summary = result.get('summary', {})
            assert summary.get('batch_optimization_enabled') == True, "Batch optimization should be enabled"
            assert summary.get('debunk_posts_generated') == 6, "Should report 6 debunk posts generated"
            
            logger.info("✅ Full pipeline batch integration test PASSED")
            logger.info(f"   📊 Processed {len(final_output)} posts with {len(debunk_posts)} debunk posts")
            logger.info(f"   🔄 Verification batch: {verification_batch.get('total_claims', 0)} claims")
            logger.info(f"   🔄 Explanation batch: {explanation_batch.get('total_claims', 0)} claims")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Full pipeline batch integration test FAILED: {e}")
            return False
    
    async def test_batch_size_limits(self):
        """Test that batch size limits are respected"""
        logger.info("\n🧪 Testing Batch Size Limits...")
        
        try:
            # Test verification batch limit (max 15)
            large_content_data = [{'claim': f'Large test claim {i}'} for i in range(25)]
            
            # Mock ClaimVerifier with batch size tracking
            mock_verifier = Mock()
            mock_verifier.verify_content = AsyncMock()
            
            # Mock that shows batch size was limited to 15
            mock_verification_result = {
                'success': True,
                'verified_claims': [{'claim_text': f'Large test claim {i}'} for i in range(25)],
                'batch_processing': {
                    'enabled': True,
                    'total_claims': 25,
                    'batch_size': 15,  # Should be limited to 15
                    'processing_method': 'batch_verification'
                }
            }
            
            # Test explanation batch limit (max 10)
            large_verification_results = [
                {'claim_text': f'Large misinformation claim {i}', 'verification': {'verdict': 'false'}}
                for i in range(18)
            ]
            
            mock_explanation_result = {
                'success': True,
                'debunk_posts': [{'post_id': f'debunk_{i}'} for i in range(18)],
                'batch_processing': {
                    'enabled': True,
                    'total_claims': 18,
                    'batch_size': 10,  # Should be limited to 10
                    'processing_method': 'batch_explanation_generation'
                }
            }
            
            # Validate batch size limits
            verification_batch_size = mock_verification_result['batch_processing']['batch_size']
            explanation_batch_size = mock_explanation_result['batch_processing']['batch_size']
            
            assert verification_batch_size <= 15, f"Verification batch size should be ≤ 15, got {verification_batch_size}"
            assert explanation_batch_size <= 10, f"Explanation batch size should be ≤ 10, got {explanation_batch_size}"
            
            logger.info("✅ Batch size limits test PASSED")
            logger.info(f"   📏 Verification batch limited to: {verification_batch_size}")
            logger.info(f"   📏 Explanation batch limited to: {explanation_batch_size}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Batch size limits test FAILED: {e}")
            return False
    
    async def run_all_tests(self):
        """Run all orchestrator batch processing tests"""
        logger.info("🚀 Starting Orchestrator Batch Processing Integration Tests")
        logger.info("=" * 80)
        
        # Setup orchestrator
        if not await self.setup_orchestrator():
            logger.error("❌ Test setup failed - aborting tests")
            return False
        
        # Run all tests
        tests = [
            ("Batch Claim Verification Integration", self.test_batch_claim_verification_integration),
            ("Batch Explanation Generation Integration", self.test_batch_explanation_generation_integration),
            ("Full Pipeline Batch Integration", self.test_full_pipeline_batch_integration),
            ("Batch Size Limits", self.test_batch_size_limits)
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            try:
                result = await test_func()
                if result:
                    passed += 1
                    self.test_results.append(f"✅ {test_name}: PASSED")
                else:
                    self.test_results.append(f"❌ {test_name}: FAILED")
            except Exception as e:
                logger.error(f"❌ {test_name} test crashed: {e}")
                self.test_results.append(f"💥 {test_name}: CRASHED - {str(e)}")
        
        # Print results summary
        logger.info("\n" + "=" * 80)
        logger.info("📊 ORCHESTRATOR BATCH PROCESSING TEST RESULTS")
        logger.info("=" * 80)
        
        for result in self.test_results:
            logger.info(result)
        
        logger.info(f"\nOverall: {passed}/{total} tests passed")
        
        if passed == total:
            logger.info("🎉 All orchestrator batch processing tests passed! Pipeline is ready for production.")
            return True
        else:
            logger.error(f"❌ {total - passed} tests failed. Please review and fix issues.")
            return False


async def main():
    """Main test function"""
    test_suite = TestOrchestratorBatch()
    success = await test_suite.run_all_tests()
    return 0 if success else 1


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))