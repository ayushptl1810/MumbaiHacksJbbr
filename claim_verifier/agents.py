"""Claim Verifier Agent - Google Agents SDK integration for fact-checking functionality"""

import os
import json
import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional
import google.generativeai as genai

from .tools import TextFactChecker
from .config import config

logger = logging.getLogger(__name__)


class GoogleAgent:
    """Individual Google AI agent with specific role and capabilities (Google Agents SDK pattern)"""
    
    def __init__(self, role: str, goal: str, model: genai.GenerativeModel, tools: List[Any] = None):
        self.role = role
        self.goal = goal
        self.model = model
        self.tools = tools or []
        self.history = []
    
    async def execute_task(self, task_description: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a specific task using this agent"""
        try:
            # If this agent has tools, try to use them first
            if self.tools:
                for tool in self.tools:
                    if hasattr(tool, 'verify') and 'verify' in task_description.lower():
                        # This is a fact-checking task with TextFactChecker tool
                        try:
                            # Extract claim text from task description or context
                            claim_text = ""
                            if context and 'claim_text' in context:
                                claim_text = context['claim_text']
                            elif 'verify:' in task_description:
                                claim_text = task_description.split('verify:', 1)[1].strip()
                            
                            if claim_text:
                                logger.info(f"Agent {self.role} executing fact-check tool for: {claim_text[:100]}...")
                                tool_result = await tool.verify(
                                    text_input=claim_text,
                                    claim_context=context.get('context', 'Unknown context'),
                                    claim_date=context.get('date', datetime.now().isoformat())
                                )
                                
                                result = {
                                    'agent_role': self.role,
                                    'task': task_description,
                                    'result': tool_result,
                                    'timestamp': datetime.now().isoformat(),
                                    'tool_used': True
                                }
                                
                                self.history.append(result)
                                return result
                        except Exception as tool_error:
                            logger.error(f"Tool execution failed: {tool_error}")
                            # Fall back to text response
                            pass
            
            # Clean context to avoid circular references
            safe_context = {}
            if context:
                # Handle different context types
                if isinstance(context, dict):
                    for key, value in context.items():
                        try:
                            if isinstance(value, (str, int, float, bool, list, dict)):
                                if isinstance(value, dict):
                                    safe_context[key] = f"Dict with {len(value)} keys"
                                elif isinstance(value, list):
                                    safe_context[key] = f"List with {len(value)} items"
                                else:
                                    safe_context[key] = value
                            else:
                                safe_context[key] = f"<{type(value).__name__} object>"
                        except:
                            safe_context[key] = "<unable to serialize>"
                elif isinstance(context, list):
                    # If context is a list, convert it to a dict format
                    safe_context = {
                        'context_type': 'list_data',
                        'items_count': len(context),
                        'context_summary': f"List with {len(context)} items"
                    }
                else:
                    # For other types, create a simple representation
                    safe_context = {
                        'context_type': type(context).__name__,
                        'context_value': str(context)[:200]  # Truncate long values
                    }
            
            # Create context-aware prompt
            context_text = ""
            if safe_context:
                context_summary = "\n".join([f"- {k}: {v}" for k, v in safe_context.items()])
                context_text = f"Context information:\n{context_summary}\n\n"
            
            prompt = f"""
You are an expert {self.role}.

Your goal: {self.goal}

{context_text}Task: {task_description}

Please provide a comprehensive response that addresses the task while staying within your role expertise.
"""
            
            response = self.model.generate_content(prompt)
            
            result = {
                'agent_role': self.role,
                'task': task_description,
                'result': response.text,
                'context_summary': safe_context,
                'timestamp': datetime.now().isoformat(),
                'tool_used': False
            }
            
            self.history.append(result)
            return result
            
        except Exception as e:
            logger.error(f"Agent {self.role} task execution failed: {e}")
            error_result = {
                'agent_role': self.role,
                'task': task_description,
                'result': f"Task execution failed: {str(e)}",
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'tool_used': False
            }
            self.history.append(error_result)
            return error_result


class GoogleAgentsOrchestrator:
    """Google Agents SDK orchestrator for managing multiple agents"""
    
    def __init__(self, gemini_api_key: Optional[str] = None):
        self.gemini_api_key = gemini_api_key or config.GEMINI_API_KEY
        if not self.gemini_api_key:
            raise ValueError("Gemini API key is required for Google Agents orchestration")
        
        # Configure Google AI
        genai.configure(api_key=self.gemini_api_key)
        self.model = genai.GenerativeModel(config.GEMINI_MODEL)
        
        # Agent registry
        self.agents = {}
        self.workflow_history = []
        
        logger.info("Google Agents Orchestrator initialized successfully")
    
    def create_agent(self, name: str, role: str, goal: str, tools: List[Any] = None) -> GoogleAgent:
        """Create and register a new Google agent"""
        agent = GoogleAgent(role, goal, self.model, tools)
        self.agents[name] = agent
        logger.info(f"Created Google Agent: {name} - {role}")
        return agent
    
    async def execute_workflow(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute a workflow with multiple agents and tasks"""
        try:
            logger.info(f"Starting Google Agents workflow with {len(tasks)} tasks")
            
            workflow_results = []
            context = {}
            
            for i, task in enumerate(tasks):
                agent_name = task.get('agent')
                task_description = task.get('task')
                task_context = task.get('context', {})
                
                # Merge global context with task-specific context
                merged_context = {**context, **task_context}
                
                if agent_name not in self.agents:
                    error_result = {
                        'agent_role': agent_name,
                        'task': task_description,
                        'result': f"Agent '{agent_name}' not found",
                        'error': f"Agent not registered: {agent_name}",
                        'timestamp': datetime.now().isoformat()
                    }
                    workflow_results.append(error_result)
                    continue
                
                logger.info(f"Executing task {i+1}/{len(tasks)}: {agent_name} - {task_description}")
                
                # Execute task with agent (now async)
                result = await self.agents[agent_name].execute_task(task_description, merged_context)
                workflow_results.append(result)
                
                # Update context with result for next tasks
                context['last_result'] = result
                context[f'{agent_name}_result'] = result
            
            # Create final workflow summary
            summary = self._create_workflow_summary(workflow_results)
            
            final_result = {
                'workflow_id': f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'total_tasks': len(tasks),
                'completed_tasks': len([r for r in workflow_results if not r.get('error')]),
                'failed_tasks': len([r for r in workflow_results if r.get('error')]),
                'workflow_results': workflow_results,
                'summary': summary,
                'timestamp': datetime.now().isoformat()
            }
            
            self.workflow_history.append(final_result)
            logger.info(f"Workflow completed: {final_result['completed_tasks']}/{final_result['total_tasks']} tasks successful")
            
            return final_result
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            error_result = {
                'workflow_id': f"workflow_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'error': str(e),
                'workflow_results': workflow_results if 'workflow_results' in locals() else [],
                'timestamp': datetime.now().isoformat()
            }
            self.workflow_history.append(error_result)
            return error_result
    
    def _create_workflow_summary(self, workflow_results: List[Dict[str, Any]]) -> str:
        """Create a summary of the workflow execution"""
        try:
            successful_tasks = [r for r in workflow_results if not r.get('error')]
            failed_tasks = [r for r in workflow_results if r.get('error')]
            
            summary_parts = []
            summary_parts.append(f"Workflow Summary: {len(successful_tasks)}/{len(workflow_results)} tasks completed successfully")
            
            if successful_tasks:
                summary_parts.append("Successful tasks:")
                for result in successful_tasks:
                    agent_role = result.get('agent_role', 'Unknown')
                    task_result = result.get('result', 'No result')
                    if isinstance(task_result, str):
                        task_summary = task_result[:100] + "..." if len(task_result) > 100 else task_result
                    else:
                        task_summary = f"<{type(task_result).__name__} result>"
                    summary_parts.append(f"- {agent_role}: {task_summary}")
            
            if failed_tasks:
                summary_parts.append("Failed tasks:")
                for result in failed_tasks:
                    agent_role = result.get('agent_role', 'Unknown')
                    error = result.get('error', 'Unknown error')
                    summary_parts.append(f"- {agent_role}: {error}")
            
            return "\n".join(summary_parts)
            
        except Exception as e:
            logger.error(f"Failed to create workflow summary: {e}")
            return f"Summary generation failed: {str(e)}"


class ClaimVerifierOrchestrator:
    """Specialized orchestrator for claim verification using Google Agents SDK"""
    
    def __init__(self, gemini_api_key: Optional[str] = None):
        self.gemini_api_key = gemini_api_key or config.GEMINI_API_KEY
        if not self.gemini_api_key:
            raise ValueError("Gemini API key is required")
        
        # Initialize Google Agents orchestrator
        self.google_agents = GoogleAgentsOrchestrator(self.gemini_api_key)
        
        # Initialize fact checker tool
        self.fact_checker = TextFactChecker()
        
        # Setup specialized agents
        self._setup_claim_verification_agents()
        
        logger.info("Claim Verifier Orchestrator initialized with Google Agents SDK")
    
    def _setup_claim_verification_agents(self):
        """Setup specialized Google agents for claim verification"""
        
        # Create claim extraction agent
        self.claim_extractor = self.google_agents.create_agent(
            name="claim_extractor",
            role="Claim Extraction Specialist",
            goal="Extract verifiable factual claims from content using advanced NLP analysis",
            tools=[]
        )
        
        # Create fact verification agent with fact-checking tool
        self.fact_verifier = self.google_agents.create_agent(
            name="fact_verifier", 
            role="Fact Verification Specialist",
            goal="Verify extracted claims against reliable sources using Google Custom Search and AI analysis",
            tools=[self.fact_checker]
        )
        
        # Create priority assessment agent
        self.priority_assessor = self.google_agents.create_agent(
            name="priority_assessor",
            role="Priority Assessment Specialist", 
            goal="Assess and prioritize claims based on risk, impact, and verification confidence",
            tools=[]
        )
        
        # Create report generation agent
        self.report_generator = self.google_agents.create_agent(
            name="report_generator",
            role="Report Generation Specialist",
            goal="Generate comprehensive fact-checking reports with actionable insights",
            tools=[]
        )
        
        logger.info("Claim verification agents setup completed with Google Agents SDK")
    
    async def verify_content(self, content_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run complete claim verification workflow using Google Agents with batch processing"""
        try:
            logger.info(f"Starting Google Agents claim verification for {len(content_data)} content items with batch processing")
            
            verified_claims = []
            
            # Prepare all claims first
            all_claims = []
            for i, content_item in enumerate(content_data):
                try:
                    title = content_item.get('title', f'Content item {i+1}')
                    content_text = content_item.get('content', '')
                    source = content_item.get('source', 'Unknown source')
                    
                    # Extract claim from title if it starts with "Claim:"
                    claim_text = ""
                    if title.startswith("Claim:"):
                        claim_text = title[6:].strip()  # Remove "Claim:" prefix
                    elif 'extracted_claim' in content_item.get('claim_metadata', {}):
                        claim_text = content_item['claim_metadata']['extracted_claim']
                    else:
                        # Use content as claim if no explicit claim found
                        claim_text = content_text[:200]  # Limit length
                    
                    if claim_text:
                        all_claims.append({
                            'text_input': claim_text,
                            'claim_context': content_text,
                            'claim_date': content_item.get('timestamp', datetime.now().isoformat()),
                            'original_content': content_item,
                            'source': source
                        })
                        
                except Exception as e:
                    logger.error(f"Failed to prepare content item {i+1}: {e}")
                    # Add error result
                    verified_claims.append({
                        'claim_text': content_item.get('title', 'Unknown claim'),
                        'verification': {
                            'verified': False,
                            'verdict': 'error',
                            'message': f'Preparation failed: {str(e)}',
                            'error': str(e)
                        },
                        'verification_timestamp': datetime.now().isoformat()
                    })
            
            if not all_claims:
                logger.warning("No valid claims found to verify")
                return {
                    'success': True,
                    'message': 'No valid claims found to verify',
                    'verified_claims': verified_claims,
                    'summary': {
                        'total_claims': len(verified_claims),
                        'successfully_verified': 0,
                        'verification_errors': len(verified_claims)
                    },
                    'timestamp': datetime.now().isoformat()
                }
            
            # Process claims in batches of max 15
            BATCH_SIZE = 15
            logger.info(f"Processing {len(all_claims)} claims in batches of {BATCH_SIZE}")
            
            for batch_start in range(0, len(all_claims), BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, len(all_claims))
                batch_claims = all_claims[batch_start:batch_end]
                
                logger.info(f"Processing batch {batch_start//BATCH_SIZE + 1}: claims {batch_start+1}-{batch_end}")
                
                try:
                    # Use batch verification
                    batch_verification_results = await self.fact_checker.verify_batch(batch_claims)
                    
                    # Process batch results
                    for claim_data, verification_result in zip(batch_claims, batch_verification_results):
                        verified_claim = {
                            'claim_text': claim_data['text_input'],
                            'content_summary': claim_data['claim_context'][:300],
                            'source': claim_data['source'],
                            'verification': verification_result,
                            'claim_metadata': claim_data['original_content'].get('claim_metadata', {}),
                            'verification_timestamp': datetime.now().isoformat()
                        }
                        verified_claims.append(verified_claim)
                        
                except Exception as e:
                    logger.error(f"Batch verification failed for batch {batch_start//BATCH_SIZE + 1}: {e}")
                    # Add error results for the entire batch
                    for claim_data in batch_claims:
                        verified_claims.append({
                            'claim_text': claim_data['text_input'],
                            'verification': {
                                'verified': False,
                                'verdict': 'error',
                                'message': f'Batch verification failed: {str(e)}',
                                'error': str(e)
                            },
                            'verification_timestamp': datetime.now().isoformat()
                        })
            
            logger.info(f"Batch verification completed: {len(verified_claims)} total claims processed")
            
            return {
                'success': True,
                'message': f'Successfully verified {len(verified_claims)} claims using batch Google Agents fact-checking',
                'verified_claims': verified_claims,
                'summary': {
                    'total_claims': len(verified_claims),
                    'successfully_verified': len([c for c in verified_claims if c.get('verification', {}).get('verified', False)]),
                    'verification_errors': len([c for c in verified_claims if 'error' in c.get('verification', {})]),
                    'batch_size_used': BATCH_SIZE,
                    'total_batches': (len(all_claims) + BATCH_SIZE - 1) // BATCH_SIZE
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Google Agents batch claim verification workflow failed: {e}")
            return {
                'success': False,
                'message': f'Batch verification workflow failed: {str(e)}',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _process_verification_workflow(self, workflow_result: Dict[str, Any], original_content: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process Google Agents workflow results into claim verification format"""
        try:
            workflow_results = workflow_result.get('workflow_results', [])
            
            # Extract results from each agent
            extracted_claims = []
            verified_claims = []
            prioritized_claims = []
            final_report = {}
            
            for result in workflow_results:
                agent_role = result.get('agent_role', '')
                
                if 'Claim Extraction' in agent_role:
                    # Process claim extraction results
                    extracted_claims = self._extract_claims_from_result(result, original_content)
                
                elif 'Fact Verification' in agent_role:
                    # Process verification results  
                    verified_claims = self._extract_verification_from_result(result, extracted_claims)
                
                elif 'Priority Assessment' in agent_role:
                    # Process prioritization results
                    prioritized_claims = self._extract_prioritization_from_result(result, verified_claims)
                
                elif 'Report Generation' in agent_role:
                    # Process final report
                    final_report = self._extract_report_from_result(result, prioritized_claims)
            
            return {
                'success': True,
                'message': f'Successfully processed {len(original_content)} content items through Google Agents workflow',
                'workflow_results': workflow_results,
                'final_report': final_report,
                'summary': {
                    'content_items_processed': len(original_content),
                    'claims_extracted': len(extracted_claims),
                    'claims_verified': len(verified_claims),
                    'high_priority_claims': len([c for c in prioritized_claims if c.get('priority_level') == 'high'])
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to process verification workflow: {e}")
            return {
                'success': False,
                'message': f'Workflow processing failed: {str(e)}',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _extract_claims_from_result(self, result: Dict[str, Any], content_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract claims from claim extraction agent result"""
        # Implementation would parse the agent's response and structure claims
        # For now, return basic structure
        return [
            {
                'claim_text': item.get('title', 'No claim identified'),
                'context': item.get('content', ''),
                'source_content': item,
                'extraction_timestamp': datetime.now().isoformat()
            }
            for item in content_data
        ]
    
    def _extract_verification_from_result(self, result: Dict[str, Any], claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract verification results from fact verification agent"""
        # Add basic verification structure to claims
        for claim in claims:
            claim['verification'] = {
                'verified': False,
                'verdict': 'not_verified', 
                'message': 'Google Agents verification in progress',
                'details': {}
            }
        return claims
    
    def _extract_prioritization_from_result(self, result: Dict[str, Any], claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract prioritization from priority assessment agent"""
        # Add priority scoring to claims
        for i, claim in enumerate(claims):
            claim['priority_score'] = 50.0  # Default medium priority
            claim['priority_level'] = 'medium'
            claim['prioritization_timestamp'] = datetime.now().isoformat()
        return claims
    
    def _extract_report_from_result(self, result: Dict[str, Any], claims: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract final report from report generation agent"""
        return {
            'report': result.get('result', 'Report generation completed'),
            'summary_stats': {
                'total_claims': len(claims),
                'high_priority': len([c for c in claims if c.get('priority_level') == 'high']),
                'medium_priority': len([c for c in claims if c.get('priority_level') == 'medium']),
                'low_priority': len([c for c in claims if c.get('priority_level') == 'low'])
            },
            'report_timestamp': datetime.now().isoformat()
        }
    
    async def quick_verify(self, claim_text: str, context: str = "Direct input") -> Dict[str, Any]:
        """Quick verification of a single claim using Google Agents"""
        try:
            logger.info(f"Quick Google Agents verification: {claim_text[:100]}...")
            
            # Define single-task workflow for quick verification
            workflow_tasks = [
                {
                    'agent': 'fact_verifier',
                    'task': f'verify: {claim_text}',
                    'context': {
                        'claim_text': claim_text,
                        'context': context,
                        'date': datetime.now().isoformat(),
                        'quick_verify': True
                    }
                }
            ]
            
            # Execute single-agent workflow
            workflow_result = await self.google_agents.execute_workflow(workflow_tasks)
            
            return {
                'success': True,
                'claim_text': claim_text,
                'verification': workflow_result.get('workflow_results', [{}])[0].get('result', {}),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Quick Google Agents verification failed: {e}")
            return {
                'success': False,
                'claim_text': claim_text,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
        """Execute a claim verification task"""
        try:
            if "extract_claims" in task_description.lower():
                return await self._extract_claims_task(claims_data)
            elif "verify_claims" in task_description.lower():
                return await self._verify_claims_task(claims_data)
            elif "prioritize_claims" in task_description.lower():
                return await self._prioritize_claims_task(claims_data)
            elif "generate_report" in task_description.lower():
                return await self._generate_report_task(claims_data)
            else:
                return await self._general_analysis_task(task_description, claims_data)
                
        except Exception as e:
            logger.error(f"Agent {self.role} task execution failed: {e}")
            return {
                'agent_role': self.role,
                'task': task_description,
                'result': f"Task execution failed: {str(e)}",
                'has_error': True,
                'error_message': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def _extract_claims_task(self, content_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract verifiable claims from content"""
        extracted_claims = []
        
        for content in content_data:
            prompt = f"""
You are a claim extraction expert. Extract specific, verifiable factual claims from the following content.

CONTENT:
Title: {content.get('title', '')}
Text: {content.get('content', '')}
Source: {content.get('source', 'Unknown')}

EXTRACT CLAIMS THAT ARE:
1. Specific factual assertions
2. Verifiable through external sources
3. Not opinions or subjective statements
4. Potentially controversial or questionable

For each claim, provide:
- The exact claim text
- Context within the content
- Why it's worth fact-checking

Respond in JSON format:
{{
    "claims": [
        {{
            "claim_text": "Specific factual claim here",
            "context": "Surrounding context",
            "reason_to_check": "Why this needs verification",
            "priority": "high|medium|low"
        }}
    ]
}}
"""
            
            try:
                response = self.model.generate_content(prompt)
                response_text = response.text.strip()
                
                # Clean JSON response
                if response_text.startswith('```json'):
                    response_text = response_text.replace('```json', '').replace('```', '').strip()
                elif response_text.startswith('```'):
                    response_text = response_text.replace('```', '').strip()
                
                claims_result = json.loads(response_text)
                
                # Add metadata to each claim
                for claim in claims_result.get('claims', []):
                    claim['source_content'] = content
                    claim['extraction_timestamp'] = datetime.now().isoformat()
                    extracted_claims.append(claim)
                    
            except Exception as e:
                logger.warning(f"Failed to extract claims from content: {e}")
                continue
        
        result = {
            'agent_role': self.role,
            'task': 'extract_claims',
            'result': {
                'extracted_claims': extracted_claims,
                'total_claims': len(extracted_claims),
                'sources_processed': len(content_data)
            },
            'has_error': False,
            'timestamp': datetime.now().isoformat()
        }
        
        self.history.append(result)
        return result
    
    async def _verify_claims_task(self, claims_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Verify extracted claims using fact-checking"""
        verified_claims = []
        
        for claim in claims_data:
            try:
                claim_text = claim.get('claim_text', '')
                context = claim.get('context', 'Unknown context')
                
                if not claim_text:
                    continue
                
                logger.info(f"Verifying claim: {claim_text[:100]}...")
                
                verification_result = await self.fact_checker.verify(
                    text_input=claim_text,
                    claim_context=context,
                    claim_date=claim.get('extraction_timestamp', 'Unknown')
                )
                
                # Combine original claim data with verification results
                verified_claim = {
                    **claim,
                    'verification': verification_result,
                    'verification_timestamp': datetime.now().isoformat()
                }
                
                verified_claims.append(verified_claim)
                
            except Exception as e:
                logger.error(f"Failed to verify claim: {e}")
                # Add error information to claim
                error_claim = {
                    **claim,
                    'verification': {
                        'verified': False,
                        'verdict': 'error',
                        'message': f"Verification failed: {str(e)}",
                        'details': {'error': str(e)}
                    },
                    'verification_timestamp': datetime.now().isoformat()
                }
                verified_claims.append(error_claim)
        
        result = {
            'agent_role': self.role,
            'task': 'verify_claims',
            'result': {
                'verified_claims': verified_claims,
                'total_verified': len(verified_claims),
                'success_rate': len([c for c in verified_claims if c['verification']['verdict'] != 'error']) / len(verified_claims) if verified_claims else 0
            },
            'has_error': False,
            'timestamp': datetime.now().isoformat()
        }
        
        self.history.append(result)
        return result
    
    async def _prioritize_claims_task(self, verified_claims_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Prioritize verified claims based on various factors"""
        prioritized_claims = []
        
        for claim in verified_claims_data:
            verification = claim.get('verification', {})
            verdict = verification.get('verdict', 'unknown')
            confidence = verification.get('details', {}).get('analysis', {}).get('confidence', 'low')
            
            # Calculate priority score
            priority_score = self._calculate_priority_score(claim, verification)
            
            prioritized_claim = {
                **claim,
                'priority_score': priority_score,
                'priority_level': self._get_priority_level(priority_score),
                'prioritization_timestamp': datetime.now().isoformat()
            }
            
            prioritized_claims.append(prioritized_claim)
        
        # Sort by priority score (highest first)
        prioritized_claims.sort(key=lambda x: x['priority_score'], reverse=True)
        
        result = {
            'agent_role': self.role,
            'task': 'prioritize_claims',
            'result': {
                'prioritized_claims': prioritized_claims,
                'high_priority_count': len([c for c in prioritized_claims if c['priority_level'] == 'high']),
                'medium_priority_count': len([c for c in prioritized_claims if c['priority_level'] == 'medium']),
                'low_priority_count': len([c for c in prioritized_claims if c['priority_level'] == 'low'])
            },
            'has_error': False,
            'timestamp': datetime.now().isoformat()
        }
        
        self.history.append(result)
        return result
    
    async def _generate_report_task(self, prioritized_claims_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a comprehensive fact-checking report"""
        if not prioritized_claims_data:
            return {
                'agent_role': self.role,
                'task': 'generate_report',
                'result': {
                    'report': "No claims to report on",
                    'summary': "No fact-checking data available"
                },
                'has_error': False,
                'timestamp': datetime.now().isoformat()
            }
        
        # Prepare data for report generation
        high_priority = [c for c in prioritized_claims_data if c.get('priority_level') == 'high']
        false_claims = [c for c in prioritized_claims_data if c.get('verification', {}).get('verdict') == 'false']
        uncertain_claims = [c for c in prioritized_claims_data if c.get('verification', {}).get('verdict') == 'uncertain']
        
        report_prompt = f"""
Generate a comprehensive fact-checking report based on the following verified claims data.

STATISTICS:
- Total claims analyzed: {len(prioritized_claims_data)}
- High priority claims: {len(high_priority)}
- False claims detected: {len(false_claims)}
- Uncertain claims: {len(uncertain_claims)}

HIGH PRIORITY CLAIMS:
{json.dumps([{'claim': c.get('claim_text', ''), 'verdict': c.get('verification', {}).get('verdict', 'unknown'), 'message': c.get('verification', {}).get('message', '')} for c in high_priority[:5]], indent=2)}

Create a professional fact-checking report that includes:
1. Executive Summary
2. Key Findings
3. High-Risk Claims Analysis
4. Recommendations
5. Methodology Notes

Format as a structured report suitable for stakeholders.
"""
        
        try:
            response = self.model.generate_content(report_prompt)
            report_content = response.text.strip()
            
            # Generate summary statistics
            summary_stats = {
                'total_claims': len(prioritized_claims_data),
                'verdict_distribution': {
                    'true': len([c for c in prioritized_claims_data if c.get('verification', {}).get('verdict') == 'true']),
                    'false': len(false_claims),
                    'mixed': len([c for c in prioritized_claims_data if c.get('verification', {}).get('verdict') == 'mixed']),
                    'uncertain': len(uncertain_claims),
                    'error': len([c for c in prioritized_claims_data if c.get('verification', {}).get('verdict') == 'error'])
                },
                'priority_distribution': {
                    'high': len(high_priority),
                    'medium': len([c for c in prioritized_claims_data if c.get('priority_level') == 'medium']),
                    'low': len([c for c in prioritized_claims_data if c.get('priority_level') == 'low'])
                }
            }
            
            result = {
                'agent_role': self.role,
                'task': 'generate_report',
                'result': {
                    'report': report_content,
                    'summary_stats': summary_stats,
                    'high_priority_claims': high_priority,
                    'false_claims': false_claims,
                    'report_timestamp': datetime.now().isoformat()
                },
                'has_error': False,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
            result = {
                'agent_role': self.role,
                'task': 'generate_report',
                'result': {
                    'report': f"Report generation failed: {str(e)}",
                    'summary_stats': {},
                    'error': str(e)
                },
                'has_error': True,
                'error_message': str(e),
                'timestamp': datetime.now().isoformat()
            }
        
        self.history.append(result)
        return result
    
    async def _general_analysis_task(self, task_description: str, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Handle general analysis tasks"""
        prompt = f"""
You are a fact-checking expert. Analyze the following data according to this task:

TASK: {task_description}

DATA:
{json.dumps(data[:3], indent=2)}... (showing first 3 items of {len(data)} total)

Provide a comprehensive analysis addressing the task requirements.
"""
        
        try:
            response = self.model.generate_content(prompt)
            analysis_result = response.text.strip()
            
            result = {
                'agent_role': self.role,
                'task': task_description,
                'result': analysis_result,
                'has_error': False,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            result = {
                'agent_role': self.role,
                'task': task_description,
                'result': f"Analysis failed: {str(e)}",
                'has_error': True,
                'error_message': str(e),
                'timestamp': datetime.now().isoformat()
            }
        
        self.history.append(result)
        return result
    
    def _calculate_priority_score(self, claim: Dict[str, Any], verification: Dict[str, Any]) -> float:
        """Calculate priority score for a claim (0-100)"""
        score = 0.0
        
        # Base score from original priority
        priority_scores = {'high': 30, 'medium': 20, 'low': 10}
        score += priority_scores.get(claim.get('priority', 'low'), 10)
        
        # Verdict impact
        verdict = verification.get('verdict', 'unknown')
        verdict_scores = {'false': 40, 'mixed': 30, 'uncertain': 20, 'true': 10, 'error': 5}
        score += verdict_scores.get(verdict, 5)
        
        # Confidence impact
        confidence = verification.get('details', {}).get('analysis', {}).get('confidence', 'low')
        confidence_scores = {'high': 20, 'medium': 15, 'low': 10}
        score += confidence_scores.get(confidence, 10)
        
        # Verification success bonus
        if verification.get('verified', False):
            score += 10
        
        return min(100.0, score)
    
    def _get_priority_level(self, score: float) -> str:
        """Convert priority score to level"""
        if score >= 70:
            return 'high'
        elif score >= 40:
            return 'medium'
        else:
            return 'low'