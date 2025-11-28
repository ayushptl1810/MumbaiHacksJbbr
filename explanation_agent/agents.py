"""Explanation Agent - Creates structured debunk posts using Google Agents SDK"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import google.generativeai as genai
from urllib.parse import urlparse

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
    
    def execute_task(self, task_description: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a specific task using this agent"""
        try:
            # If this agent has tools, try to use them first
            if self.tools:
                for tool in self.tools:
                    if hasattr(tool, 'process'):
                        # This is a processing tool - try to use it based on agent role
                        try:
                            logger.info(f"Agent {self.role} executing tool...")
                            tool_result = tool.process(context)
                            
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
            
            # Create context-aware prompt
            context_text = ""
            if context:
                if 'verification_result' in context:
                    vr = context['verification_result']
                    context_text = f"""
Verification Results:
- Claim: {vr.get('claim_text', 'Unknown')}
- Verdict: {vr.get('verdict', 'unknown')}
- Confidence: {vr.get('confidence', 'unknown')}
- Reasoning: {vr.get('reasoning', 'No reasoning')}
- Sources: {len(vr.get('sources', {}).get('links', []))} sources available
"""
                else:
                    context_text = f"Context: {json.dumps(context, indent=2)}"
            
            prompt = f"""
You are a {self.role} with the goal: {self.goal}

Current task: {task_description}

{context_text}

Execute this task thoroughly and provide detailed results in JSON format.
"""
            
            # Execute with Gemini
            response = self.model.generate_content(prompt)
            response_text = getattr(response, 'text', str(response))
            
            result = {
                'agent_role': self.role,
                'task': task_description,
                'result': response_text,
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
                'has_error': True,
                'error_message': str(e),
                'timestamp': datetime.now().isoformat()
            }
            self.history.append(error_result)
            return error_result
    
    def process_batch(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process a batch of items using available tools"""
        try:
            # If this agent has tools, try to use them for batch processing
            if self.tools:
                for tool in self.tools:
                    if hasattr(tool, 'process_batch'):
                        # This tool supports batch processing
                        try:
                            logger.info(f"Agent {self.role} executing batch processing tool...")
                            tool_result = tool.process_batch(context)
                            return tool_result
                            
                        except Exception as tool_error:
                            logger.error(f"Batch tool execution failed: {tool_error}")
                            return {
                                'success': False,
                                'error': str(tool_error),
                                'message': f'Batch processing failed: {str(tool_error)}'
                            }
            
            # No batch processing tools available
            return {
                'success': False,
                'error': 'No batch processing tools available',
                'message': 'This agent does not support batch processing'
            }
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'message': f'Batch processing error: {str(e)}'
            }


class GoogleAgentsOrchestrator:
    """Orchestrator for managing multiple Google Agents in explanation workflow"""
    
    def __init__(self, api_key: str):
        """Initialize the orchestrator with Google Agents"""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(config.GEMINI_MODEL)
        self.agents = {}
        self.workflow_history = []
        
        logger.info("Google Agents Orchestrator initialized for Explanation Agent")
    
    def create_agent(self, name: str, role: str, goal: str, tools: List[Any] = None) -> GoogleAgent:
        """Create and register a new Google Agent"""
        agent = GoogleAgent(role, goal, self.model, tools)
        self.agents[name] = agent
        logger.info(f"Created Google Agent: {name} - {role}")
        return agent
    
    def execute_workflow(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute a workflow of tasks using Google Agents"""
        workflow_results = []
        context = {}
        
        logger.info(f"Starting Google Agents workflow with {len(tasks)} tasks")
        
        for i, task in enumerate(tasks):
            agent_name = task['agent']
            task_description = task['description']
            
            if agent_name not in self.agents:
                raise ValueError(f"Agent '{agent_name}' not found")
            
            # Update context with previous results
            if workflow_results:
                context['previous_results'] = workflow_results
                context['last_result'] = workflow_results[-1]
            
            # Add task-specific context
            if 'context' in task:
                context.update(task['context'])
            
            # Execute task
            logger.info(f"Executing task {i+1}/{len(tasks)} with agent '{agent_name}'")
            try:
                result = self.agents[agent_name].execute_task(task_description, context)
                workflow_results.append(result)
                logger.info(f"Task {i+1} completed successfully by '{agent_name}'")
            except Exception as e:
                logger.error(f"Task {i+1} failed for agent '{agent_name}': {e}")
                error_result = {
                    'agent_role': agent_name,
                    'task': task_description,
                    'result': f"Task execution failed: {str(e)}",
                    'has_error': True,
                    'error_message': str(e),
                    'timestamp': datetime.now().isoformat()
                }
                workflow_results.append(error_result)
        
        # Create final workflow summary
        final_result = {
            'workflow_type': 'explanation_generation',
            'total_tasks': len(tasks),
            'results': workflow_results,
            'timestamp': datetime.now().isoformat()
        }
        
        self.workflow_history.append(final_result)
        return final_result


class ContentGeneratorTool:
    """Tool for generating explanation content"""
    
    def __init__(self, model: genai.GenerativeModel):
        self.model = model
    
    def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate content for debunk post"""
        verification_result = context.get('verification_result', {})
        
        claim_text = verification_result.get('claim_text', 'Unknown claim')
        verdict = verification_result.get('verdict', 'uncertain')
        verified = verification_result.get('verified', False)
        reasoning = verification_result.get('reasoning', 'No reasoning provided')
        message = verification_result.get('message', 'No message provided')
        confidence = verification_result.get('confidence', 'medium')
        sources = verification_result.get('sources', {})
        
        # Convert confidence to percentage
        confidence_percentage = self._convert_confidence_to_percentage(confidence)
        
        sources_text = ""
        if sources.get('links') and sources.get('titles'):
            for i, (link, title) in enumerate(zip(sources['links'][:3], sources['titles'][:3]), 1):
                sources_text += f"{i}. {title}\n   Source: {link}\n\n"
        
        prompt = f"""
You are an expert content generator creating a clear, informative debunk post for the public.

CLAIM TO ANALYZE: "{claim_text}"

FACT-CHECK RESULTS:
- Verdict: {verdict}
- Verified: {verified}
- Confidence: {confidence_percentage}%
- Reasoning: {reasoning}
- Summary: {message}

SOURCES USED:
{sources_text}

Create a structured debunk post that:
1. Has a clear, engaging heading (max {config.MAX_HEADING_LENGTH} characters)
2. Provides a detailed but accessible body explanation (max {config.MAX_BODY_LENGTH} characters)
3. Includes a concise summary

Guidelines:
- Use clear, non-technical language
- Be authoritative but not condescending
- Include specific facts and evidence
- Acknowledge uncertainty when appropriate
- Focus on educating rather than just contradicting

Respond in this exact JSON format:
{{
    "heading": "Clear, engaging headline about the fact-check result",
    "body": "Detailed explanation of what the evidence shows, why the claim is true/false/mixed, and what people should know. Include specific facts and context.",
    "summary": "Concise one-sentence summary of the key finding"
}}
"""
        
        try:
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Clean up response
            if response_text.startswith('```json'):
                response_text = response_text.replace('```json', '').replace('```', '').strip()
            elif response_text.startswith('```'):
                response_text = response_text.replace('```', '').strip()
            
            content_data = json.loads(response_text)
            
            # Validate and truncate if needed
            content_data['heading'] = content_data.get('heading', '')[:config.MAX_HEADING_LENGTH]
            content_data['body'] = content_data.get('body', '')[:config.MAX_BODY_LENGTH]
            
            return {
                'success': True,
                'content': content_data,
                'confidence_percentage': confidence_percentage
            }
            
        except Exception as e:
            logger.error(f"Content generation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'fallback_content': {
                    'heading': self._generate_fallback_heading(verdict, claim_text),
                    'body': self._generate_fallback_body(reasoning, message),
                    'summary': message
                },
                'confidence_percentage': confidence_percentage
            }
    
    def process_batch(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate content for multiple debunk posts in batch"""
        verification_results = context.get('verification_results', [])
        
        if not verification_results:
            return {
                'success': False,
                'error': 'No verification results provided for batch processing',
                'batch_contents': []
            }
        
        # Prepare batch prompt
        batch_prompt = f"""
You are an expert content generator creating clear, informative debunk posts for the public.

You will analyze {len(verification_results)} claims and create structured debunk posts for each.

CLAIMS TO ANALYZE:

"""
        
        # Add each claim to the prompt
        for i, verification_result in enumerate(verification_results, 1):
            claim_text = verification_result.get('claim_text', 'Unknown claim')
            verdict = verification_result.get('verdict', 'uncertain')
            verified = verification_result.get('verified', False)
            reasoning = verification_result.get('reasoning', 'No reasoning provided')
            message = verification_result.get('message', 'No message provided')
            confidence = verification_result.get('confidence', 'medium')
            sources = verification_result.get('sources', {})
            
            confidence_percentage = self._convert_confidence_to_percentage(confidence)
            
            sources_text = ""
            if sources.get('links') and sources.get('titles'):
                for j, (link, title) in enumerate(zip(sources['links'][:2], sources['titles'][:2]), 1):
                    sources_text += f"   {j}. {title}\n      Source: {link}\n"
            
            batch_prompt += f"""
--- CLAIM {i} ---
CLAIM: "{claim_text}"
VERDICT: {verdict}
VERIFIED: {verified}
CONFIDENCE: {confidence_percentage}%
REASONING: {reasoning}
SUMMARY: {message}
SOURCES:
{sources_text}

"""
        
        batch_prompt += f"""

For each claim, create a structured debunk post with:
1. Clear, engaging heading (max {config.MAX_HEADING_LENGTH} characters)
2. Detailed but accessible body explanation (max {config.MAX_BODY_LENGTH} characters)  
3. Concise summary

Guidelines:
- Use clear, non-technical language
- Be authoritative but not condescending
- Include specific facts and evidence
- Acknowledge uncertainty when appropriate
- Focus on educating rather than just contradicting

Respond with a JSON array containing exactly {len(verification_results)} objects in the same order as the claims above:

Example format:
[
    {{
        "heading": "Clear, engaging headline about the fact-check result",
        "body": "Detailed explanation of what the evidence shows...",
        "summary": "Concise one-sentence summary of the key finding"
    }},
    {{
        "heading": "Another clear headline...",
        "body": "Another detailed explanation...", 
        "summary": "Another summary"
    }}
]
"""
        
        try:
            response = self.model.generate_content(batch_prompt)
            response_text = response.text.strip()
            
            # Log the raw response for debugging
            logger.info(f"Raw Gemini response (first 500 chars): {response_text[:500]}")
            
            # Clean up response
            if response_text.startswith('```json'):
                response_text = response_text.replace('```json', '').replace('```', '').strip()
            elif response_text.startswith('```'):
                response_text = response_text.replace('```', '').strip()
            
            # Log cleaned response for debugging
            logger.info(f"Cleaned response (first 500 chars): {response_text[:500]}")
            
            # Try to extract JSON array from response if it's not clean JSON
            import re
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                json_text = json_match.group(0)
                logger.info(f"Extracted JSON: {json_text[:500]}")
                batch_contents = json.loads(json_text)
            else:
                batch_contents = json.loads(response_text)
            
            # Validate response
            if not isinstance(batch_contents, list):
                raise ValueError("Expected JSON array response")
            
            if len(batch_contents) != len(verification_results):
                logger.warning(f"Expected {len(verification_results)} contents, got {len(batch_contents)}")
                # Pad or truncate as needed
                while len(batch_contents) < len(verification_results):
                    batch_contents.append({
                        "heading": "Content generation incomplete",
                        "body": "Unable to generate full content due to batch processing error",
                        "summary": "Content generation failed"
                    })
                batch_contents = batch_contents[:len(verification_results)]
            
            # Validate and truncate each content
            for i, content_data in enumerate(batch_contents):
                verification_result = verification_results[i]
                confidence = verification_result.get('confidence', 'medium')
                confidence_percentage = self._convert_confidence_to_percentage(confidence)
                
                content_data['heading'] = content_data.get('heading', '')[:config.MAX_HEADING_LENGTH]
                content_data['body'] = content_data.get('body', '')[:config.MAX_BODY_LENGTH]
                content_data['confidence_percentage'] = confidence_percentage
            
            return {
                'success': True,
                'batch_contents': batch_contents,
                'batch_size': len(verification_results)
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error in batch content generation: {e}")
            logger.error(f"Response that failed to parse: {response_text[:1000] if 'response_text' in locals() else 'No response text'}")
            
            # Generate fallback content for each claim
            fallback_contents = []
            for verification_result in verification_results:
                claim_text = verification_result.get('claim_text', 'Unknown claim')
                verdict = verification_result.get('verdict', 'uncertain')
                reasoning = verification_result.get('reasoning', 'No reasoning provided')
                message = verification_result.get('message', 'No message provided')
                confidence = verification_result.get('confidence', 'medium')
                
                fallback_contents.append({
                    'heading': self._generate_fallback_heading(verdict, claim_text),
                    'body': self._generate_fallback_body(reasoning, message),
                    'summary': message,
                    'confidence_percentage': self._convert_confidence_to_percentage(confidence)
                })
            
            return {
                'success': False,
                'error': f"JSON parsing error: {str(e)}",
                'batch_contents': fallback_contents,
                'batch_size': len(verification_results)
            }
        except Exception as e:
            logger.error(f"Batch content generation failed: {e}")
            # Generate fallback content for each claim
            fallback_contents = []
            for verification_result in verification_results:
                claim_text = verification_result.get('claim_text', 'Unknown claim')
                verdict = verification_result.get('verdict', 'uncertain')
                reasoning = verification_result.get('reasoning', 'No reasoning provided')
                message = verification_result.get('message', 'No message provided')
                confidence = verification_result.get('confidence', 'medium')
                
                fallback_contents.append({
                    'heading': self._generate_fallback_heading(verdict, claim_text),
                    'body': self._generate_fallback_body(reasoning, message),
                    'summary': message,
                    'confidence_percentage': self._convert_confidence_to_percentage(confidence)
                })
            
            return {
                'success': False,
                'error': str(e),
                'batch_contents': fallback_contents,
                'batch_size': len(verification_results)
            }
    
    def _convert_confidence_to_percentage(self, confidence: Any) -> float:
        """Convert confidence to percentage"""
        if isinstance(confidence, (int, float)):
            if confidence <= 1.0:
                return confidence * 100
            else:
                return min(confidence, 100.0)
        
        if isinstance(confidence, str):
            confidence_map = {
                'high': 85.0,
                'medium': 60.0,
                'low': 35.0,
                'very_high': 95.0,
                'very_low': 20.0
            }
            return confidence_map.get(confidence.lower(), 50.0)
        
        return 50.0  # Default
    
    def _generate_fallback_heading(self, verdict: str, claim_text: str) -> str:
        """Generate fallback heading if AI generation fails"""
        verdict_map = {
            'true': 'VERIFIED: ',
            'false': 'FALSE: ',
            'mixed': 'PARTIALLY TRUE: ',
            'uncertain': 'UNVERIFIED: '
        }
        
        prefix = verdict_map.get(verdict.lower(), 'FACT-CHECK: ')
        heading = prefix + claim_text[:config.MAX_HEADING_LENGTH - len(prefix)]
        
        return heading[:config.MAX_HEADING_LENGTH]
    
    def _generate_fallback_body(self, reasoning: str, message: str) -> str:
        """Generate fallback body if AI generation fails"""
        body = f"Our fact-checking analysis shows: {message}\n\n"
        body += f"Detailed reasoning: {reasoning}"
        
        return body[:config.MAX_BODY_LENGTH]


class SourceAnalyzerTool:
    """Tool for analyzing and categorizing sources"""
    
    def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze and categorize sources from verification results"""
        verification_result = context.get('verification_result', {})
        sources = verification_result.get('sources', {})
        
        links = sources.get('links', [])
        titles = sources.get('titles', [])
        
        logger.info(f"Processing sources: {len(links)} links, {len(titles)} titles")
        logger.info(f"Links: {links}")
        logger.info(f"Titles: {titles}")
        
        all_sources = []
        
        for i, (link, title) in enumerate(zip(links, titles)):
            if not link or not title:
                continue
                
            source_domain = self._extract_domain(link)
            
            source_entry = {
                "title": title,
                "url": link,
                "domain": source_domain
            }
            
            all_sources.append(source_entry)
        
        logger.info(f"Created {len(all_sources)} source entries")
        # Attempt to locate the original/original post link (OG post) to include as a misinformation source
        orig_url = None
        # Common places where original post link might be stored
        if verification_result.get('source'):
            orig_url = verification_result.get('source')
        elif isinstance(verification_result.get('original_content'), dict):
            oc = verification_result.get('original_content')
            orig_url = oc.get('Post_link') or oc.get('post_link') or oc.get('url') or oc.get('link')

        misinformation_sources = []
        if orig_url:
            # Only add original post link if it's not already included in confirmation sources
            if not any(s['url'] == orig_url for s in all_sources):
                misinformation_sources.append({
                    "title": verification_result.get('claim_text', 'Original post'),
                    "url": orig_url,
                    "domain": self._extract_domain(orig_url)
                })

        total_count = len(all_sources) + len(misinformation_sources)

        return {
            "confirmation_sources": all_sources,  # All sources treated as confirmation sources
            "misinformation_sources": misinformation_sources,
            "total_sources": total_count  # Use actual count of processed sources + original post
        }
    
    def process_batch(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze and categorize sources from multiple verification results"""
        verification_results = context.get('verification_results', [])
        
        if not verification_results:
            return {
                'success': False,
                'error': 'No verification results provided for batch source analysis',
                'batch_sources': []
            }
        
        logger.info(f"Processing batch sources for {len(verification_results)} verification results")
        
        batch_sources = []
        
        for idx, verification_result in enumerate(verification_results):
            sources = verification_result.get('sources', {})
            links = sources.get('links', [])
            titles = sources.get('titles', [])
            
            logger.info(f"Verification result {idx}: {len(links)} links, {len(titles)} titles")
            if links:
                logger.info(f"  Links: {links[:2]}")  # Log first 2 links
            if titles:
                logger.info(f"  Titles: {titles[:2]}")  # Log first 2 titles
            
            all_sources = []
            
            for i, (link, title) in enumerate(zip(links, titles)):
                if not link or not title:
                    logger.warning(f"Skipping empty link or title at index {i}: link='{link}', title='{title}'")
                    continue
                    
                source_domain = self._extract_domain(link)
                
                source_entry = {
                    "title": title,
                    "url": link,
                    "domain": source_domain
                }
                
                all_sources.append(source_entry)
            
            logger.info(f"Verification result {idx}: Created {len(all_sources)} source entries")
            # Try to capture the original post link for this verification result
            orig_url = None
            if verification_result.get('source'):
                orig_url = verification_result.get('source')
            elif isinstance(verification_result.get('original_content'), dict):
                oc = verification_result.get('original_content')
                orig_url = oc.get('Post_link') or oc.get('post_link') or oc.get('url') or oc.get('link')

            misinformation_sources = []
            if orig_url:
                if not any(s['url'] == orig_url for s in all_sources):
                    misinformation_sources.append({
                        "title": verification_result.get('claim_text', 'Original post'),
                        "url": orig_url,
                        "domain": self._extract_domain(orig_url)
                    })

            source_analysis = {
                "confirmation_sources": all_sources,  # All sources treated as confirmation sources
                "misinformation_sources": misinformation_sources,
                "total_sources": len(all_sources) + len(misinformation_sources)  # Use actual count of processed sources
            }
            
            batch_sources.append(source_analysis)
        
        logger.info(f"Batch source analysis completed: {sum(s['total_sources'] for s in batch_sources)} total sources processed")
        
        return {
            'success': True,
            'batch_sources': batch_sources,
            'batch_size': len(verification_results)
        }
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            parsed = urlparse(url)
            return parsed.netloc.lower()
        except:
            return "unknown"
class ExplanationAgent:
    """Main agent that orchestrates explanation generation using Google Agents SDK"""
    
    def __init__(self):
        """Initialize the Explanation Agent with Google Agents orchestration"""
        if not config.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is required for Explanation Agent")
        
        # Initialize Google Agents Orchestrator
        self.orchestrator = GoogleAgentsOrchestrator(config.GEMINI_API_KEY)
        
        # Create specialized agents with tools
        self._setup_agents()
        
        # Ensure output directory exists
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        
        logger.info("Explanation Agent initialized with Google Agents SDK")
    
    def _setup_agents(self):
        """Setup specialized Google Agents for explanation workflow"""
        
        # Content Generator Agent with tool
        content_tool = ContentGeneratorTool(self.orchestrator.model)
        content_agent = self.orchestrator.create_agent(
            name="content_generator",
            role="Content Generation Specialist",
            goal="Create clear, engaging explanations for fact-check results",
            tools=[content_tool]
        )
        
        # Store as direct attribute for easier access in batch processing
        self.content_generator = content_agent
        
        # Source Analyzer Agent with tool  
        source_tool = SourceAnalyzerTool()
        source_agent = self.orchestrator.create_agent(
            name="source_analyzer", 
            role="Source Analysis Specialist",
            goal="Analyze and categorize sources for credibility assessment",
            tools=[source_tool]
        )
        
        # Store as direct attribute for easier access in batch processing
        self.source_analyzer = source_agent
        
        # Post Formatter Agent
        self.orchestrator.create_agent(
            name="post_formatter",
            role="Post Formatting Specialist", 
            goal="Structure and format the final debunk post for publication"
        )
        
        logger.info("Google Agents setup completed for explanation workflow")
    
    def create_debunk_post(self, verification_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a structured debunk post using Google Agents workflow
        
        Args:
            verification_result: Output from claim verifier containing verdict, reasoning, sources, etc.
            
        Returns:
            Structured debunk post in JSON format
        """
        try:
            logger.info(f"Creating debunk post for claim: {verification_result.get('claim_text', 'Unknown')[:50]}...")
            
            # Define workflow tasks for Google Agents
            tasks = [
                {
                    'agent': 'content_generator',
                    'description': 'Generate clear, engaging content for the debunk post including heading, body, and summary',
                    'context': {'verification_result': verification_result}
                },
                {
                    'agent': 'source_analyzer', 
                    'description': 'Analyze and categorize the sources into confirmation vs misinformation sources',
                    'context': {'verification_result': verification_result}
                },
                {
                    'agent': 'post_formatter',
                    'description': 'Format the final structured debunk post with all components integrated'
                }
            ]
            
            # Execute Google Agents workflow
            workflow_result = self.orchestrator.execute_workflow(tasks)
            
            # Process workflow results
            debunk_post = self._process_workflow_results(workflow_result, verification_result)
            
            # Save the post
            post_file = self._save_post(debunk_post)
            debunk_post["saved_to"] = post_file
            
            logger.info(f"Debunk post created successfully with Google Agents: {post_file}")
            return debunk_post
            
        except Exception as e:
            logger.error(f"Error creating debunk post with Google Agents: {e}")
            return self._create_error_post(verification_result, str(e))
    
    def _process_workflow_results(self, workflow_result: Dict[str, Any], 
                                verification_result: Dict[str, Any]) -> Dict[str, Any]:
        """Process Google Agents workflow results into final debunk post"""
        
        # Extract results from each agent
        content_data = {}
        source_data = {}
        confidence_percentage = 50.0
        
        for result in workflow_result.get('results', []):
            agent_role = result.get('agent_role', '')
            
            if 'Content Generation' in agent_role and result.get('tool_used'):
                try:
                    tool_result = result['result']
                    if tool_result.get('success'):
                        content_data = tool_result.get('content', {})
                        confidence_percentage = tool_result.get('confidence_percentage', 50.0)
                    else:
                        content_data = tool_result.get('fallback_content', {})
                        confidence_percentage = tool_result.get('confidence_percentage', 50.0)
                except Exception as e:
                    logger.error(f"Error processing content generation result: {e}")
            
            elif 'Source Analysis' in agent_role and result.get('tool_used'):
                try:
                    source_data = result['result']
                except Exception as e:
                    logger.error(f"Error processing source analysis result: {e}")
        
        # Create structured debunk post
        debunk_post = {
            "post_id": self._generate_post_id(),
            "timestamp": datetime.now().isoformat(),
            "verification_date": verification_result.get('verification_date', datetime.now().isoformat()),
            "claim": {
                "text": verification_result.get('claim_text', 'No claim provided'),
                "verdict": verification_result.get('verdict', 'uncertain'),
                "verified": verification_result.get('verified', False)
            },
            "post_content": {
                "heading": content_data.get('heading', self._generate_fallback_heading(
                    verification_result.get('verdict', 'uncertain'),
                    verification_result.get('claim_text', 'Unknown claim')
                )),
                "body": content_data.get('body', self._generate_fallback_body(
                    verification_result.get('reasoning', 'No reasoning'),
                    verification_result.get('message', 'No message')
                )),
                "summary": content_data.get('summary', verification_result.get('message', 'No summary'))
            },
            "sources": {
                "misinformation_sources": source_data.get('misinformation_sources', []),
                "confirmation_sources": source_data.get('confirmation_sources', []),
                "total_sources": source_data.get('total_sources', 0)
            },
            "confidence_percentage": confidence_percentage,
            "metadata": {
                "agent_version": "2.0",
                "processing_method": "google_agents_sdk",
                "workflow_id": workflow_result.get('workflow_type', 'unknown'),
                "original_verification": verification_result
            }
        }
        
        return debunk_post
    
    def _generate_post_id(self) -> str:
        """Generate unique post ID with microseconds and UUID suffix for uniqueness"""
        import uuid
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Add microseconds and short UUID to ensure uniqueness in batch processing
        microseconds = datetime.now().microsecond
        unique_suffix = str(uuid.uuid4())[:8]  # Short UUID suffix
        return f"aegis_post_{timestamp}_{microseconds}_{unique_suffix}"
    
    def _generate_fallback_heading(self, verdict: str, claim_text: str) -> str:
        """Generate fallback heading"""
        verdict_map = {
            'true': 'VERIFIED: ',
            'false': 'FALSE: ',
            'mixed': 'PARTIALLY TRUE: ',
            'uncertain': 'UNVERIFIED: '
        }
        
        prefix = verdict_map.get(verdict.lower(), 'FACT-CHECK: ')
        heading = prefix + claim_text[:config.MAX_HEADING_LENGTH - len(prefix)]
        
        return heading[:config.MAX_HEADING_LENGTH]
    
    def _generate_fallback_body(self, reasoning: str, message: str) -> str:
        """Generate fallback body"""
        body = f"Our fact-checking analysis shows: {message}\n\n"
        body += f"Detailed reasoning: {reasoning}"
        
        return body[:config.MAX_BODY_LENGTH]
    
    def _save_post(self, post_data: Dict[str, Any]) -> str:
        """Save the debunk post to file"""
        post_id = post_data.get('post_id', 'unknown')
        filename = f"{post_id}.json"
        filepath = os.path.join(config.OUTPUT_DIR, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(post_data, f, indent=2, ensure_ascii=False)
        
        return filepath
    
    def _create_error_post(self, verification_result: Dict[str, Any], error_message: str) -> Dict[str, Any]:
        """Create error post when processing fails"""
        return {
            "post_id": self._generate_post_id(),
            "timestamp": datetime.now().isoformat(),
            "error": True,
            "error_message": error_message,
            "claim": {
                "text": verification_result.get('claim_text', 'Unknown'),
                "verdict": "error",
                "verified": False
            },
            "post_content": {
                "heading": "Error Processing Fact-Check",
                "body": f"An error occurred while processing this fact-check: {error_message}",
                "summary": "Processing error occurred"
            },
            "sources": {
                "misinformation_sources": [],
                "confirmation_sources": [],
                "total_sources": 0
            },
            "confidence_percentage": 0.0,
            "metadata": {
                "agent_version": "2.0",
                "processing_method": "google_agents_sdk_error",
                "original_verification": verification_result
            }
        }
    
    def batch_create_posts(self, verification_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create multiple debunk posts using efficient batch processing with max 10 per batch"""
        try:
            if not verification_results:
                return {
                    'success': True,
                    'message': 'No verification results provided',
                    'debunk_posts': []
                }
            
            logger.info(f"Creating {len(verification_results)} debunk posts with batch Google Agents processing...")
            
            all_posts = []
            BATCH_SIZE = 10
            
            # Process in batches of max 10
            for batch_start in range(0, len(verification_results), BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, len(verification_results))
                batch_results = verification_results[batch_start:batch_end]
                
                logger.info(f"Processing batch {batch_start//BATCH_SIZE + 1}: posts {batch_start+1}-{batch_end}")
                
                try:
                    # Generate content for entire batch in single Gemini call
                    content_context = {'verification_results': batch_results}
                    content_result = self.content_generator.process_batch(content_context)
                    
                    # Generate source analysis for entire batch
                    source_context = {'verification_results': batch_results}
                    source_result = self.source_analyzer.process_batch(source_context)
                    
                    if not content_result.get('success', False):
                        logger.error(f"Batch content generation failed: {content_result.get('error', 'Unknown error')}")
                        # Use fallback content instead of failing
                        batch_contents = content_result.get('batch_contents', [])
                        if not batch_contents:
                            # Generate fallback content if none provided
                            for verification_result in batch_results:
                                batch_contents.append({
                                    'heading': self._generate_fallback_heading(
                                        verification_result.get('verdict', 'uncertain'),
                                        verification_result.get('claim_text', 'Unknown claim')
                                    ),
                                    'body': self._generate_fallback_body(
                                        verification_result.get('reasoning', 'No reasoning'),
                                        verification_result.get('message', 'No message')
                                    ),
                                    'summary': verification_result.get('message', 'No summary'),
                                    'confidence_percentage': 30.0
                                })
                    else:
                        batch_contents = content_result.get('batch_contents', [])
                    
                    if not source_result.get('success', False):
                        logger.error(f"Batch source analysis failed: {source_result.get('error', 'Unknown error')}")
                        # Use fallback sources
                        batch_sources = source_result.get('batch_sources', [])
                        if not batch_sources:
                            for verification_result in batch_results:
                                batch_sources.append({
                                    'misinformation_sources': [],
                                    'confirmation_sources': [],
                                    'total_sources': 0
                                })
                    else:
                        batch_sources = source_result.get('batch_sources', [])
                    
                    # Create individual posts from batch results
                    for i, (verification_result, content_data, source_data) in enumerate(zip(batch_results, batch_contents, batch_sources)):
                        
                        confidence_percentage = content_data.get('confidence_percentage', 50.0)
                        
                        debunk_post = {
                            "post_id": self._generate_post_id(),
                            "verification_date": verification_result.get('verification_date', datetime.now().isoformat()),
                            "claim": {
                                "text": verification_result.get('claim_text', 'No claim provided'),
                                "verdict": verification_result.get('verdict', 'uncertain'),
                                "verified": verification_result.get('verified', False)
                            },
                            "post_content": {
                                "heading": content_data.get('heading', self._generate_fallback_heading(
                                    verification_result.get('verdict', 'uncertain'),
                                    verification_result.get('claim_text', 'Unknown claim')
                                )),
                                "body": content_data.get('body', self._generate_fallback_body(
                                    verification_result.get('reasoning', 'No reasoning'),
                                    verification_result.get('message', 'No message')
                                )),
                                "summary": content_data.get('summary', verification_result.get('message', 'No summary'))
                            },
                            "sources": {
                                "misinformation_sources": source_data.get('misinformation_sources', []),
                                "confirmation_sources": source_data.get('confirmation_sources', []),
                                "total_sources": source_data.get('total_sources', 0)
                            },
                            "confidence_percentage": confidence_percentage,
                            "metadata": {
                                "agent_version": "2.0",
                                "processing_method": "google_agents_batch_sdk",
                                "batch_position": i + 1,
                                "batch_size": len(batch_results),
                                "total_batch_number": batch_start//BATCH_SIZE + 1,
                                "original_verification": verification_result
                            }
                        }
                        
                        # Save the post
                        post_file = self._save_post(debunk_post)
                        debunk_post["saved_to"] = post_file
                        
                        all_posts.append(debunk_post)
                        
                except Exception as e:
                    logger.error(f"Batch processing failed for batch {batch_start//BATCH_SIZE + 1}: {e}")
                    # Create error posts for the failed batch
                    for verification_result in batch_results:
                        error_post = {
                            "post_id": self._generate_post_id(),
                            "verification_date": verification_result.get('verification_date', datetime.now().isoformat()),
                            "claim": {
                                "text": verification_result.get('claim_text', 'No claim provided'),
                                "verdict": verification_result.get('verdict', 'uncertain'),
                                "verified": verification_result.get('verified', False)
                            },
                            "post_content": {
                                "heading": self._generate_fallback_heading(
                                    verification_result.get('verdict', 'uncertain'),
                                    verification_result.get('claim_text', 'Unknown claim')
                                ),
                                "body": self._generate_fallback_body(
                                    verification_result.get('reasoning', 'No reasoning'),
                                    verification_result.get('message', 'No message')
                                ),
                                "summary": verification_result.get('message', 'Batch processing failed')
                            },
                            "sources": {
                                "misinformation_sources": [],
                                "confirmation_sources": [],
                                "total_sources": 0
                            },
                            "confidence_percentage": 30.0,  # Low confidence for error posts
                            "metadata": {
                                "agent_version": "2.0",
                                "processing_method": "fallback_batch_error",
                                "error": str(e),
                                "original_verification": verification_result
                            }
                        }
                        
                        # Save the error post
                        post_file = self._save_post(error_post)
                        error_post["saved_to"] = post_file
                        
                        all_posts.append(error_post)
            
            logger.info(f"Batch processing completed: {len(all_posts)} total posts created")
            
            return {
                'success': True,
                'message': f'Successfully created {len(all_posts)} debunk posts using batch processing',
                'debunk_posts': all_posts,
                'batch_statistics': {
                    'total_posts': len(all_posts),
                    'batch_size_used': BATCH_SIZE,
                    'total_batches': (len(verification_results) + BATCH_SIZE - 1) // BATCH_SIZE,
                    'successful_posts': len([p for p in all_posts if 'error' not in p.get('metadata', {})]),
                    'error_posts': len([p for p in all_posts if 'error' in p.get('metadata', {})])
                }
            }
            
        except Exception as e:
            logger.error(f"Batch create posts failed: {e}")
            return {
                'success': False,
                'message': f'Batch processing failed: {str(e)}',
                'error': str(e),
                'debunk_posts': []
            }