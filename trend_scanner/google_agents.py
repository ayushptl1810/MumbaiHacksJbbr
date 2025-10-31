"""Google AI integration for trend scanner with orchestration capabilities"""

import os
import logging
from typing import Optional, Dict, Any, List
import google.generativeai as genai
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class GoogleAgent:
    """Individual Google AI agent with specific role and capabilities"""
    
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
                    if hasattr(tool, '_run') and 'scan' in task_description.lower():
                        # This is likely a Reddit scanning task
                        try:
                            # Extract subreddit from task description (e.g., "Scan r/DebunkThis for...")
                            import re
                            subreddit_match = re.search(r'r/(\w+)', task_description)
                            target_subreddit = subreddit_match.group(1) if subreddit_match else 'worldnews'
                            
                            logger.info(f"Agent {self.role} executing tool scan for r/{target_subreddit}")
                            tool_result = tool._run(target_subreddit)
                            
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
                for key, value in context.items():
                    try:
                        # Only include simple serializable data
                        if isinstance(value, (str, int, float, bool, list, dict)):
                            if isinstance(value, dict):
                                # For dict values, create a safe summary
                                safe_context[key] = f"Dict with {len(value)} keys"
                            elif isinstance(value, list):
                                # For list values, create a safe summary
                                safe_context[key] = f"List with {len(value)} items"
                            else:
                                safe_context[key] = value
                        else:
                            # For complex objects, just note their type
                            safe_context[key] = f"<{type(value).__name__} object>"
                    except:
                        safe_context[key] = "<unable to serialize>"
            
            # Create context-aware prompt with special handling for trending posts
            if safe_context:
                # Special handling for trending posts data
                if 'last_result' in safe_context and isinstance(context.get('last_result'), dict):
                    last_result = context['last_result']
                    if (last_result.get('tool_used') and 
                        'result' in last_result and 
                        isinstance(last_result['result'], str)):
                        try:
                            # Try to parse the tool result as JSON (Reddit scan results)
                            import json
                            tool_data = json.loads(last_result['result'])
                            if 'trending_posts' in tool_data:
                                trending_posts = tool_data['trending_posts']
                                posts_summary = f"Found {len(trending_posts)} trending posts from Reddit scan:\n"
                                for i, post in enumerate(trending_posts[:5], 1):  # Show first 5 posts
                                    posts_summary += f"{i}. '{post.get('title', 'No title')}' (Risk: {post.get('risk_level', 'Unknown')}, Score: {post.get('score', 0)})\n"
                                if len(trending_posts) > 5:
                                    posts_summary += f"... and {len(trending_posts) - 5} more posts\n"
                                context_text = f"Previous Reddit scan results:\n{posts_summary}\nFull data available for analysis."
                            else:
                                context_summary = "\n".join([f"- {k}: {v}" for k, v in safe_context.items()])
                                context_text = f"Context information:\n{context_summary}"
                        except (json.JSONDecodeError, KeyError):
                            context_summary = "\n".join([f"- {k}: {v}" for k, v in safe_context.items()])
                            context_text = f"Context information:\n{context_summary}"
                    else:
                        context_summary = "\n".join([f"- {k}: {v}" for k, v in safe_context.items()])
                        context_text = f"Context information:\n{context_summary}"
                else:
                    context_summary = "\n".join([f"- {k}: {v}" for k, v in safe_context.items()])
                    context_text = f"Context information:\n{context_summary}"
            else:
                context_text = "No context provided"
            
            prompt = f"""
            You are a {self.role} with the goal: {self.goal}
            
            Current task: {task_description}
            
            {context_text}
            
            {"IMPORTANT: If you are analyzing trending posts from previous results, make sure to provide specific analysis for each post found. Do not just acknowledge the requirement - actually perform the analysis." if "risk_assessor" in self.role.lower() or "assess" in task_description.lower() else ""}
            
            Execute this task thoroughly and provide detailed results.
            """
            
            # Execute with Gemini
            response = self.model.generate_content(prompt)
            response_text = getattr(response, 'text', str(response))
            
            # Special handling for Content Risk Assessor - provide actual trending posts data
            if ("risk_assessor" in self.role.lower() or "assess" in task_description.lower()) and context and 'last_result' in context:
                try:
                    last_result = context['last_result']
                    if (last_result.get('tool_used') and 'result' in last_result):
                        import json
                        tool_data = json.loads(last_result['result'])
                        if 'trending_posts' in tool_data and tool_data['trending_posts']:
                            trending_posts = tool_data['trending_posts']
                            
                            # Create detailed analysis prompt with actual data
                            detailed_prompt = f"""
                            You are a Content Risk Assessor. Here are the trending posts found by the Reddit scanner:
                            
                            TRENDING POSTS DATA:
                            {json.dumps(trending_posts, indent=2)}
                            
                            Task: {task_description}
                            
                            Analyze each post above and provide:
                            1. Risk level (HIGH/MEDIUM/LOW) with detailed reasoning
                            2. Priority score (1-10) for fact-checking
                            3. Key claims to verify
                            4. Source credibility assessment
                            5. Viral potential analysis
                            6. Recommended actions
                            
                            Provide a structured analysis for each trending post found.
                            """
                            
                            # Re-execute with detailed trending posts data
                            response = self.model.generate_content(detailed_prompt)
                            response_text = getattr(response, 'text', str(response))
                            logger.info(f"Content Risk Assessor provided with {len(trending_posts)} trending posts for detailed analysis")
                        else:
                            logger.warning("No trending posts found in previous results for Risk Assessor")
                except Exception as e:
                    logger.warning(f"Failed to extract trending posts for Risk Assessor: {e}")
            
            result = {
                'agent_role': self.role,
                'task': task_description,
                'result': response_text,
                'timestamp': datetime.now().isoformat(),
                'context_summary': safe_context,
                'tool_used': False
            }
            
            # Store in history
            self.history.append(result)
            
            return result
            
        except Exception as e:
            error_result = {
                'agent_role': self.role,
                'task': task_description,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            self.history.append(error_result)
            return error_result


class GoogleOrchestrator:
    """Orchestrates multiple Google AI agents in workflows"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("Google API key is required")
        
        # Configure Google AI
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Agent registry
        self.agents = {}
        self.workflow_history = []
    
    def create_agent(self, name: str, role: str, goal: str, tools: List[Any] = None) -> GoogleAgent:
        """Create and register a new agent"""
        agent = GoogleAgent(role, goal, self.model, tools)
        self.agents[name] = agent
        logger.info(f"Created Google Agent: {name} - {role}")
        return agent
    
    def sequential_workflow(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute tasks sequentially, passing results between agents"""
        workflow_results = []
        context = {}
        
        logger.info(f"Starting sequential workflow with {len(tasks)} tasks")
        
        for i, task in enumerate(tasks):
            agent_name = task['agent']
            task_description = task['description']
            
            if agent_name not in self.agents:
                raise ValueError(f"Agent '{agent_name}' not found")
            
            # Update context with previous results
            if workflow_results:
                context['previous_results'] = workflow_results
                context['last_result'] = workflow_results[-1]
                
                # Log context details for debugging
                logger.debug(f"Passing context to agent '{agent_name}': {len(workflow_results)} previous results")
                if workflow_results[-1].get('tool_used'):
                    logger.debug(f"Last result contains tool execution data")
            
            # Execute task
            logger.info(f"Executing task {i+1}/{len(tasks)} with agent '{agent_name}'")
            try:
                result = self.agents[agent_name].execute_task(task_description, context)
                if not result.get('has_error', False):
                    logger.info(f"Task {i+1} completed successfully by '{agent_name}'")
                else:
                    logger.warning(f"Task {i+1} completed with errors by '{agent_name}': {result.get('error_message', 'Unknown error')}")
                workflow_results.append(result)
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
            'workflow_type': 'sequential',
            'total_tasks': len(tasks),
            'results': workflow_results,
            'summary': self._create_workflow_summary(workflow_results),
            'timestamp': datetime.now().isoformat()
        }
        
        self.workflow_history.append(final_result)
        return final_result
    
    def parallel_workflow(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute tasks in parallel (simulated - Gemini doesn't support true async)"""
        workflow_results = []
        
        logger.info(f"Starting parallel workflow with {len(tasks)} tasks")
        
        for i, task in enumerate(tasks):
            agent_name = task['agent']
            task_description = task['description']
            
            if agent_name not in self.agents:
                raise ValueError(f"Agent '{agent_name}' not found")
            
            # Execute task (in parallel simulation)
            logger.info(f"Executing parallel task {i+1}/{len(tasks)} with agent '{agent_name}'")
            result = self.agents[agent_name].execute_task(task_description)
            workflow_results.append(result)
        
        # Create final workflow summary
        final_result = {
            'workflow_type': 'parallel',
            'total_tasks': len(tasks),
            'results': workflow_results,
            'summary': self._create_workflow_summary(workflow_results),
            'timestamp': datetime.now().isoformat()
        }
        
        self.workflow_history.append(final_result)
        return final_result
    
    def _create_workflow_summary(self, results: List[Dict[str, Any]]) -> str:
        """Create a summary of workflow execution"""
        try:
            # Create a safe summary without circular references
            safe_results = []
            for result in results:
                safe_result = {
                    'agent_role': result.get('agent_role', 'Unknown'),
                    'task': result.get('task', 'No task description')[:100],  # Limit length
                    'result_snippet': str(result.get('result', 'No result'))[:200],  # Limit length
                    'timestamp': result.get('timestamp', 'Unknown'),
                    'has_error': 'error' in result
                }
                safe_results.append(safe_result)
            
            summary_prompt = f"""
            Analyze these workflow results and provide a concise summary:
            
            Total agents executed: {len(safe_results)}
            Results overview: {safe_results}
            
            Provide:
            1. Key findings from each agent
            2. Overall workflow success
            3. Important insights or patterns
            4. Any issues or errors encountered
            
            Keep the summary concise and focused.
            """
            
            response = self.model.generate_content(summary_prompt)
            return response.text
            
        except Exception as e:
            return f"Summary generation failed: {str(e)}. Workflow completed with {len(results)} tasks."


class GoogleAIManager(GoogleOrchestrator):
    """Extended GoogleAIManager with orchestration capabilities"""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key)
        # Initialize Gemini 2.5 Flash model for backward compatibility
        self.model = genai.GenerativeModel('gemini-2.5-flash')
    
    def create_trend_scanner_workflow(self, reddit_tool, subreddits: List[str]) -> Dict[str, Any]:
        """Create a trend scanner workflow using Google orchestration instead of CrewAI"""
        
        # Create specialized agents
        scanner_agent = self.create_agent(
            name="reddit_scanner",
            role="Reddit Trend Scout",
            goal="Identify rapidly trending posts that could contain misinformation",
            tools=[reddit_tool]
        )
        
        assessor_agent = self.create_agent(
            name="risk_assessor", 
            role="Content Risk Assessor",
            goal="Evaluate and prioritize trending posts by misinformation risk",
            tools=[]
        )
        
        # Define workflow tasks
        tasks = []
        
        # Step 1: Scan each subreddit
        for subreddit in subreddits:
            tasks.append({
                'agent': 'reddit_scanner',
                'description': f"""
                Scan r/{subreddit} for trending posts with potential misinformation.
                Look for:
                - High velocity posts (rapid upvote growth)
                - Suspicious claims or unsourced assertions
                - Emotional manipulation techniques
                - Links to questionable sources
                
                Analyze both Reddit content and any linked external content.
                """
            })
        
        # Step 2: Assess and prioritize all findings
        tasks.append({
            'agent': 'risk_assessor',
            'description': """
            You are a Content Risk Assessor. Analyze ALL trending posts from the previous Reddit scan.
            
            IMPORTANT: Use the trending posts data from the previous agent's results.
            
            For each trending post found, provide:
            1. Detailed risk analysis (HIGH/MEDIUM/LOW) with reasoning
            2. Priority ranking for manual fact-checking (1-10 scale)
            3. Key claims that need verification
            4. Source credibility assessment
            5. Viral spread potential (based on velocity and content type)
            6. Recommended actions (flag, investigate, monitor, or clear)
            
            Format your response as a structured analysis of each post with clear risk levels and actionable recommendations.
            
            Focus on posts that combine high velocity with questionable content, misinformation patterns, or unverified claims.
            """
        })
        
        # Execute workflow
        return self.sequential_workflow(tasks)


# Keep the old class name for backward compatibility but redirect to new implementation
class GoogleAgentsManager(GoogleAIManager):
    """Backward compatibility alias for GoogleAIManager"""
    pass


class TrendScannerOrchestrator:
    """Main orchestrator that replaces CrewAI crew functionality using only Google Agents"""
    
    def __init__(self, reddit_config: Dict[str, str], gemini_api_key: Optional[str] = None):
        gemini_key = gemini_api_key or os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
        if not gemini_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY must be provided")
        
        # Initialize Google Agents Manager with orchestration capabilities
        try:
            self.google_agents = GoogleAgentsManager(api_key=gemini_key)
            logger.info("Google Agents orchestration initialized successfully")
        except Exception as e:
            logger.error(f"Google Agents orchestration failed: {e}")
            raise

        # Initialize Reddit client
        import praw
        self.reddit = praw.Reddit(
            client_id=reddit_config['client_id'],
            client_secret=reddit_config['client_secret'],
            user_agent=reddit_config['user_agent']
        )
        
        # Test Reddit authentication
        try:
            # Try to access the Reddit user to test authentication
            user = self.reddit.user.me()
            logger.info(f"Reddit authentication successful (read-only mode)")
        except Exception as e:
            logger.warning(f"Reddit authentication test: {e} (This is expected for app-only authentication)")
        
        # Test basic subreddit access
        try:
            test_subreddit = self.reddit.subreddit("test")
            logger.info(f"Reddit API connection verified - can access subreddits")
        except Exception as e:
            logger.error(f"Reddit API connection failed: {e}")

        # Simple LLM wrapper for backward compatibility
        class SimpleLLMWrapper:
            def __init__(self):
                import litellm
                self.completion = litellm.completion

            def invoke(self, prompt):
                response = self.completion(
                    model="gemini/gemini-2.5-flash",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1
                )
                class ResponseWrapper:
                    def __init__(self, content):
                        self.content = content
                return ResponseWrapper(response.choices[0].message.content)

        self.llm = SimpleLLMWrapper()

        # Reddit tool for Google agents
        from .tools import RedditScanTool
        self.reddit_tool = RedditScanTool(
            self.reddit, 
            self.llm, 
            velocity_threshold=25.0, 
            min_score_threshold=50,
            google_api_key=gemini_key
        )

        # Dynamic subreddit targeting - will be determined from task descriptions
        self.target_subreddits = []

        # Initialize Google agents for orchestration
        self._setup_google_agents()

    def _setup_google_agents(self):
        """Setup Google agents for orchestration instead of CrewAI agents"""
        # Create Reddit scanner agent
        self.reddit_scanner = self.google_agents.create_agent(
            name="reddit_scanner",
            role="Enhanced Reddit Trend Scout",
            goal="Identify rapidly trending posts across Reddit that could contain misinformation using Google AI credibility analysis",
            tools=[self.reddit_tool]
        )
        
        # Create risk assessor agent  
        self.risk_assessor = self.google_agents.create_agent(
            name="risk_assessor",
            role="Advanced Content Risk Assessor", 
            goal="Evaluate the risk level of trending posts using Google AI credibility analysis and content scoring",
            tools=[]
        )
        
        logger.info("Google agents created successfully for orchestration")
    
    def set_target_subreddits(self, subreddits: List[str]):
        """Set the target subreddits for scanning"""
        self.target_subreddits = subreddits
        logger.info(f"Target subreddits configured: {subreddits}")
    
    def add_target_subreddit(self, subreddit: str):
        """Add a single subreddit to target list"""
        if subreddit not in self.target_subreddits:
            self.target_subreddits.append(subreddit)
            logger.info(f"Added subreddit to targets: r/{subreddit}")

    def scan_trending_content(self) -> Dict[str, Any]:
        """Execute the trend scanning workflow using Google orchestration (replaces CrewAI crew.kickoff())"""
        logger.info("Starting Google-orchestrated trend scanning...")
        
        if not self.target_subreddits:
            raise ValueError("No target subreddits configured. Use set_target_subreddits() to configure subreddits to scan.")
        
        # Use Google orchestration instead of CrewAI
        workflow_result = self.google_agents.create_trend_scanner_workflow(
            reddit_tool=self.reddit_tool,
            subreddits=self.target_subreddits
        )
        
        # Process workflow results to match expected output format
        return self._process_workflow_results(workflow_result)
    
    def _process_workflow_results(self, workflow_result: Dict[str, Any]) -> Dict[str, Any]:
        """Process Google orchestration results to match expected output format"""
        try:
            all_trending_posts = []
            scan_summaries = []
            total_scraped = 0
            posts_with_scraped_content = 0
            
            # Extract results from Google workflow
            if 'results' in workflow_result:
                for result in workflow_result['results']:
                    # Check if this is the Reddit scanner result with actual tool execution
                    if (result.get('agent_role') == 'Reddit Trend Scout' and 
                        result.get('tool_used', False)):
                        
                        # Parse the actual tool result (JSON string from RedditScanTool)
                        try:
                            scan_data = json.loads(result['result'])
                            if 'trending_posts' in scan_data:
                                all_trending_posts.extend(scan_data['trending_posts'])
                                total_scraped += scan_data.get('scraped_count', 0)
                                posts_with_scraped_content += len([p for p in scan_data['trending_posts'] if p.get('scraped_content')])
                            
                            scan_summaries.append(scan_data.get('scan_summary', 'Scan completed'))
                            logger.info(f"Successfully processed Reddit scan data: {len(scan_data.get('trending_posts', []))} posts found")
                            
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to parse Reddit scan results: {e}")
                            # Try to extract any useful info from the raw result
                            scan_summaries.append(f"Tool execution completed but parsing failed: {str(result['result'])[:100]}")
                    
                    # Handle text-based results (fallback AI responses)
                    elif result.get('agent_role') in ['Reddit Trend Scout', 'Content Risk Assessor']:
                        scan_summaries.append(f"{result['agent_role']}: {result.get('result', 'No result')[:200]}")
            
            # If no posts found from tool execution, try direct tool execution as fallback
            if not all_trending_posts and hasattr(self, 'reddit_tool'):
                try:
                    logger.info("No posts found from workflow - attempting direct tool execution")
                    # Use the first target subreddit if available, otherwise raise error
                    fallback_subreddit = self.target_subreddits[0] if self.target_subreddits else None
                    if not fallback_subreddit:
                        logger.warning("No target subreddits configured for fallback execution")
                        return {
                            'success': False,
                            'message': 'No scan targets available',
                            'trending_posts': [],
                            'total_scraped': 0,
                            'posts_with_scraped_content': 0,
                            'scan_summaries': []
                        }
                    
                    direct_result = self.reddit_tool._run(fallback_subreddit)
                    scan_data = json.loads(direct_result)
                    if 'trending_posts' in scan_data:
                        all_trending_posts.extend(scan_data['trending_posts'])
                        total_scraped += scan_data.get('scraped_count', 0)
                        posts_with_scraped_content += len([p for p in scan_data['trending_posts'] if p.get('scraped_content')])
                        scan_summaries.append("Direct tool execution successful")
                        logger.info(f"Direct tool execution found {len(scan_data['trending_posts'])} posts")
                except Exception as e:
                    logger.error(f"Direct tool execution failed: {e}")
            
            # Calculate risk distribution
            risk_distribution = {
                'HIGH': len([p for p in all_trending_posts if p.get('risk_level') == 'HIGH']),
                'MEDIUM': len([p for p in all_trending_posts if p.get('risk_level') == 'MEDIUM']),
                'LOW': len([p for p in all_trending_posts if p.get('risk_level') == 'LOW'])
            }
            
            # Get assessment from workflow
            risk_assessment = workflow_result.get('summary', 'Google orchestration completed successfully')
            
            # Enhanced logging with Google Agents integration status
            logger.info(f"Scan completed with Google Agents orchestration - Found {len(all_trending_posts)} trending posts")

            return {
                'trending_posts': all_trending_posts,
                'scan_summaries': scan_summaries,
                'risk_assessment': risk_assessment,
                'total_posts_found': len(all_trending_posts),
                'posts_with_scraped_content': posts_with_scraped_content,
                'total_links_scraped': total_scraped,
                'risk_distribution': risk_distribution,
                'timestamp': datetime.now().isoformat(),
                'orchestration_type': 'Google Agents SDK',
                'workflow_details': workflow_result
            }
            
        except Exception as e:
            logger.error(f"Error processing Google workflow results: {e}")
            return {
                'trending_posts': [],
                'scan_summaries': [f"Error processing results: {str(e)}"],
                'risk_assessment': f"Workflow processing failed: {str(e)}",
                'total_posts_found': 0,
                'posts_with_scraped_content': 0,
                'total_links_scraped': 0,
                'risk_distribution': {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0},
                'timestamp': datetime.now().isoformat(),
                'orchestration_type': 'Google Agents SDK (Error)',
                'error': str(e)
            }