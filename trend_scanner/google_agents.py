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
                    # Handle Reddit scanner
                    if hasattr(tool, '_run') and 'scan' in task_description.lower() and 'reddit' in task_description.lower():
                        try:
                            import re
                            subreddit_match = re.search(r'r/(\w+)', task_description)
                            target_subreddit = subreddit_match.group(1) if subreddit_match else 'worldnews'
                            
                            logger.info(f"Agent {self.role} executing Reddit scan for r/{target_subreddit}")
                            tool_result = tool._run(target_subreddit)
                            
                            result = {
                                'agent_role': self.role,
                                'task': task_description,
                                'result': tool_result,
                                'timestamp': datetime.now().isoformat(),
                                'tool_used': True,
                                'platform': 'reddit'
                            }
                            
                            self.history.append(result)
                            return result
                            
                        except Exception as tool_error:
                            logger.error(f"Reddit tool execution failed: {tool_error}")
                    
                    # Handle Threads scanner
                    elif hasattr(tool, '_run') and 'scan' in task_description.lower() and 'threads' in task_description.lower():
                        try:
                            import re
                            # Extract username from task description (e.g., "Scan @username for...")
                            username_match = re.search(r'@(\w+)', task_description)
                            target_username = username_match.group(1) if username_match else None
                            
                            if not target_username:
                                logger.error("No Threads username found in task description")
                                break
                            
                            logger.info(f"Agent {self.role} executing Threads scan for @{target_username}")
                            tool_result = tool._run(target_username)
                            
                            result = {
                                'agent_role': self.role,
                                'task': task_description,
                                'result': tool_result,
                                'timestamp': datetime.now().isoformat(),
                                'tool_used': True,
                                'platform': 'threads'
                            }
                            
                            self.history.append(result)
                            return result
                            
                        except Exception as tool_error:
                            logger.error(f"Threads tool execution failed: {tool_error}")
                    
                    # Handle Telegram scanner
                    elif hasattr(tool, '_run') and 'scan' in task_description.lower() and 'telegram' in task_description.lower():
                        try:
                            import re
                            # Extract channel from task description
                            channel_match = re.search(r'Scan\s+(\S+)\s+on Telegram', task_description)
                            target_channel = channel_match.group(1) if channel_match else None
                            
                            if not target_channel:
                                logger.error("No Telegram channel found in task description")
                                break
                            
                            logger.info(f"Agent {self.role} executing Telegram scan for {target_channel}")
                            tool_result = tool._run(target_channel)
                            
                            result = {
                                'agent_role': self.role,
                                'task': task_description,
                                'result': tool_result,
                                'timestamp': datetime.now().isoformat(),
                                'tool_used': True,
                                'platform': 'telegram'
                            }
                            
                            self.history.append(result)
                            return result
                            
                        except Exception as tool_error:
                            logger.error(f"Telegram tool execution failed: {tool_error}")
                    
                    # Handle Twitter scanner
                    elif hasattr(tool, '_run') and 'scan' in task_description.lower() and 'twitter' in task_description.lower():
                        try:
                            import re
                            # Extract target from task description (capture everything until "on Twitter")
                            target_match = re.search(r'Scan\s+(.+?)\s+on Twitter', task_description)
                            scan_type_match = re.search(r'scan_type:\s*(\w+)', task_description)
                            
                            target = target_match.group(1).strip() if target_match else None
                            scan_type = scan_type_match.group(1) if scan_type_match else 'user'
                            
                            if not target:
                                logger.error(f"No Twitter target found in task description: {task_description[:100]}")
                                break
                            
                            logger.info(f"Agent {self.role} executing Twitter scan for {target} (type: {scan_type})")
                            tool_result = tool._run(target, scan_type)
                            
                            result = {
                                'agent_role': self.role,
                                'task': task_description,
                                'result': tool_result,
                                'timestamp': datetime.now().isoformat(),
                                'tool_used': True,
                                'platform': 'twitter'
                            }
                            
                            self.history.append(result)
                            return result
                            
                        except Exception as tool_error:
                            logger.error(f"Twitter tool execution failed: {tool_error}")
                            import traceback
                            logger.error(traceback.format_exc())
            
            # Clean context to avoid circular references
            safe_context = {}
            if context:
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
            
            # Create context-aware prompt with special handling for trending posts
            if safe_context:
                if 'previous_results' in safe_context and isinstance(context.get('previous_results'), list):
                    previous_results = context['previous_results']
                    combined_posts = []
                    
                    # Extract trending posts from all previous results
                    for prev_result in previous_results:
                        if (prev_result.get('tool_used') and 
                            'result' in prev_result and 
                            isinstance(prev_result['result'], str)):
                            try:
                                tool_data = json.loads(prev_result['result'])
                                if 'trending_posts' in tool_data:
                                    platform = prev_result.get('platform', 'unknown')
                                    posts = tool_data['trending_posts']
                                    for post in posts:
                                        post['source_platform'] = platform
                                    combined_posts.extend(posts)
                            except (json.JSONDecodeError, KeyError) as e:
                                logger.warning(f"Failed to parse previous result: {e}")
                    
                    if combined_posts:
                        posts_summary = f"Found {len(combined_posts)} total trending posts across platforms:\n"
                        
                        # Group by platform
                        reddit_posts = [p for p in combined_posts if p.get('source_platform') == 'reddit']
                        threads_posts = [p for p in combined_posts if p.get('source_platform') == 'threads']
                        
                        if reddit_posts:
                            posts_summary += f"\nReddit: {len(reddit_posts)} posts\n"
                            for i, post in enumerate(reddit_posts[:3], 1):
                                posts_summary += f"  {i}. '{post.get('title', 'No title')}' (Risk: {post.get('risk_level', 'Unknown')})\n"
                        
                        if threads_posts:
                            posts_summary += f"\nThreads: {len(threads_posts)} posts\n"
                            for i, post in enumerate(threads_posts[:3], 1):
                                posts_summary += f"  {i}. '{post.get('title', 'No title')}' (Risk: {post.get('risk_level', 'Unknown')})\n"
                        
                        context_text = f"Previous scan results from multiple platforms:\n{posts_summary}\nFull data available for cross-platform analysis."
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
            
            {"IMPORTANT: Analyze posts from BOTH Reddit and Threads platforms. Identify cross-platform trends and coordinated misinformation campaigns." if "risk_assessor" in self.role.lower() or "assess" in task_description.lower() else ""}
            
            Execute this task thoroughly and provide detailed results.
            """
            
            # Execute with Gemini
            response = self.model.generate_content(prompt)
            response_text = getattr(response, 'text', str(response))
            
            # Special handling for Content Risk Assessor - provide actual trending posts data from both platforms
            if ("risk_assessor" in self.role.lower() or "assess" in task_description.lower()) and context and 'previous_results' in context:
                try:
                    combined_posts = []
                    previous_results = context['previous_results']
                    
                    for prev_result in previous_results:
                        if (prev_result.get('tool_used') and 'result' in prev_result):
                            tool_data = json.loads(prev_result['result'])
                            if 'trending_posts' in tool_data and tool_data['trending_posts']:
                                platform = prev_result.get('platform', 'unknown')
                                posts = tool_data['trending_posts']
                                for post in posts:
                                    post['source_platform'] = platform
                                combined_posts.extend(posts)
                    
                    if combined_posts:
                        # Create detailed cross-platform analysis prompt
                        detailed_prompt = f"""
                        You are a Content Risk Assessor analyzing trending posts from MULTIPLE PLATFORMS.
                        
                        TRENDING POSTS DATA FROM ALL PLATFORMS:
                        {json.dumps(combined_posts, indent=2)}
                        
                        Task: {task_description}
                        
                        Perform CROSS-PLATFORM ANALYSIS:
                        1. Identify similar narratives or claims appearing on both Reddit and Threads
                        2. Detect potential coordinated misinformation campaigns
                        3. Assess which platform shows higher viral potential for each topic
                        4. Compare engagement patterns and risk levels across platforms
                        
                        For each high-priority post, provide:
                        - Platform-specific risk assessment
                        - Cross-platform correlation analysis
                        - Viral potential on each platform
                        - Priority ranking considering multi-platform spread
                        - Recommended verification strategy
                        
                        Provide a structured analysis covering both platforms and cross-platform trends.
                        """
                        
                        # Re-execute with detailed cross-platform data
                        response = self.model.generate_content(detailed_prompt)
                        response_text = getattr(response, 'text', str(response))
                        logger.info(f"Content Risk Assessor provided with {len(combined_posts)} posts from multiple platforms for cross-platform analysis")
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
            safe_results = []
            for result in results:
                safe_result = {
                    'agent_role': result.get('agent_role', 'Unknown'),
                    'task': result.get('task', 'No task description')[:100],
                    'result_snippet': str(result.get('result', 'No result'))[:200],
                    'timestamp': result.get('timestamp', 'Unknown'),
                    'has_error': 'error' in result,
                    'platform': result.get('platform', 'unknown')
                }
                safe_results.append(safe_result)
            
            summary_prompt = f"""
            Analyze these workflow results and provide a concise summary:
            
            Total agents executed: {len(safe_results)}
            Results overview: {safe_results}
            
            Provide:
            1. Key findings from each agent and platform
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
        self.model = genai.GenerativeModel('gemini-2.5-flash')
    
    def create_multi_platform_workflow(self, reddit_tool, threads_tool, telegram_tool, twitter_tool,
                                        subreddits: List[str], threads_profiles: List[str], telegram_channels: List[str],
                                        twitter_accounts: List[str], twitter_keywords: List[str], twitter_scan_type: str = 'both',
                                        twitter_auto_discover: bool = False) -> Dict[str, Any]:
        """Create a multi-platform workflow scanning Reddit, Threads, Telegram, and Twitter in parallel"""
        
        # Create specialized agents
        reddit_agent = self.create_agent(
            name="reddit_scanner",
            role="Reddit Trend Scout",
            goal="Identify rapidly trending posts on Reddit that could contain misinformation",
            tools=[reddit_tool]
        )
        
        threads_agent = self.create_agent(
            name="threads_scanner",
            role="Threads Trend Scout",
            goal="Identify trending threads posts that could contain misinformation",
            tools=[threads_tool]
        ) if threads_tool else None
        
        telegram_agent = self.create_agent(
            name="telegram_scanner",
            role="Telegram Channel Trend Scout",
            goal="Identify trending messages on Telegram channels that could contain misinformation",
            tools=[telegram_tool]
        ) if telegram_tool else None
        
        twitter_agent = self.create_agent(
            name="twitter_scanner",
            role="Twitter/X Trend Scout",
            goal="Identify trending tweets from users or trending topics that could contain misinformation",
            tools=[twitter_tool]
        ) if twitter_tool else None
        
        assessor_agent = self.create_agent(
            name="risk_assessor", 
            role="Cross-Platform Content Risk Assessor",
            goal="Evaluate and prioritize trending posts from multiple platforms by misinformation risk",
            tools=[]
        )
        
        # Define parallel scanning tasks
        parallel_scan_tasks = []
        
        # Add Reddit scanning tasks
        for subreddit in subreddits:
            parallel_scan_tasks.append({
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
        
        # Add Threads scanning tasks
        for profile in threads_profiles:
            parallel_scan_tasks.append({
                'agent': 'threads_scanner',
                'description': f"""
                Scan @{profile} on Threads for trending posts with potential misinformation.
                Look for:
                - High engagement posts (rapid like growth)
                - Unverified claims or misleading information
                - Emotional manipulation or sensationalism
                - Viral spread patterns
                
                Analyze thread content for misinformation indicators.
                """
            })
        
        # Add Telegram scanning tasks
        for channel in telegram_channels:
            parallel_scan_tasks.append({
                'agent': 'telegram_scanner',
                'description': f"""
                Scan {channel} on Telegram for trending messages with potential misinformation.
                Look for:
                - High view count messages (viral spread)
                - Unverified claims or conspiracy theories
                - Coordinated messaging patterns
                - Links to questionable sources
                
                Analyze message content and linked media for misinformation indicators.
                """
            })
        
        # Add Twitter scanning tasks
        twitter_targets = []
        
        # Handle auto-discovery of trending topics
        if twitter_auto_discover and twitter_tool:
            logger.info("Auto-discovering Twitter trending topics...")
            try:
                # Get ALL trending topics from Twitter (no limit)
                trending_topics = twitter_tool.get_trending_topics(limit=50)
                if trending_topics:
                    logger.info(f"Auto-discovered {len(trending_topics)} trending topics: {trending_topics}")
                    twitter_keywords = trending_topics
                else:
                    logger.warning("No trending topics auto-discovered, using manual keywords")
            except Exception as e:
                logger.error(f"Failed to auto-discover trending topics: {e}")
                logger.warning("Falling back to manual keywords")
        
        if twitter_accounts:
            twitter_targets.extend(twitter_accounts)
        if twitter_keywords:
            twitter_targets.extend(twitter_keywords)
        
        for target in twitter_targets:
            # Determine if it's a user account or keyword
            is_keyword = target in twitter_keywords if twitter_keywords else False
            scan_type = twitter_scan_type if twitter_scan_type != 'both' else ('trending' if is_keyword else 'user')
            
            parallel_scan_tasks.append({
                'agent': 'twitter_scanner',
                'description': f"""
                Scan {target} on Twitter/X (scan_type: {scan_type}) for trending tweets with potential misinformation.
                Look for:
                - High engagement tweets (viral spread via likes/retweets)
                - Unverified claims or misleading information
                - Coordinated bot activity or astroturfing
                - Emotional manipulation or divisive content
                - Links to questionable sources
                
                Analyze tweet content, quoted tweets, and linked media for misinformation indicators.
                """
            })
        
        # Execute parallel scanning
        platform_count = f"{len(subreddits)} subreddits"
        if threads_profiles:
            platform_count += f", {len(threads_profiles)} Threads profiles"
        if telegram_channels:
            platform_count += f", {len(telegram_channels)} Telegram channels"
        if twitter_targets:
            platform_count += f", {len(twitter_targets)} Twitter targets"
        logger.info(f"Starting parallel scan across {platform_count}")
        parallel_results = self.parallel_workflow(parallel_scan_tasks)
        
        # Sequential risk assessment after all scans complete
        assessment_task = [{
            'agent': 'risk_assessor',
            'description': """
            You are a Cross-Platform Content Risk Assessor. Analyze ALL trending posts from Reddit, Threads, and Telegram.
            
            IMPORTANT: Use the trending posts data from previous agents' results across all platforms.
            
            Perform comprehensive cross-platform analysis:
            1. Identify narratives appearing on multiple platforms (coordinated campaigns)
            2. Compare risk levels and engagement patterns between Reddit and Threads
            3. Detect platform-specific misinformation strategies
            4. Assess viral potential on each platform
            5. Priority ranking for fact-checking considering multi-platform spread
            
            For each platform and cross-platform trends, provide:
            - Detailed risk analysis (HIGH/MEDIUM/LOW) with reasoning
            - Priority ranking (1-10 scale) for manual fact-checking
            - Key claims needing verification
            - Platform-specific spread analysis
            - Coordinated campaign detection
            - Recommended actions per platform
            
            Format your response as structured analysis covering:
            - Reddit-specific findings
            - Threads-specific findings  
            - Cross-platform trends and coordinated campaigns
            - Overall priority recommendations
            """
        }]
        
        # Create combined context from parallel results
        assessment_context = {'previous_results': parallel_results['results']}
        
        # Execute assessment with combined context
        logger.info("Executing cross-platform risk assessment")
        assessment_result = self.agents['risk_assessor'].execute_task(
            assessment_task[0]['description'],
            assessment_context
        )
        
        # Combine all results
        all_results = parallel_results['results'] + [assessment_result]
        
        final_result = {
            'workflow_type': 'multi_platform',
            'platforms': ['reddit', 'threads'],
            'total_tasks': len(all_results),
            'results': all_results,
            'summary': self._create_workflow_summary(all_results),
            'timestamp': datetime.now().isoformat()
        }
        
        self.workflow_history.append(final_result)
        return final_result


# Keep the old class name for backward compatibility
class GoogleAgentsManager(GoogleAIManager):
    """Backward compatibility alias for GoogleAIManager"""
    pass


class TrendScannerOrchestrator:
    """Main orchestrator for multi-platform scanning (Reddit + Threads + Telegram + Twitter)"""
    
    def __init__(self, reddit_config: Dict[str, str], gemini_api_key: Optional[str] = None, 
                 threads_enabled: bool = True, telegram_enabled: bool = False, telegram_config: Dict[str, str] = None,
                 twitter_enabled: bool = False, twitter_config: Dict[str, str] = None):
        gemini_key = gemini_api_key or os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
        if not gemini_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY must be provided")
        
        # Initialize Google Agents Manager
        try:
            self.google_agents = GoogleAgentsManager(api_key=gemini_key)
            logger.info("Google Agents orchestration initialized successfully")
        except Exception as e:
            logger.error(f"Google Agents orchestration failed: {e}")
            raise

        # Initialize PRAW Reddit client
        import praw
        self.reddit = praw.Reddit(
            client_id=reddit_config['client_id'],
            client_secret=reddit_config['client_secret'],
            user_agent=reddit_config['user_agent']
        )
        
        # Test Reddit authentication
        try:
            test_user = self.reddit.user.me()
            logger.info(f"PRAW Reddit client authenticated successfully")
        except Exception as e:
            logger.warning(f"Reddit auth warning (may be read-only): {e}")
            logger.info("PRAW Reddit client initialized (read-only mode)")

        # Simple LLM wrapper
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

        # Initialize Reddit tool
        from .tools import RedditScanTool
        self.reddit_tool = RedditScanTool(
            self.reddit, 
            self.llm, 
            velocity_threshold=25.0, 
            min_score_threshold=50,
            google_api_key=gemini_key
        )

        # Initialize Threads tool if enabled
        self.threads_enabled = threads_enabled
        if threads_enabled:
            try:
                from .tools import ThreadsScanTool
                self.threads_tool = ThreadsScanTool(
                    self.llm,
                    velocity_threshold=10,
                    min_like_threshold=100,
                    google_api_key=gemini_key
                )
                logger.info("Threads scanner initialized successfully")
            except Exception as e:
                logger.warning(f"Threads scanner initialization failed: {e}")
                self.threads_enabled = False
                self.threads_tool = None
        else:
            self.threads_tool = None

        # Initialize Telegram tool if enabled
        self.telegram_enabled = telegram_enabled
        if telegram_enabled:
            if not telegram_config:
                logger.warning("Telegram enabled but no config provided")
                self.telegram_enabled = False
                self.telegram_tool = None
            else:
                try:
                    from .tools import TelegramScanTool
                    self.telegram_tool = TelegramScanTool(
                        api_id=telegram_config.get('api_id'),
                        api_hash=telegram_config.get('api_hash'),
                        phone=telegram_config.get('phone'),
                        session_name=telegram_config.get('session_name', 'trend_scanner_session'),
                        llm_wrapper=self.llm,
                        velocity_threshold=50,
                        min_views_threshold=1000,
                        google_api_key=gemini_key
                    )
                    logger.info("Telegram scanner initialized successfully")
                except Exception as e:
                    logger.warning(f"Telegram scanner initialization failed: {e}")
                    self.telegram_enabled = False
                    self.telegram_tool = None
        else:
            self.telegram_tool = None

        # Initialize Twitter tool if enabled
        self.twitter_enabled = twitter_enabled
        if twitter_enabled:
            if not twitter_config:
                logger.warning("Twitter enabled but no config provided")
                self.twitter_enabled = False
                self.twitter_tool = None
            else:
                try:
                    from .tools import TwitterScanTool
                    self.twitter_tool = TwitterScanTool(
                        username=twitter_config.get('username'),
                        email=twitter_config.get('email'),
                        password=twitter_config.get('password'),
                        cookies_file=twitter_config.get('cookies_file', 'twitter_cookies.json'),
                        llm_wrapper=self.llm,
                        velocity_threshold=200,  # Higher threshold: 200 engagements/hour (viral content)
                        min_engagement_threshold=100,  # Higher threshold: 100 total engagement (popular tweets)
                        google_api_key=gemini_key
                    )
                    logger.info("Twitter scanner initialized successfully")
                except Exception as e:
                    logger.warning(f"Twitter scanner initialization failed: {e}")
                    self.twitter_enabled = False
                    self.twitter_tool = None
        else:
            self.twitter_tool = None

        # Target configuration
        self.target_subreddits = []
        self.target_threads_profiles = []
        self.target_telegram_channels = []
        self.target_twitter_accounts = []
        self.target_twitter_keywords = []
        self.twitter_scan_type = 'both'
        self.twitter_auto_discover = False

        # Initialize agents
        self._setup_google_agents()

    def _setup_google_agents(self):
        """Setup Google agents for multi-platform orchestration"""
        # Reddit scanner agent
        self.reddit_scanner = self.google_agents.create_agent(
            name="reddit_scanner",
            role="Enhanced Reddit Trend Scout",
            goal="Identify rapidly trending posts on Reddit that could contain misinformation",
            tools=[self.reddit_tool]
        )
        
        # Threads scanner agent (if enabled)
        if self.threads_enabled and self.threads_tool:
            self.threads_scanner = self.google_agents.create_agent(
                name="threads_scanner",
                role="Threads Trend Scout",
                goal="Identify trending posts on Threads that could contain misinformation",
                tools=[self.threads_tool]
            )
        else:
            self.threads_scanner = None
        
        # Telegram scanner agent (if enabled)
        if self.telegram_enabled and self.telegram_tool:
            self.telegram_scanner = self.google_agents.create_agent(
                name="telegram_scanner",
                role="Telegram Channel Trend Scout",
                goal="Identify trending messages on Telegram channels that could contain misinformation",
                tools=[self.telegram_tool]
            )
        else:
            self.telegram_scanner = None
        
        # Twitter scanner agent (if enabled)
        if self.twitter_enabled and self.twitter_tool:
            self.twitter_scanner = self.google_agents.create_agent(
                name="twitter_scanner",
                role="Twitter/X Trend Scout",
                goal="Identify trending tweets from users or trending topics that could contain misinformation",
                tools=[self.twitter_tool]
            )
        else:
            self.twitter_scanner = None
        
        # Risk assessor agent for cross-platform analysis
        self.risk_assessor = self.google_agents.create_agent(
            name="risk_assessor",
            role="Cross-Platform Content Risk Assessor", 
            goal="Evaluate risk levels across all platforms using cross-platform analysis",
            tools=[]
        )
        
        platforms_enabled = []
        if self.reddit_tool: platforms_enabled.append('Reddit')
        if self.threads_enabled: platforms_enabled.append('Threads')
        if self.telegram_enabled: platforms_enabled.append('Telegram')
        if self.twitter_enabled: platforms_enabled.append('Twitter')
        logger.info(f"Google agents created for platforms: {', '.join(platforms_enabled)}")
    
    def set_target_subreddits(self, subreddits: List[str]):
        """Set target Reddit subreddits"""
        self.target_subreddits = subreddits
        logger.info(f"Target subreddits configured: {subreddits}")
    
    def set_target_threads_profiles(self, profiles: List[str]):
        """Set target Threads profiles"""
        self.target_threads_profiles = profiles
        logger.info(f"Target Threads profiles configured: {profiles}")
    
    def set_target_telegram_channels(self, channels: List[str]):
        """Set target Telegram channels"""
        self.target_telegram_channels = channels
        logger.info(f"Target Telegram channels configured: {channels}")
    
    def set_target_twitter(self, accounts: List[str], keywords: List[str], scan_type: str = 'both', auto_discover: bool = False):
        """
        Set target Twitter accounts and keywords
        
        Args:
            accounts: List of Twitter usernames to scan
            keywords: List of keywords/hashtags to search (ignored if auto_discover=True)
            scan_type: 'user', 'trending', or 'both'
            auto_discover: If True, automatically fetch trending topics instead of using manual keywords
        """
        self.target_twitter_accounts = accounts or []
        self.target_twitter_keywords = keywords or []
        self.twitter_scan_type = scan_type
        self.twitter_auto_discover = auto_discover
        
        if auto_discover:
            logger.info(f"Target Twitter configured: {len(accounts)} accounts, AUTO-DISCOVER trending topics, scan_type={scan_type}")
        else:
            logger.info(f"Target Twitter configured: {len(accounts)} accounts, {len(keywords)} manual keywords, scan_type={scan_type}")
    
    def add_target_subreddit(self, subreddit: str):
        """Add a single subreddit"""
        if subreddit not in self.target_subreddits:
            self.target_subreddits.append(subreddit)
            logger.info(f"Added subreddit: r/{subreddit}")
    
    def add_target_threads_profile(self, profile: str):
        """Add a single Threads profile"""
        if profile not in self.target_threads_profiles:
            self.target_threads_profiles.append(profile)
            logger.info(f"Added Threads profile: @{profile}")
    
    def add_target_telegram_channel(self, channel: str):
        """Add a single Telegram channel"""
        if channel not in self.target_telegram_channels:
            self.target_telegram_channels.append(channel)
            logger.info(f"Added Telegram channel: {channel}")

    def scan_trending_content(self) -> Dict[str, Any]:
        """Execute multi-platform trend scanning with parallel execution"""
        platforms = []
        if self.target_subreddits:
            platforms.append("Reddit")
        if self.target_threads_profiles and self.threads_enabled:
            platforms.append("Threads")
        if self.target_telegram_channels and self.telegram_enabled:
            platforms.append("Telegram")
        if (self.target_twitter_accounts or self.target_twitter_keywords) and self.twitter_enabled:
            platforms.append(f"Twitter/{self.twitter_scan_type}")
        
        logger.info(f"Starting multi-platform trend scanning ({' + '.join(platforms)})...")
        
        has_targets = (self.target_subreddits or self.target_threads_profiles or 
                      self.target_telegram_channels or self.target_twitter_accounts or self.target_twitter_keywords)
        if not has_targets:
            raise ValueError("No scan targets configured. Use set_target_subreddits(), set_target_threads_profiles(), set_target_telegram_channels(), or set_target_twitter()")
        
        # Use multi-platform workflow
        workflow_result = self.google_agents.create_multi_platform_workflow(
            reddit_tool=self.reddit_tool,
            threads_tool=self.threads_tool if self.threads_enabled else None,
            telegram_tool=self.telegram_tool if self.telegram_enabled else None,
            twitter_tool=self.twitter_tool if self.twitter_enabled else None,
            subreddits=self.target_subreddits,
            threads_profiles=self.target_threads_profiles if self.threads_enabled else [],
            telegram_channels=self.target_telegram_channels if self.telegram_enabled else [],
            twitter_accounts=self.target_twitter_accounts if self.twitter_enabled else [],
            twitter_keywords=self.target_twitter_keywords if self.twitter_enabled else [],
            twitter_scan_type=self.twitter_scan_type if self.twitter_enabled else 'both',
            twitter_auto_discover=getattr(self, 'twitter_auto_discover', False) if self.twitter_enabled else False
        )
        
        # Process workflow results
        return self._process_workflow_results(workflow_result)
    
    def _process_workflow_results(self, workflow_result: Dict[str, Any]) -> Dict[str, Any]:
        """Process multi-platform workflow results"""
        try:
            all_trending_posts = []
            reddit_posts = []
            threads_posts = []
            telegram_posts = []
            twitter_posts = []
            scan_summaries = []
            total_scraped = 0
            posts_with_scraped_content = 0
            
            # Extract results from workflow
            if 'results' in workflow_result:
                for result in workflow_result['results']:
                    platform = result.get('platform', 'unknown')
                    
                    # Reddit scanner results
                    if platform == 'reddit' and result.get('tool_used', False):
                        try:
                            scan_data = json.loads(result['result'])
                            if 'trending_posts' in scan_data:
                                posts = scan_data['trending_posts']
                                for post in posts:
                                    post['source_platform'] = 'reddit'
                                reddit_posts.extend(posts)
                                all_trending_posts.extend(posts)
                                total_scraped += scan_data.get('scraped_count', 0)
                                posts_with_scraped_content += len([p for p in posts if p.get('scraped_content')])
                            scan_summaries.append(f"Reddit: {scan_data.get('scan_summary', 'Completed')}")
                            logger.info(f"Processed Reddit scan: {len(posts)} posts")
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to parse Reddit results: {e}")
                    
                    # Threads scanner results
                    elif platform == 'threads' and result.get('tool_used', False):
                        try:
                            scan_data = json.loads(result['result'])
                            if 'trending_posts' in scan_data:
                                posts = scan_data['trending_posts']
                                for post in posts:
                                    post['source_platform'] = 'threads'
                                threads_posts.extend(posts)
                                all_trending_posts.extend(posts)
                                total_scraped += scan_data.get('scraped_count', 0)
                                posts_with_scraped_content += len([p for p in posts if p.get('scraped_content')])
                            scan_summaries.append(f"Threads: {scan_data.get('scan_summary', 'Completed')}")
                            logger.info(f"Processed Threads scan: {len(posts)} posts")
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to parse Threads results: {e}")
                    
                    # Telegram scanner results
                    elif platform == 'telegram' and result.get('tool_used', False):
                        try:
                            scan_data = json.loads(result['result'])
                            if 'trending_posts' in scan_data:
                                posts = scan_data['trending_posts']
                                for post in posts:
                                    post['source_platform'] = 'telegram'
                                telegram_posts.extend(posts)
                                all_trending_posts.extend(posts)
                                total_scraped += scan_data.get('scraped_count', 0)
                                posts_with_scraped_content += len([p for p in posts if p.get('scraped_content')])
                            scan_summaries.append(f"Telegram: {scan_data.get('scan_summary', 'Completed')}")
                            logger.info(f"Processed Telegram scan: {len(posts)} posts")
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to parse Telegram results: {e}")
                    
                    # Twitter scanner results
                    elif platform == 'twitter' and result.get('tool_used', False):
                        try:
                            scan_data = json.loads(result['result'])
                            if 'trending_posts' in scan_data:
                                posts = scan_data['trending_posts']
                                for post in posts:
                                    post['source_platform'] = 'twitter'
                                twitter_posts.extend(posts)
                                all_trending_posts.extend(posts)
                                total_scraped += scan_data.get('scraped_count', 0)
                                posts_with_scraped_content += len([p for p in posts if p.get('scraped_content')])
                            scan_summaries.append(f"Twitter: {scan_data.get('scan_summary', 'Completed')}")
                            logger.info(f"Processed Twitter scan: {len(posts)} posts")
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to parse Twitter results: {e}")
            
            # Calculate risk distribution across all platforms
            risk_distribution = {
                'HIGH': len([p for p in all_trending_posts if p.get('risk_level') == 'HIGH']),
                'MEDIUM': len([p for p in all_trending_posts if p.get('risk_level') == 'MEDIUM']),
                'LOW': len([p for p in all_trending_posts if p.get('risk_level') == 'LOW'])
            }
            
            # Platform-specific risk distribution
            platform_risk = {
                'reddit': {
                    'HIGH': len([p for p in reddit_posts if p.get('risk_level') == 'HIGH']),
                    'MEDIUM': len([p for p in reddit_posts if p.get('risk_level') == 'MEDIUM']),
                    'LOW': len([p for p in reddit_posts if p.get('risk_level') == 'LOW'])
                },
                'threads': {
                    'HIGH': len([p for p in threads_posts if p.get('risk_level') == 'HIGH']),
                    'MEDIUM': len([p for p in threads_posts if p.get('risk_level') == 'MEDIUM']),
                    'LOW': len([p for p in threads_posts if p.get('risk_level') == 'LOW'])
                },
                'telegram': {
                    'HIGH': len([p for p in telegram_posts if p.get('risk_level') == 'HIGH']),
                    'MEDIUM': len([p for p in telegram_posts if p.get('risk_level') == 'MEDIUM']),
                    'LOW': len([p for p in telegram_posts if p.get('risk_level') == 'LOW'])
                },
                'twitter': {
                    'HIGH': len([p for p in twitter_posts if p.get('risk_level') == 'HIGH']),
                    'MEDIUM': len([p for p in twitter_posts if p.get('risk_level') == 'MEDIUM']),
                    'LOW': len([p for p in twitter_posts if p.get('risk_level') == 'LOW'])
                }
            }
            
            risk_assessment = workflow_result.get('summary', 'Multi-platform orchestration completed')
            
            logger.info(f"Multi-platform scan completed - Reddit: {len(reddit_posts)}, Threads: {len(threads_posts)}, Telegram: {len(telegram_posts)}, Twitter: {len(twitter_posts)}, Total: {len(all_trending_posts)}")

            return {
                'trending_posts': all_trending_posts,
                'reddit_posts': reddit_posts,
                'threads_posts': threads_posts,
                'telegram_posts': telegram_posts,
                'twitter_posts': twitter_posts,
                'scan_summaries': scan_summaries,
                'risk_assessment': risk_assessment,
                'total_posts_found': len(all_trending_posts),
                'posts_by_platform': {
                    'reddit': len(reddit_posts),
                    'threads': len(threads_posts),
                    'telegram': len(telegram_posts),
                    'twitter': len(twitter_posts)
                },
                'posts_with_scraped_content': posts_with_scraped_content,
                'total_links_scraped': total_scraped,
                'risk_distribution': risk_distribution,
                'platform_risk_distribution': platform_risk,
                'timestamp': datetime.now().isoformat(),
                'orchestration_type': 'Google Agents SDK - Multi-Platform',
                'platforms': (['reddit'] if self.target_subreddits else []) + 
                            (['threads'] if self.threads_enabled else []) + 
                            (['telegram'] if self.telegram_enabled else []) + 
                            (['twitter'] if self.twitter_enabled else []),
                'workflow_details': workflow_result
            }
            
        except Exception as e:
            logger.error(f"Error processing multi-platform workflow: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'trending_posts': [],
                'reddit_posts': [],
                'threads_posts': [],
                'telegram_posts': [],
                'twitter_posts': [],
                'threads_posts': [],
                'scan_summaries': [f"Error: {str(e)}"],
                'risk_assessment': f"Workflow processing failed: {str(e)}",
                'total_posts_found': 0,
                'posts_by_platform': {'reddit': 0, 'threads': 0},
                'posts_with_scraped_content': 0,
                'total_links_scraped': 0,
                'risk_distribution': {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0},
                'platform_risk_distribution': {
                    'reddit': {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0},
                    'threads': {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
                },
                'timestamp': datetime.now().isoformat(),
                'orchestration_type': 'Google Agents SDK - Multi-Platform (Error)',
                'error': str(e)
            }