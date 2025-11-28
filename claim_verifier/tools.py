import requests
import json
from typing import Dict, List, Optional, Any
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from .config import config


class TextFactChecker:
    """Service for fact-checking textual claims using Google Custom Search API with fact-checking sites"""
    
    def __init__(self):
        self.api_key = config.GOOGLE_FACT_CHECK_API_KEY
        self.search_engine_id = config.GOOGLE_FACT_CHECK_CX
        self.base_url = "https://www.googleapis.com/customsearch/v1"
        
        # Configure Gemini for analysis
        genai.configure(api_key=config.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(config.GEMINI_MODEL)
        
        if not self.api_key:
            raise ValueError("Google Custom Search API key is required")
        if not self.search_engine_id:
            raise ValueError("Google Custom Search Engine ID (cx) is required")
    
    async def verify(self, text_input: str, claim_context: str = "Unknown context", claim_date: str = "Unknown date") -> Dict[str, Any]:
        """
        Verify a textual claim using Google Custom Search API with fact-checking sites
        
        Args:
            text_input: The text claim to verify
            claim_context: Context about the claim
            claim_date: Date when the claim was made
            
        Returns:
            Dictionary containing verification results
        """
        try:
            print(f"Starting verification for: {text_input}")
            # Search for fact-checked claims related to the input text
            search_results = await self._search_claims(text_input)
            
            if not search_results:
                return {
                    "verified": False,
                    "verdict": "no_content",
                    "message": "No fact-checked information found for this claim",
                    "confidence": "low",
                    "reasoning": "No reliable sources found to verify this claim",
                    "sources": {
                        "links": [],
                        "titles": [],
                        "count": 0
                    },
                    "claim_text": text_input,
                    "verification_date": claim_date
                }
            
            # Analyze the search results
            analysis = self._analyze_results(search_results, text_input)
            
            # Extract source links for clean output
            source_links = [result.get("link", "") for result in search_results[:5] if result.get("link")]
            source_titles = [result.get("title", "") for result in search_results[:5] if result.get("title")]
            
            return {
                "verified": analysis["verified"],
                "verdict": analysis["verdict"],
                "message": analysis["message"],
                "confidence": analysis.get("confidence", "medium"),
                "reasoning": analysis.get("reasoning", "Analysis completed"),
                "sources": {
                    "links": source_links,
                    "titles": source_titles,
                    "count": len(search_results)
                },
                "claim_text": text_input,
                "verification_date": claim_date
            }
            
        except Exception as e:
            return {
                "verified": False,
                "verdict": "error",
                "message": f"Error during fact-checking: {str(e)}",
                "details": {
                    "claim_text": text_input,
                    "claim_context": claim_context,
                    "claim_date": claim_date,
                    "error": str(e)
                }
            }
    
    async def verify_batch(self, claims_batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Verify multiple claims in a single batch using optimized Gemini processing
        
        Args:
            claims_batch: List of claim dictionaries with keys: text_input, claim_context, claim_date
            
        Returns:
            List of verification results for each claim
        """
        try:
            print(f"Starting batch verification for {len(claims_batch)} claims")
            
            # Process all search operations first
            search_results_list = []
            for i, claim_data in enumerate(claims_batch):
                text_input = claim_data.get('text_input', '')
                print(f"Searching for claim {i+1}/{len(claims_batch)}: {text_input[:50]}...")
                
                search_results = await self._search_claims(text_input)
                search_results_list.append({
                    'claim_data': claim_data,
                    'search_results': search_results
                })
            
            # Batch analyze all claims with Gemini in a single call
            batch_analysis = await self._analyze_batch_with_gemini(search_results_list)
            
            # Format final results
            verification_results = []
            for i, (claim_item, analysis) in enumerate(zip(search_results_list, batch_analysis)):
                claim_data = claim_item['claim_data']
                search_results = claim_item['search_results']
                
                if not search_results:
                    verification_results.append({
                        "verified": False,
                        "verdict": "no_content",
                        "message": "No fact-checked information found for this claim",
                        "confidence": "low",
                        "reasoning": "No reliable sources found to verify this claim",
                        "sources": {
                            "links": [],
                            "titles": [],
                            "count": 0
                        },
                        "claim_text": claim_data.get('text_input', ''),
                        "verification_date": claim_data.get('claim_date', 'Unknown date')
                    })
                    continue
                
                # Extract source links for clean output
                source_links = [result.get("link", "") for result in search_results[:5] if result.get("link")]
                source_titles = [result.get("title", "") for result in search_results[:5] if result.get("title")]
                
                verification_results.append({
                    "verified": analysis["verified"],
                    "verdict": analysis["verdict"],
                    "message": analysis["message"],
                    "confidence": analysis.get("confidence", "medium"),
                    "reasoning": analysis.get("reasoning", "Analysis completed"),
                    "sources": {
                        "links": source_links,
                        "titles": source_titles,
                        "count": len(search_results)
                    },
                    "claim_text": claim_data.get('text_input', ''),
                    "verification_date": claim_data.get('claim_date', 'Unknown date')
                })
            
            print(f"Batch verification completed for {len(verification_results)} claims")
            return verification_results
            
        except Exception as e:
            print(f"Batch verification failed: {e}")
            # Return error results for all claims
            error_results = []
            for claim_data in claims_batch:
                error_results.append({
                    "verified": False,
                    "verdict": "error",
                    "message": f"Error during batch fact-checking: {str(e)}",
                    "details": {
                        "claim_text": claim_data.get('text_input', ''),
                        "claim_context": claim_data.get('claim_context', ''),
                        "claim_date": claim_data.get('claim_date', ''),
                        "error": str(e)
                    }
                })
            return error_results
    
    async def _search_claims(self, query: str) -> List[Dict[str, Any]]:
        """
        Search for fact-checked claims using Google Custom Search API with LLM-powered fallback strategies
        
        Args:
            query: The search query
            
        Returns:
            List of search results
        """
        # Try the original query first
        results = await self._perform_search(query)
        
        # If no results, use LLM to create alternative queries
        if not results:
            print("No results found, using LLM to create alternative queries...")
            
            alternative_queries = self._create_alternative_queries(query)
            print(f"Generated alternative queries: {alternative_queries}")

            results = await self._perform_search(alternative_queries)
            if results:
                print(f"Found {len(results)} results with alternative query")
            else:
                print("No results found with alternative query")
        return results
    
    async def _perform_search(self, query: str) -> List[Dict[str, Any]]:
        """
        Perform a single search request
        
        Args:
            query: The search query
            
        Returns:
            List of search results
        """
        params = {
            "q": query,
            "key": self.api_key,
            "cx": self.search_engine_id,
            "num": 10  # Limit results to 10 for better performance
        }
        
        try:
            print(f"Making request to: {self.base_url}")
            print(f"Params: {params}")
            
            response = requests.get(self.base_url, params=params, timeout=30)
            print(f"Response status: {response.status_code}")
            print(f"Response text: {response.text}")
            
            response.raise_for_status()
            
            data = response.json()
            items = data.get("items", [])
            
            return items
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed: {str(e)}")
        except json.JSONDecodeError as e:
            raise Exception(f"Failed to parse API response: {str(e)}")
        except Exception as e:
            raise Exception(f"Search error: {str(e)}")
    
    def _create_alternative_queries(self, query: str) -> List[str]:
        """
        Use LLM to create alternative search queries (broader and simpler)
        
        Args:
            query: Original query
            
        Returns:
            List of alternative queries to try
        """
        prompt = f"""
You are a search query optimizer. Given a fact-checking query that returned no results, create alternative queries that might find relevant information.

ORIGINAL QUERY: "{query}"

Create an alternative query:
1. A BROADER query that removes specific assumptions and focuses on key entities/events

Examples:
- "Is it true the CEO of Astronomer resigned because of toxic workplace allegations?" 
  → Broader: "Astronomer CEO resignation"

- "Did Apple release a new iPhone with 5G in 2023?"
  → Broader: "Apple iPhone 2023 release"

Respond in this exact JSON format:
{{
    "broader_query": "your broader query here",
}}
"""
        
        try:
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Try to parse JSON response
            if response_text.startswith('```json'):
                response_text = response_text.replace('```json', '').replace('```', '').strip()
            elif response_text.startswith('```'):
                response_text = response_text.replace('```', '').strip()
            
            alternatives = json.loads(response_text)
            
            # Return both alternatives
            queries = []
            if alternatives.get("broader_query") and alternatives["broader_query"] != query:
                queries.append(alternatives["broader_query"])
            if alternatives.get("simpler_query") and alternatives["simpler_query"] != query:
                queries.append(alternatives["simpler_query"])
            
            return queries
            
        except Exception as e:
            print(f"Failed to create alternative queries with LLM: {e}")
    
    def _analyze_results(self, results: List[Dict[str, Any]], original_text: str) -> Dict[str, Any]:
        """
        Analyze the search results using Gemini AI to determine overall verdict
        
        Args:
            results: List of search results from the API
            original_text: The original text being verified
            
        Returns:
            Analysis results including verdict and message
        """
        if not results:
            return {
                "verified": False,
                "verdict": "no_content",
                "message": "No fact-checked information found for this claim"
            }
        
        # Filter relevant results
        relevant_results = []
        for result in results:
            title = result.get("title", "").lower()
            snippet = result.get("snippet", "").lower()
            original_lower = original_text.lower()
            
            # Check if the result is relevant to our original text
            relevance_score = self._calculate_relevance(result, original_text)
            
            print(f"Relevance score for '{title[:50]}...': {relevance_score:.3f}")
            if relevance_score > 0.05:  # Very low threshold to catch all relevant results
                relevant_results.append(result)
        
        if not relevant_results:
            return {
                "verified": False,
                "verdict": "no_content",
                "message": "No relevant fact-checked information found for this specific claim"
            }
        
        # Use Gemini to analyze the results
        try:
            analysis = self._analyze_with_gemini(original_text, relevant_results)
            return analysis
        except Exception as e:
            print(f"Gemini analysis failed: {str(e)}")
            # Fallback to simple analysis
            return self._fallback_analysis(relevant_results)
    
    def _calculate_relevance(self, result: Dict[str, Any], original_text: str) -> float:
        """
        Calculate relevance score using TF-IDF similarity with multiple components
        
        Args:
            result: Search result dictionary
            original_text: Original text being verified
            
        Returns:
            Relevance score between 0 and 1
        """
        score = 0.0
        
        # 1. Title relevance (40% weight)
        title = result.get("title", "")
        if title:
            title_score = self._tfidf_similarity(title, original_text)
            score += title_score * 0.6
        
        # 2. Snippet relevance (30% weight)  
        snippet = result.get("snippet", "")
        if snippet:
            snippet_score = self._tfidf_similarity(snippet, original_text)
            score += snippet_score * 0.4
        
        # 3. Fact-check specific bonus (30% weight)
        factcheck_score = self._has_factcheck_data(result)
        score += factcheck_score * 0.1
        
        return min(1.0, score)
    
    def _tfidf_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate TF-IDF cosine similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        if not text1.strip() or not text2.strip():
            return 0.0
        
        try:
            # Preprocess texts
            texts = [self._preprocess_text(text1), self._preprocess_text(text2)]
            
            # Create TF-IDF vectors
            vectorizer = TfidfVectorizer(
                stop_words='english',
                ngram_range=(1, 2),  # Include bigrams
                max_features=500,
                lowercase=True
            )
            
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            # Calculate cosine similarity
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            return float(similarity)
            
        except Exception as e:
            print(f"TF-IDF calculation failed: {e}")
            # Fallback to simple word overlap
            return self._simple_word_overlap(text1, text2)
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for TF-IDF analysis
        
        Args:
            text: Raw text
            
        Returns:
            Preprocessed text
        """
        import re
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def _simple_word_overlap(self, text1: str, text2: str) -> float:
        """
        Fallback similarity calculation using word overlap
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _has_factcheck_data(self, result: Dict[str, Any]) -> float:
        """
        Check if result has fact-check specific metadata
        
        Args:
            result: Search result dictionary
            
        Returns:
            1.0 if has fact-check data, 0.0 otherwise
        """
        # Check for ClaimReview metadata
        pagemap = result.get("pagemap", {})
        claim_review = pagemap.get("ClaimReview", [])
        
        if claim_review:
            return 1.0
        
        # Check for fact-check related keywords in URL or title
        url = result.get("link", "").lower()
        title = result.get("title", "").lower()
        
        factcheck_keywords = [
            "fact-check", "factcheck", "snopes", "politifact", 
            "factcrescendo", "boomlive", "newschecker", "afp"
        ]
        
        for keyword in factcheck_keywords:
            if keyword in url or keyword in title:
                return 1.0
        
        return 0.0
    
    def _analyze_with_gemini(self, original_text: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Use Gemini AI to analyze fact-check results and determine verdict
        
        Args:
            original_text: The original claim being verified
            results: List of relevant search results
            
        Returns:
            Analysis results with verdict and message
        """
        # Prepare the prompt
        results_text = ""
        for i, result in enumerate(results[:5], 1):  # Limit to top 5 results
            title = result.get("title", "")
            snippet = result.get("snippet", "")
            link = result.get("link", "")
            results_text += f"{i}. Title: {title}\n   Snippet: {snippet}\n   Link: {link}\n\n"
        
        prompt = f"""
You are a fact-checking expert. Analyze the following claim against the provided fact-checking sources.

CLAIM TO VERIFY: "{original_text}"

FACT-CHECKING SOURCES:
{results_text}

STEP-BY-STEP ANALYSIS:
1. What does each source say ACTUALLY HAPPENED?
2. What does each source say was FAKE or MISLEADING?
3. Based on the evidence, what is the most likely truth about the claim?

Think through this systematically and provide your analysis.

Respond in this exact JSON format:
{{
    "verdict": "true|false|uncertain",
    "verified": true|false,
    "message": "Your explanation here",
    "confidence": "high|medium|low",
    "reasoning": "Your step-by-step reasoning process"
}}
"""
        
        try:
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Try to parse JSON response
            if response_text.startswith('```json'):
                response_text = response_text.replace('```json', '').replace('```', '').strip()
            elif response_text.startswith('```'):
                response_text = response_text.replace('```', '').strip()
            
            analysis = json.loads(response_text)
            
            # Ensure required fields
            analysis.setdefault("verdict", "uncertain")
            analysis.setdefault("verified", False)
            analysis.setdefault("message", "Analysis completed")
            analysis.setdefault("confidence", "medium")
            analysis.setdefault("reasoning", "Analysis completed")
            
            # Add metadata
            analysis["relevant_results_count"] = len(results)
            analysis["analysis_method"] = "gemini"
            
            return analysis
            
        except json.JSONDecodeError as e:
            print(f"Failed to parse Gemini response as JSON: {e}")
            print(f"Raw response: {response_text}")
            return self._fallback_analysis(results)
        except Exception as e:
            print(f"Gemini analysis error: {e}")
            return self._fallback_analysis(results)
    
    async def _analyze_batch_with_gemini(self, search_results_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Use Gemini AI to analyze multiple fact-check results in a single batch call
        
        Args:
            search_results_list: List of dictionaries containing claim_data and search_results
            
        Returns:
            List of analysis results with verdict and message for each claim
        """
        try:
            # Prepare batch prompt
            batch_prompt = """
You are a fact-checking expert. Analyze the following claims against their provided fact-checking sources.

INSTRUCTIONS:
1. For each claim, determine if it's true, false, or uncertain
2. Provide clear reasoning based on the evidence
3. Assign confidence levels (high/medium/low)
4. Be consistent in your analysis approach

"""
            
            claims_text = ""
            for i, item in enumerate(search_results_list, 1):
                claim_data = item['claim_data']
                search_results = item['search_results']
                claim_text = claim_data.get('text_input', f'Claim {i}')
                
                claims_text += f"\n--- CLAIM {i} ---\n"
                claims_text += f"CLAIM TO VERIFY: \"{claim_text}\"\n\n"
                
                if search_results:
                    claims_text += "FACT-CHECKING SOURCES:\n"
                    for j, result in enumerate(search_results[:3], 1):  # Limit to top 3 results per claim
                        title = result.get("title", "")
                        snippet = result.get("snippet", "")
                        link = result.get("link", "")
                        claims_text += f"{j}. Title: {title}\n   Snippet: {snippet}\n   Link: {link}\n\n"
                else:
                    claims_text += "FACT-CHECKING SOURCES: No sources found\n\n"
            
            batch_prompt += claims_text
            batch_prompt += f"""

Respond with a JSON array containing exactly {len(search_results_list)} analysis objects in the same order as the claims above.

Each object should have this exact format:
{{
    "verdict": "true|false|uncertain",
    "verified": true|false,
    "message": "Your explanation here",
    "confidence": "high|medium|low",
    "reasoning": "Your step-by-step reasoning process"
}}

Example response format:
[
    {{
        "verdict": "false",
        "verified": false,
        "message": "This claim is false based on evidence...",
        "confidence": "high",
        "reasoning": "The sources show that..."
    }},
    {{
        "verdict": "true",
        "verified": true,
        "message": "This claim is accurate...",
        "confidence": "medium",
        "reasoning": "Multiple sources confirm..."
    }}
]
"""

            # Make single Gemini API call for all claims
            response = self.model.generate_content(batch_prompt)
            response_text = response.text.strip()
            
            # Clean up response
            if response_text.startswith('```json'):
                response_text = response_text.replace('```json', '').replace('```', '').strip()
            elif response_text.startswith('```'):
                response_text = response_text.replace('```', '').strip()
            
            # Parse JSON array
            batch_analysis = json.loads(response_text)
            
            # Validate and ensure we have the right number of results
            if not isinstance(batch_analysis, list):
                raise ValueError("Expected JSON array response")
            
            if len(batch_analysis) != len(search_results_list):
                print(f"Warning: Expected {len(search_results_list)} results, got {len(batch_analysis)}")
                # Pad or truncate as needed
                while len(batch_analysis) < len(search_results_list):
                    batch_analysis.append({
                        "verdict": "uncertain",
                        "verified": False,
                        "message": "Analysis incomplete due to batch processing error",
                        "confidence": "low",
                        "reasoning": "Insufficient analysis data"
                    })
                batch_analysis = batch_analysis[:len(search_results_list)]
            
            # Ensure all required fields and add metadata
            for i, analysis in enumerate(batch_analysis):
                analysis.setdefault("verdict", "uncertain")
                analysis.setdefault("verified", False)
                analysis.setdefault("message", "Analysis completed")
                analysis.setdefault("confidence", "medium")
                analysis.setdefault("reasoning", "Analysis completed")
                analysis["analysis_method"] = "gemini_batch"
                analysis["batch_position"] = i + 1
                analysis["batch_size"] = len(search_results_list)
            
            print(f"Batch Gemini analysis completed for {len(batch_analysis)} claims")
            return batch_analysis
            
        except json.JSONDecodeError as e:
            print(f"Failed to parse batch Gemini JSON response: {str(e)}")
            return self._fallback_batch_analysis(search_results_list)
        except Exception as e:
            print(f"Batch Gemini analysis error: {str(e)}")
            return self._fallback_batch_analysis(search_results_list)
    
    def _fallback_batch_analysis(self, search_results_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fallback analysis when batch Gemini processing fails"""
        fallback_results = []
        for item in search_results_list:
            search_results = item['search_results']
            if search_results:
                fallback_results.append(self._fallback_analysis(search_results))
            else:
                fallback_results.append({
                    "verified": False,
                    "verdict": "no_content",
                    "message": "No fact-checked information found for this claim",
                    "confidence": "low",
                    "reasoning": "No reliable sources found"
                })
        return fallback_results
    
    def _fallback_analysis(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Fallback analysis when Gemini fails
        
        Args:
            results: List of search results
            
        Returns:
            Basic analysis results
        """
        return {
            "verified": False,
            "verdict": "uncertain",
            "message": "Unable to determine claim accuracy from available sources. Found fact-checking articles but analysis failed.",
            "confidence": "low",
            "relevant_results_count": len(results),
            "analysis_method": "fallback"
        }
    
    def _extract_verdict_from_content(self, content: str) -> str:
        """
        Extract verdict from search result content
        
        Args:
            content: Combined title and snippet text
            
        Returns:
            Verdict string
        """
        content_lower = content.lower()
        
        # Look for verdict indicators
        if any(word in content_lower for word in ["false", "misleading", "incorrect", "debunked", "not true"]):
            return "false"
        elif any(word in content_lower for word in ["true", "accurate", "correct", "verified", "confirmed", "is true", "is correct"]):
            return "true"
        elif any(word in content_lower for word in ["partially", "mixed", "somewhat", "half", "unverified", "unproven", "uncertain", "disputed"]):
            return "uncertain"
        else:
            return "unknown"
    
    def _analyze_verdicts(self, verdicts: List[str]) -> Dict[str, Any]:
        """
        Analyze verdicts to determine overall result
        
        Args:
            verdicts: List of verdict strings
            
        Returns:
            Analysis of verdicts
        """
        if not verdicts:
            return {
                "verified": False,
                "verdict": "uncertain",
                "message": "No verdicts found"
            }
        
        true_count = verdicts.count("true")
        false_count = verdicts.count("false")
        uncertain_count = verdicts.count("uncertain")
        unknown_count = verdicts.count("unknown")
        
        total = len(verdicts)
        
        # Determine overall verdict
        if false_count > 0:
            overall_verdict = "false"
            verified = False
        elif true_count > 0 and false_count == 0:
            overall_verdict = "true"
            verified = True
        elif uncertain_count > 0:
            overall_verdict = "uncertain"
            verified = False
        else:
            overall_verdict = "unknown"
            verified = False
        
        return {
            "verified": verified,
            "verdict": overall_verdict,
            "true_count": true_count,
            "false_count": false_count,
            "uncertain_count": uncertain_count,
            "unknown_count": unknown_count,
            "total_verdicts": total
        }
    
    def _build_message(self, analysis: Dict[str, Any], results: List[Dict[str, Any]]) -> str:
        """
        Build a human-readable message based on the analysis
        
        Args:
            analysis: Analysis results
            results: Relevant search results
            
        Returns:
            Formatted message
        """
        verdict = analysis["verdict"]
        total_verdicts = analysis["total_verdicts"]
        relevant_results_count = len(results)
        
        base_messages = {
            "true": "This claim appears to be TRUE based on fact-checking sources.",
            "false": "This claim appears to be FALSE based on fact-checking sources.",
            "uncertain": "This claim is UNCERTAIN - insufficient evidence to determine accuracy.",
            "unknown": "This claim needs further investigation - verdict unclear from available sources.",
            "no_content": "No fact-checked information found for this claim."
        }
        
        message = base_messages.get(verdict, "Unable to determine claim accuracy.")
        
        # Add details about sources
        if relevant_results_count > 0:
            message += f" Found {relevant_results_count} relevant fact-check(s) with {total_verdicts} total verdicts."
            
            # Add top sources
            top_sources = []
            for result in results[:3]:  # Show top 3 sources
                title = result.get("title", "Unknown")
                link = result.get("link", "")
                if title not in top_sources and link:
                    top_sources.append(f"{title}")
            
            if top_sources:
                message += f" Sources include: {', '.join(top_sources[:3])}."
        
        return message