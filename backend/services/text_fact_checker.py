import requests
import json
from typing import Dict, List, Optional, Any
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from config import config


class TextFactChecker:
    """Service for fact-checking textual claims using Google Custom Search API with fact-checking sites"""
    
    def __init__(self):
        self.api_key = config.GOOGLE_API_KEY
        self.search_engine_id = config.GOOGLE_FACT_CHECK_CX
        self.base_url = "https://www.googleapis.com/customsearch/v1"
        
        # Configure Gemini for analysis
        if not config.GEMINI_API_KEY:
            print("⚠️ WARNING: GEMINI_API_KEY not set. Gemini features will not work.")
        else:
            try:
                genai.configure(api_key=config.GEMINI_API_KEY)
                self.model = genai.GenerativeModel(config.GEMINI_MODEL)
                print(f"✅ Gemini configured with model: {config.GEMINI_MODEL}")
            except Exception as e:
                print(f"❌ Failed to configure Gemini: {e}")
                raise
        
        if not self.api_key:
            raise ValueError("Google Custom Search API key is required")
        if not self.search_engine_id:
            raise ValueError("Google Custom Search Engine ID (cx) is required")
    
    async def verify(self, text_input: str, claim_context: str = "Unknown context", claim_date: str = "Unknown date") -> Dict[str, Any]:
        """
        Verify a textual claim using a three-phase approach:
        1. Immediate Gemini read-through for a quick, reference-free baseline
        2. Curated SERP (fact-check) harvesting with structured analysis
        3. A final Gemini synthesis that reasons over BOTH the baseline and SERP data
        
        Args:
            text_input: The text claim to verify
            claim_context: Context about the claim
            claim_date: Date when the claim was made
            
        Returns:
            Dictionary containing verification results
        """
        try:
            print(f"🔍 DEBUG: TextFactChecker.verify called")
            print(f"🔍 DEBUG: text_input = {text_input}")
            print(f"🔍 DEBUG: claim_context = {claim_context}")
            print(f"🔍 DEBUG: claim_date = {claim_date}")
            print(f"Starting verification for: {text_input}")
            
            # STEP 0: quick general-knowledge pass (baseline)
            preliminary_analysis = await self._verify_with_general_knowledge(
                text_input, claim_context, claim_date
            )
            print(f"🔍 DEBUG: preliminary_analysis = {preliminary_analysis}")
            
            # STEP 1: Search for fact-checked claims in curated sources
            search_results = await self._search_claims(text_input)
            print(f"🔍 DEBUG: search_results = {search_results}")
            
            curated_analysis = None
            if search_results:
                # Analyze the search results with Gemini
                curated_analysis = self._analyze_results(search_results, text_input)
            
            final_response = self._synthesize_final_response(
                text_input=text_input,
                claim_context=claim_context,
                claim_date=claim_date,
                preliminary_analysis=preliminary_analysis,
                curated_analysis=curated_analysis,
                search_results=search_results or []
            )
            
            if final_response:
                return final_response
            
            # Fallback ladder: curated -> preliminary -> default error
            if curated_analysis:
                return self._build_simple_response(
                    curated_analysis,
                    text_input,
                    claim_context,
                    claim_date,
                    search_results or [],
                    method_label="curated_sources_only",
                    extra_details={
                        "preliminary_analysis": preliminary_analysis,
                        "curated_analysis": curated_analysis,
                    },
                )
            
            if preliminary_analysis:
                return self._build_simple_response(
                    preliminary_analysis,
                    text_input,
                    claim_context,
                    claim_date,
                    search_results or [],
                    method_label="general_knowledge_only",
                    extra_details={"preliminary_analysis": preliminary_analysis},
                )
            
            return {
                "verified": False,
                "verdict": "error",
                "message": "Unable to generate a verification response.",
                "details": {
                    "claim_text": text_input,
                    "claim_context": claim_context,
                    "claim_date": claim_date,
                    "fact_checks": search_results or [],
                    "analysis": {},
                    "verification_method": "unavailable",
                },
            }
            
        except Exception as e:
            print(f"❌ Error in verify: {e}")
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

IMPORTANT INSTRUCTIONS FOR YOUR RESPONSE:
- When referring to sources in your message, DO NOT use specific numbers like "Source 1", "Source 3", or "Sources 2, 4, and 5"
- Instead, use generic references like "the sources", "multiple sources", "one source", "several sources"
- Example: Instead of "Sources 3, 4, and 5 confirm..." say "Multiple sources confirm..." or "The sources confirm..."

Respond in this exact JSON format:
{{
    "verdict": "true|false|mixed|uncertain",
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
    
    def _format_source_summary(self, results: List[Dict[str, Any]]) -> str:
        """Create a short, human readable summary of the surfaced sources."""
        if not results:
            return "No vetted sources surfaced yet."

        highlights = []
        for result in results[:3]:
            title = result.get("title") or "Unknown source"
            outlet = result.get("displayLink")
            summary = title
            if outlet:
                summary += f" ({outlet})"
            highlights.append(summary)

        return "Sources surfaced: " + "; ".join(highlights)

    def _fallback_analysis(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Fallback analysis when Gemini fails
        
        Args:
            results: List of search results
            
        Returns:
            Basic analysis results
        """
        summary = self._format_source_summary(results)

        return {
            "verified": False,
            "verdict": "uncertain",
            "message": f"Could not verify this claim yet. {summary}",
            "confidence": "low",
            "relevant_results_count": len(results),
            "analysis_method": "fallback"
        }
    
    async def _verify_with_general_knowledge(self, text_input: str, claim_context: str, claim_date: str) -> Dict[str, Any]:
        """
        Verify a claim using Gemini's general knowledge base directly (no curated sources)
        This is used as a fallback when curated sources don't have enough information
        
        Args:
            text_input: The text claim to verify
            claim_context: Context about the claim
            claim_date: Date when the claim was made
            
        Returns:
            Analysis results with verdict and message
        """
        from datetime import datetime
        current_date = datetime.now().strftime("%B %d, %Y")
        
        prompt = f"""
You are a fact-checking expert AI with access to current information as of {current_date}.

CLAIM TO VERIFY: "{text_input}"
CONTEXT: {claim_context if claim_context != "Unknown context" else "No additional context provided"}
CLAIM DATE: {claim_date if claim_date != "Unknown date" else "Unknown"}

Your task is to verify this claim using your knowledge base. Since this is a direct factual question that may not be covered by news articles:

1. **Use your most recent training data** to answer the question directly
2. If this is about current events, political positions, or time-sensitive facts, be especially careful to provide the MOST CURRENT information
3. If you're uncertain about recent changes, acknowledge that
4. Always answer based on the most recent information you have

Provide a clear, direct answer. Think step-by-step:
- What does the claim assert?
- Based on your knowledge (as of your training cutoff and any recent data you have), is this true or false?
- If it's a time-sensitive claim, what is the current status?

Respond in this exact JSON format:
{{
    "verdict": "true|false|mixed|uncertain",
    "verified": true|false,
    "message": "Your clear, direct answer explaining whether the claim is true or false and why",
    "confidence": "high|medium|low",
    "reasoning": "Your step-by-step reasoning process",
    "knowledge_cutoff_note": "Optional note if the answer might be outdated or if recent changes are possible"
}}

IMPORTANT: For current events or political positions, provide the MOST RECENT information you have access to.
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
            analysis.setdefault("message", "Analysis completed using general knowledge")
            analysis.setdefault("confidence", "medium")
            analysis.setdefault("reasoning", "Direct verification using AI knowledge base")
            
            # Add metadata
            analysis["analysis_method"] = "general_knowledge"
            analysis["verification_date"] = current_date
            
            print(f"✅ General knowledge verification result: {analysis['verdict']}")
            return analysis
            
        except json.JSONDecodeError as e:
            print(f"Failed to parse Gemini general knowledge response as JSON: {e}")
            print(f"Raw response: {response_text[:500]}")
            # Try to extract plain text answer
            return {
                "verified": False,
                "verdict": "uncertain",
                "message": response_text if response_text else "Unable to verify using general knowledge",
                "confidence": "low",
                "analysis_method": "general_knowledge",
                "error": "JSON parsing failed, used plain text response"
            }
        except Exception as e:
            print(f"General knowledge verification error: {e}")
            return {
                "verified": False,
                "verdict": "error",
                "message": f"Error during general knowledge verification: {str(e)}",
                "confidence": "low",
                "analysis_method": "general_knowledge"
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
        elif any(word in content_lower for word in ["partially", "mixed", "somewhat", "half"]):
            return "mixed"
        elif any(word in content_lower for word in ["unverified", "unproven", "uncertain", "disputed"]):
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
        mixed_count = verdicts.count("mixed")
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
        elif mixed_count > 0:
            overall_verdict = "mixed"
            verified = False
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
            "mixed_count": mixed_count,
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
            "mixed": "This claim has MIXED evidence - some parts are true, others are false.",
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

    def _synthesize_final_response(
        self,
        text_input: str,
        claim_context: str,
        claim_date: str,
        preliminary_analysis: Optional[Dict[str, Any]],
        curated_analysis: Optional[Dict[str, Any]],
        search_results: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """
        Ask Gemini to reconcile preliminary + curated evidence into a single user-facing verdict.
        """
        try:
            source_briefs = []
            for item in search_results[:5]:
                source_briefs.append(
                    {
                        "title": item.get("title"),
                        "snippet": item.get("snippet"),
                        "outlet": item.get("displayLink"),
                        "link": item.get("link"),
                    }
                )

            prompt = f"""
You are an AI fact-checking editor. Combine the baseline assessment and curated sources to produce the final answer.

CLAIM: "{text_input}"
CONTEXT: {claim_context}
CLAIM DATE: {claim_date}

BASELINE ANALYSIS (Gemini quick look):
{json.dumps(preliminary_analysis or {}, indent=2, ensure_ascii=False)}

CURATED FACT-CHECK ANALYSIS:
{json.dumps(curated_analysis or {}, indent=2, ensure_ascii=False)}

FACT-CHECK SOURCES:
{json.dumps(source_briefs, indent=2, ensure_ascii=False)}

INSTRUCTIONS:
- Make a reasoned decision (true/false/mixed/uncertain) based on the above.
- If evidence is thin, keep the tone cautious and say it is unverified/uncertain but mention what was found.
- Refer to sources generically (e.g., "one BBC article", "multiple outlets") — never number them.
- Provide clear, actionable messaging for the end user.

Respond ONLY in this JSON format:
{{
  "verdict": "true|false|mixed|uncertain",
  "verified": true|false,
  "message": "Concise user-facing summary referencing evidence in plain language",
  "confidence": "high|medium|low",
  "reasoning": "Brief reasoning trail you followed",
  "tone": "confident|balanced|cautious"
}}
"""
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()

            if response_text.startswith("```json"):
                response_text = response_text.replace("```json", "").replace("```", "").strip()
            elif response_text.startswith("```"):
                response_text = response_text.replace("```", "").strip()

            final_analysis = json.loads(response_text)
            final_analysis.setdefault("verdict", "uncertain")
            final_analysis.setdefault("verified", False)
            final_analysis.setdefault("message", "Unable to synthesize final verdict.")
            final_analysis.setdefault("confidence", "low")
            final_analysis.setdefault("reasoning", "")
            final_analysis.setdefault("tone", "cautious")
            final_analysis["analysis_method"] = "hybrid_synthesis"

            return self._build_simple_response(
                final_analysis,
                text_input,
                claim_context,
                claim_date,
                search_results,
                method_label="hybrid_synthesis",
                extra_details={
                    "preliminary_analysis": preliminary_analysis,
                    "curated_analysis": curated_analysis,
                    "source_highlights": source_briefs,
                },
            )
        except Exception as e:
            print(f"Hybrid synthesis error: {e}")
            return None

    def _build_simple_response(
        self,
        analysis: Dict[str, Any],
        text_input: str,
        claim_context: str,
        claim_date: str,
        search_results: List[Dict[str, Any]],
        method_label: str,
        extra_details: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        details = {
            "claim_text": text_input,
            "claim_context": claim_context,
            "claim_date": claim_date,
            "fact_checks": search_results,
            "analysis": analysis,
            "verification_method": method_label,
        }
        if extra_details:
            details.update(extra_details)

        return {
            "verified": analysis.get("verified", False),
            "verdict": analysis.get("verdict", "uncertain"),
            "message": analysis.get("message", "No message produced."),
            "details": details,
        }
