import requests
import json
from typing import Dict, List, Optional, Any
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from .config import config
import re
import os


class TextFactChecker:
    """Service for fact-checking textual claims using Google Custom Search API with fact-checking sites"""
    
    def __init__(self):
        self.api_key = config.GOOGLE_FACT_CHECK_API_KEY
        self.search_engine_id = config.GOOGLE_FACT_CHECK_CX
        self.base_url = "https://www.googleapis.com/customsearch/v1"
        
        # Configure Gemini for analysis
        genai.configure(api_key=config.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(config.GEMINI_MODEL)
        
        # Initialize media verifiers (lazy loaded)
        self._image_verifier = None
        self._video_verifier = None
        
        if not self.api_key:
            raise ValueError("Google Custom Search API key is required")
        if not self.search_engine_id:
            raise ValueError("Google Custom Search Engine ID (cx) is required")
    
    @property
    def image_verifier(self):
        """Lazy load image verifier"""
        if self._image_verifier is None:
            try:
                from image_verifier import ImageVerifier
                self._image_verifier = ImageVerifier(api_key=getattr(config, 'SERP_API_KEY', None))
                print("✅ Image verifier loaded successfully")
            except Exception as e:
                print(f"⚠️ Image verifier not available: {e}")
                self._image_verifier = False  # Mark as unavailable
        return self._image_verifier if self._image_verifier is not False else None
    
    @property
    def video_verifier(self):
        """Lazy load video verifier"""
        if self._video_verifier is None:
            try:
                from video_verifier import VideoVerifier
                self._video_verifier = VideoVerifier(api_key=getattr(config, 'SERP_API_KEY', None))
                print("✅ Video verifier loaded successfully")
            except Exception as e:
                print(f"⚠️ Video verifier not available: {e}")
                self._video_verifier = False  # Mark as unavailable
        return self._video_verifier if self._video_verifier is not False else None
    
    def _detect_media_in_claim(self, claim_context: str, claim_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Detect if the claim involves media (images or videos)
        
        Args:
            claim_context: The context text of the claim
            claim_data: Additional claim data that might contain media URLs
            
        Returns:
            Dictionary with media detection results
        """
        media_info = {
            'has_media': False,
            'media_type': None,
            'media_urls': [],
            'media_description': None
        }
        
        # Check for media URLs in claim data
        if claim_data:
            # Check for direct media URLs
            for key in ['url', 'link', 'media_url', 'image_url', 'video_url', 'post_url']:
                if key in claim_data and claim_data[key]:
                    url = claim_data[key]
                    if self._is_image_url(url):
                        media_info['has_media'] = True
                        media_info['media_type'] = 'image'
                        media_info['media_urls'].append(url)
                    elif self._is_video_url(url):
                        media_info['has_media'] = True
                        media_info['media_type'] = 'video'
                        media_info['media_urls'].append(url)
            
            # Check for content_source indicating media
            if 'content_source' in claim_data:
                source = claim_data['content_source']
                if 'scraped' in source or 'image' in source.lower():
                    media_info['has_media'] = True
                    if not media_info['media_type']:
                        media_info['media_type'] = 'image'
        
        # Check text content for media indicators
        media_keywords = [
            'image', 'photo', 'picture', 'video', 'footage', 'clip',
            'screenshot', 'viral video', 'viral image', 'shows video',
            'shows image', 'in the image', 'in the video', 'this video',
            'this image', 'frame', 'visual'
        ]
        
        text_lower = claim_context.lower()
        if any(keyword in text_lower for keyword in media_keywords):
            media_info['has_media'] = True
            if 'video' in text_lower or 'footage' in text_lower or 'clip' in text_lower:
                if not media_info['media_type']:
                    media_info['media_type'] = 'video'
                media_info['media_description'] = 'Text mentions video content'
            elif 'image' in text_lower or 'photo' in text_lower or 'picture' in text_lower:
                if not media_info['media_type']:
                    media_info['media_type'] = 'image'
                media_info['media_description'] = 'Text mentions image content'
        
        # Extract URLs from text
        url_pattern = r'https?://[^\s<>"]+|www\.[^\s<>"]+'
        found_urls = re.findall(url_pattern, claim_context)
        for url in found_urls:
            if self._is_image_url(url):
                media_info['has_media'] = True
                media_info['media_type'] = 'image'
                media_info['media_urls'].append(url)
            elif self._is_video_url(url):
                media_info['has_media'] = True
                media_info['media_type'] = 'video'
                media_info['media_urls'].append(url)
        
        return media_info
    
    def _is_image_url(self, url: str) -> bool:
        """Check if URL points to an image"""
        if not url:
            return False
        url_lower = url.lower()
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.svg']
        image_domains = ['i.redd.it', 'imgur.com', 'i.imgur.com', 'pbs.twimg.com']
        
        return (any(url_lower.endswith(ext) for ext in image_extensions) or
                any(domain in url_lower for domain in image_domains))
    
    def _is_video_url(self, url: str) -> bool:
        """Check if URL points to a video"""
        if not url:
            return False
        url_lower = url.lower()
        video_extensions = ['.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v']
        video_domains = ['youtube.com', 'youtu.be', 'vimeo.com', 'tiktok.com',
                        'instagram.com', 'facebook.com', 'twitter.com', 'x.com']
        
        return (any(url_lower.endswith(ext) for ext in video_extensions) or
                any(domain in url_lower for domain in video_domains))
    
    async def _verify_media_content(self, media_info: Dict[str, Any], claim_context: str, claim_date: str) -> Optional[Dict[str, Any]]:
        """
        Run media verification (image or video) as additional context
        
        Args:
            media_info: Media detection information
            claim_context: The claim context
            claim_date: The claim date
            
        Returns:
            Media verification results or None if no media or verifier unavailable
        """
        if not media_info.get('has_media'):
            return None
        
        try:
            media_type = media_info.get('media_type')
            media_urls = media_info.get('media_urls', [])
            
            if media_type == 'image' and self.image_verifier:
                print(f"🔍 Running image verification for additional context...")
                # Use first image URL if available
                image_url = media_urls[0] if media_urls else None
                
                result = await self.image_verifier.verify(
                    image_url=image_url,
                    claim_context=claim_context,
                    claim_date=claim_date
                )
                
                return {
                    'media_type': 'image',
                    'verification_result': result,
                    'analysis_summary': result.get('summary') or result.get('message', 'Image analysis completed'),
                    'verdict': result.get('verdict', 'uncertain'),
                    'confidence': result.get('confidence', 'medium')
                }
            
            elif media_type == 'video' and self.video_verifier:
                print(f"🔍 Running video verification for additional context...")
                # Use first video URL if available
                video_url = media_urls[0] if media_urls else None
                
                result = await self.video_verifier.verify(
                    video_url=video_url,
                    claim_context=claim_context,
                    claim_date=claim_date
                )
                
                return {
                    'media_type': 'video',
                    'verification_result': result,
                    'analysis_summary': result.get('message', 'Video analysis completed'),
                    'verdict': result.get('verified', False),
                    'details': result.get('details', {})
                }
            
            else:
                print(f"⚠️ Media detected but verifier not available for type: {media_type}")
                return None
                
        except Exception as e:
            print(f"⚠️ Media verification failed: {e}")
            return None
    
    async def verify(self, text_input: str, claim_context: str = "Unknown context", claim_date: str = "Unknown date", claim_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Verify a textual claim using Google Custom Search API with fact-checking sites
        Enhanced with optional media verification for additional context
        
        Args:
            text_input: The text claim to verify
            claim_context: Context about the claim
            claim_date: Date when the claim was made
            claim_data: Additional claim data (may contain media URLs)
            
        Returns:
            Dictionary containing verification results with optional media analysis
        """
        try:
            print(f"Starting verification for: {text_input}")
            
            # Step 1: Detect if claim involves media
            media_info = self._detect_media_in_claim(claim_context, claim_data)
            media_analysis = None
            
            # Step 2: Run media verification if media detected (as additional context)
            if media_info.get('has_media'):
                print(f"📸 Media detected: {media_info['media_type']} - Running additional media verification...")
                media_analysis = await self._verify_media_content(media_info, claim_context, claim_date)
            
            # Step 3: Search for fact-checked claims related to the input text
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
            
            # Step 4: Analyze the search results (with media analysis as additional context)
            analysis = self._analyze_results(search_results, text_input, media_analysis=media_analysis)
            
            # Extract source links for clean output
            source_links = [result.get("link", "") for result in search_results[:5] if result.get("link")]
            source_titles = [result.get("title", "") for result in search_results[:5] if result.get("title")]
            
            # Prepare final result
            result = {
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
            
            # Include media analysis if available
            if media_analysis:
                result["media_analysis"] = {
                    "type": media_analysis.get('media_type'),
                    "summary": media_analysis.get('analysis_summary'),
                    "verdict": media_analysis.get('verdict'),
                    "confidence": media_analysis.get('confidence')
                }
                print(f"✅ Media analysis included: {media_analysis.get('media_type')} verification completed")
            
            return result
            
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
            claims_batch: List of claim dictionaries with keys: text_input, claim_context, claim_date, original_content
            
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
                
                # Detect media in claim
                claim_context = claim_data.get('claim_context', '')
                original_content = claim_data.get('original_content', {})
                media_info = self._detect_media_in_claim(claim_context, original_content)
                
                # Run media verification if detected
                media_analysis = None
                if media_info.get('has_media'):
                    print(f"📸 Media detected in claim {i+1}: {media_info['media_type']} - Running verification...")
                    media_analysis = await self._verify_media_content(
                        media_info, 
                        claim_context, 
                        claim_data.get('claim_date', 'Unknown date')
                    )
                
                search_results = await self._search_claims(text_input)
                search_results_list.append({
                    'claim_data': claim_data,
                    'search_results': search_results,
                    'media_analysis': media_analysis
                })
            
            # Batch analyze all claims with Gemini in a single call
            batch_analysis = await self._analyze_batch_with_gemini(search_results_list)
            
            # Format final results
            verification_results = []
            for i, (claim_item, analysis) in enumerate(zip(search_results_list, batch_analysis)):
                claim_data = claim_item['claim_data']
                search_results = claim_item['search_results']
                media_analysis = claim_item.get('media_analysis')
                
                if not search_results:
                    result = {
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
                    }
                    if media_analysis:
                        result['media_analysis'] = media_analysis
                    verification_results.append(result)
                    continue
                
                # Extract source links for clean output
                source_links = [result.get("link", "") for result in search_results[:5] if result.get("link")]
                source_titles = [result.get("title", "") for result in search_results[:5] if result.get("title")]
                
                result = {
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
                }
                
                # Add media analysis if available
                if media_analysis:
                    result['media_analysis'] = media_analysis
                    
                verification_results.append(result)
            
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
    
    def _analyze_results(self, results: List[Dict[str, Any]], original_text: str, media_analysis: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze the search results using Gemini AI to determine overall verdict
        Enhanced with optional media analysis as additional context
        
        Args:
            results: List of search results from the API
            original_text: The original text being verified
            media_analysis: Optional media verification results (image/video analysis)
            
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
        
        # Use Gemini to analyze the results (with media analysis as additional context)
        try:
            analysis = self._analyze_with_gemini(original_text, relevant_results, media_analysis=media_analysis)
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
    
    def _analyze_with_gemini(self, original_text: str, results: List[Dict[str, Any]], media_analysis: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Use Gemini AI to analyze fact-check results and determine verdict
        Enhanced with optional media analysis as additional context
        
        Args:
            original_text: The original claim being verified
            results: List of relevant search results
            media_analysis: Optional media verification results for additional context
            
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
        
        # Add media analysis section if available
        media_context = ""
        if media_analysis:
            media_type = media_analysis.get('media_type', 'unknown')
            media_summary = media_analysis.get('analysis_summary', 'No summary available')
            media_verdict = media_analysis.get('verdict', 'uncertain')
            media_confidence = media_analysis.get('confidence', 'unknown')
            
            media_context = f"""

ADDITIONAL CONTEXT - {media_type.upper()} VERIFICATION:
The claim involves {media_type} content. An independent {media_type} verification analysis was performed:
- Verdict: {media_verdict}
- Confidence: {media_confidence}
- Analysis: {media_summary}

This {media_type} analysis provides additional context but should be considered alongside the fact-checking sources below.
"""
        
        prompt = f"""
You are a fact-checking expert. Analyze the following claim against the provided fact-checking sources and any additional context.

CLAIM TO VERIFY: "{original_text}"{media_context}

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
                media_analysis = item.get('media_analysis')
                claim_text = claim_data.get('text_input', f'Claim {i}')
                
                claims_text += f"\n--- CLAIM {i} ---\n"
                claims_text += f"CLAIM TO VERIFY: \"{claim_text}\"\n\n"
                
                # Add media analysis as additional context
                if media_analysis:
                    claims_text += "ADDITIONAL CONTEXT - MEDIA ANALYSIS:\n"
                    claims_text += f"Media Type: {media_analysis.get('media_type', 'Unknown')}\n"
                    claims_text += f"Summary: {media_analysis.get('summary', 'No summary available')}\n\n"
                
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