"""
MongoDB Service for Backend
Handles MongoDB operations for debunk posts
"""

import os
import logging
from typing import List, Dict, Any, Optional
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from dotenv import load_dotenv

load_dotenv()

# Setup logging
logger = logging.getLogger(__name__)

class MongoDBService:
    """MongoDB service for backend operations"""
    
    def __init__(self, connection_string: Optional[str] = None):
        """Initialize MongoDB connection
        
        Args:
            connection_string: MongoDB connection string. If None, uses MONGO_CONNECTION_STRING env var
        """
        self.connection_string = connection_string or os.getenv('MONGO_CONNECTION_STRING')
        
        if not self.connection_string:
            raise ValueError("MongoDB connection string is required. Set MONGO_CONNECTION_STRING environment variable.")
        
        self.client = None
        self.db = None
        self.collection = None
        self.chat_sessions = None
        self.chat_messages = None
        
        self._connect()
    
    def _connect(self):
        """Establish MongoDB connection"""
        try:
            self.client = MongoClient(self.connection_string)
            # Test connection
            self.client.admin.command('ping')
            
            # Use 'aegis' database
            self.db = self.client["aegis"]
            self.collection = self.db["debunk_posts"]

            # Additional collections used by other features
            self.chat_sessions = self.db["chat_sessions"]
            self.chat_messages = self.db["chat_messages"]
            
            logger.info("✅ Successfully connected to MongoDB")
            
        except ConnectionFailure as e:
            logger.error(f"❌ Failed to connect to MongoDB: {e}")
            raise
    
    def get_recent_posts(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent debunk posts from MongoDB
        
        Args:
            limit: Maximum number of posts to return
            
        Returns:
            List of recent debunk posts
        """
        try:
            logger.info(f"🔍 DEBUG: Starting get_recent_posts with limit={limit}")
            logger.info(f"🔍 DEBUG: Collection name: {self.collection.name}")
            logger.info(f"🔍 DEBUG: Database name: {self.db.name}")
            
            # Check if collection exists and has documents
            total_count = self.collection.count_documents({})
            logger.info(f"🔍 DEBUG: Total documents in collection: {total_count}")
            
            if total_count == 0:
                logger.warning("⚠️ DEBUG: Collection is empty!")
                return []
            
            # Get sample document to check structure
            sample_doc = self.collection.find_one()
            if sample_doc:
                logger.info(f"🔍 DEBUG: Sample document keys: {list(sample_doc.keys())}")
                logger.info(f"🔍 DEBUG: Sample document _id: {sample_doc.get('_id')}")
                logger.info(f"🔍 DEBUG: Sample document stored_at: {sample_doc.get('stored_at')}")
            else:
                logger.warning("⚠️ DEBUG: No sample document found!")
            
            posts = list(self.collection
                        .find()
                        .sort("stored_at", -1)
                        .limit(limit))
            
            logger.info(f"🔍 DEBUG: Raw query returned {len(posts)} posts")
            
            # Convert ObjectId to string for JSON serialization
            for i, post in enumerate(posts):
                if '_id' in post:
                    post['_id'] = str(post['_id'])
                logger.info(f"🔍 DEBUG: Post {i+1} keys: {list(post.keys())}")
                logger.info(f"🔍 DEBUG: Post {i+1} stored_at: {post.get('stored_at')}")
            
            logger.info(f"📋 Retrieved {len(posts)} recent debunk posts")
            return posts
            
        except Exception as e:
            logger.error(f"❌ Failed to get recent posts: {e}")
            logger.error(f"🔍 DEBUG: Exception type: {type(e).__name__}")
            logger.error(f"🔍 DEBUG: Exception details: {str(e)}")
            return []

    def search_similar_rumours(self, query: str, similarity_threshold: float = 0.6, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for rumours similar to the query text using TF-IDF similarity
        
        Args:
            query: Search query text
            similarity_threshold: Minimum similarity score (0.0 to 1.0)
            limit: Maximum number of results to return
            
        Returns:
            List of similar rumours with similarity scores
        """
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            import re
            
            if not query or not query.strip():
                logger.warning("⚠️ Empty query provided")
                return []
            
            logger.info(f"🔍 Searching for rumours similar to: {query[:50]}...")
            
            # Get all rumours from database
            all_posts = list(self.collection.find())
            
            if not all_posts:
                logger.warning("⚠️ No rumours found in database")
                return []
            
            # Extract claim text from each post
            claims = []
            posts_data = []
            
            for post in all_posts:
                # Extract claim text - try multiple fields
                claim_text = (
                    post.get('claim') or 
                    post.get('summary') or 
                    ""
                )
                
                # Handle nested claim structure
                if isinstance(claim_text, dict):
                    claim_text = claim_text.get('text') or claim_text.get('claim_text') or ""
                
                if claim_text and claim_text.strip():
                    claims.append(claim_text)
                    posts_data.append(post)
            
            if not claims:
                logger.warning("⚠️ No claims found in posts")
                return []
            
            # Preprocess query
            def preprocess_text(text: str) -> str:
                text = text.lower()
                text = re.sub(r'[^\w\s]', ' ', text)
                text = ' '.join(text.split())
                return text
            
            query_processed = preprocess_text(query)
            
            # Calculate TF-IDF similarity
            try:
                vectorizer = TfidfVectorizer(
                    stop_words='english',
                    ngram_range=(1, 2),
                    max_features=500,
                    lowercase=True
                )
                
                # Combine query and claims for vectorization
                all_texts = [query_processed] + [preprocess_text(c) for c in claims]
                tfidf_matrix = vectorizer.fit_transform(all_texts)
                
                # Calculate similarity between query and each claim
                query_vector = tfidf_matrix[0:1]
                claims_matrix = tfidf_matrix[1:]
                
                similarities = cosine_similarity(query_vector, claims_matrix)[0]
                
            except Exception as e:
                logger.error(f"❌ TF-IDF calculation failed: {e}")
                # Fallback to simple word overlap
                similarities = []
                query_words = set(query_processed.split())
                for claim in claims:
                    claim_words = set(preprocess_text(claim).split())
                    if not query_words or not claim_words:
                        similarities.append(0.0)
                    else:
                        intersection = query_words.intersection(claim_words)
                        union = query_words.union(claim_words)
                        similarities.append(len(intersection) / len(union) if union else 0.0)
            
            # Filter by threshold and sort by similarity
            results = []
            for i, (post, similarity) in enumerate(zip(posts_data, similarities)):
                if similarity >= similarity_threshold:
                    # Convert ObjectId to string
                    if '_id' in post:
                        post['_id'] = str(post['_id'])
                    
                    result = {
                        **post,
                        'similarity_score': float(similarity)
                    }
                    results.append(result)
            
            # Sort by similarity score (descending) and limit
            results.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
            results = results[:limit]
            
            logger.info(f"✅ Found {len(results)} similar rumours (threshold: {similarity_threshold})")
            return results
            
        except Exception as e:
            logger.error(f"❌ Failed to search similar rumours: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []

    # ---------- Chat sessions & messages ----------

    def get_chat_sessions(
        self,
        user_id: Optional[str] = None,
        anonymous_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Return chat sessions for a given user or anonymous visitor."""
        if self.chat_sessions is None:
            return []

        query: Dict[str, Any] = {}
        if user_id:
          query["user_id"] = user_id
        if anonymous_id and not user_id:
          # For anonymous visitors we only look at sessions that have not yet been
          # attached to a concrete user id.
          query["anonymous_id"] = anonymous_id
          query["user_id"] = None

        cursor = (
            self.chat_sessions.find(query)
            .sort("updated_at", -1)
            .limit(limit)
        )
        sessions: List[Dict[str, Any]] = []
        for doc in cursor:
            doc["session_id"] = str(doc.get("session_id") or doc.get("_id"))
            doc["_id"] = str(doc["_id"])
            sessions.append(doc)
        return sessions

    def migrate_anonymous_sessions(self, anonymous_id: str, user_id: str) -> int:
        """Attach existing anonymous sessions to a logged-in user.

        This keeps history when a visitor later signs in.
        """
        if self.chat_sessions is None or not anonymous_id or not user_id:
            return 0

        result = self.chat_sessions.update_many(
            {"anonymous_id": anonymous_id, "user_id": None},
            {"$set": {"user_id": user_id}},
        )
        return int(getattr(result, "modified_count", 0))

    def upsert_chat_session(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Create or update a chat session document.

        Expected keys in payload: session_id (optional), user_id, anonymous_id,
        title, last_verdict, last_summary.
        """
        if self.chat_sessions is None:
            raise RuntimeError("chat_sessions collection not initialised")

        from datetime import datetime

        session_id = payload.get("session_id")
        now = datetime.utcnow()

        base_updates: Dict[str, Any] = {
            "title": payload.get("title") or "New Chat",
            "user_id": payload.get("user_id"),
            "anonymous_id": payload.get("anonymous_id"),
            "last_verdict": payload.get("last_verdict"),
            "last_summary": payload.get("last_summary"),
            "updated_at": now,
        }

        if session_id:
            doc = self.chat_sessions.find_one_and_update(
                {"session_id": session_id},
                {"$set": base_updates},
                upsert=True,
                return_document=True,
            )
        else:
            doc_to_insert = {
                **base_updates,
                "session_id": payload.get("session_id") or os.urandom(12).hex(),
                "created_at": now,
            }
            inserted = self.chat_sessions.insert_one(doc_to_insert)
            doc = self.chat_sessions.find_one({"_id": inserted.inserted_id})

        doc["_id"] = str(doc["_id"])
        doc["session_id"] = str(doc.get("session_id"))
        return doc

    def append_chat_messages(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
        user_id: Optional[str] = None,
        anonymous_id: Optional[str] = None,
    ) -> int:
        """Append one or more messages to a given session."""
        if self.chat_messages is None:
            raise RuntimeError("chat_messages collection not initialised")

        from datetime import datetime

        docs = []
        for msg in messages:
            docs.append(
                {
                    "session_id": session_id,
                    "user_id": user_id,
                    "anonymous_id": anonymous_id,
                    "role": msg.get("role"),
                    "content": msg.get("content"),
                    "attachments": msg.get("attachments") or [],
                    "verdict": msg.get("verdict"),
                    "confidence": msg.get("confidence"),
                    "sources": msg.get("sources"),
                    "created_at": msg.get("created_at") or datetime.utcnow(),
                    "metadata": msg.get("metadata") or {},
                }
            )

        if not docs:
            return 0

        result = self.chat_messages.insert_many(docs)
        return len(getattr(result, "inserted_ids", []))

    def get_chat_messages(
        self, session_id: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Return messages for a particular session ordered by time."""
        if self.chat_messages is None:
            return []

        cursor = (
            self.chat_messages.find({"session_id": session_id})
            .sort("created_at", 1)
            .limit(limit)
        )
        docs: List[Dict[str, Any]] = []
        for doc in cursor:
            doc["_id"] = str(doc["_id"])
            docs.append(doc)
        return docs

    def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            logger.info("🔌 MongoDB connection closed")
