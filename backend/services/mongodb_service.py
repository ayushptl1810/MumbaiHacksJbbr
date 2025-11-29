"""
MongoDB Service for Backend
Handles MongoDB operations for debunk posts
"""

import os
import logging
from typing import List, Dict, Any, Optional
from pymongo import MongoClient, ReturnDocument
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
            self.subscriptions = self.db["subscriptions"]
            self.users = self.db["users"]
            self.weekly_posts = self.db["weekly_posts"]
            self.usage_limits = self.db["usage_limits"]
            
            logger.info("✅ Successfully connected to MongoDB")
            
        except ConnectionFailure as e:
            logger.error(f"❌ Failed to connect to MongoDB: {e}")
            raise

    # ---------- Usage limits (verification / chat) ----------

    def increment_usage_and_check_limits(
        self,
        key: str,
        feature: str,
        daily_limit: Optional[int],
        monthly_limit: Optional[int],
    ) -> Dict[str, Any]:
        """
        Increment usage counters for a given key/feature and return limit status.

        Args:
            key: Logical user key (user_id or anonymous_id)
            feature: Feature name, e.g. "verification" or "chat_message"
            daily_limit: Max allowed per day (None or <=0 means unlimited)
            monthly_limit: Max allowed per month (None or <=0 means unlimited)

        Returns:
            Dict with:
              - allowed: bool
              - tier_limits: {daily, monthly}
              - daily: {count, limit}
              - monthly: {count, limit}
        """
        from datetime import datetime

        if self.usage_limits is None:
            logger.warning("⚠️ usage_limits collection not initialised")
            return {
                "allowed": True,
                "tier_limits": {
                    "daily": daily_limit,
                    "monthly": monthly_limit,
                },
                "daily": {"count": 0, "limit": daily_limit},
                "monthly": {"count": 0, "limit": monthly_limit},
            }

        now = datetime.utcnow()
        date_str = now.strftime("%Y-%m-%d")
        month_str = now.strftime("%Y-%m")

        result: Dict[str, Any] = {
            "allowed": True,
            "tier_limits": {
                "daily": daily_limit,
                "monthly": monthly_limit,
            },
            "daily": {"count": 0, "limit": daily_limit},
            "monthly": {"count": 0, "limit": monthly_limit},
        }

        # Helper to increment and evaluate a single period
        def _inc_and_check(period_type: str, period_value: str, limit: Optional[int]):
            if not limit or limit <= 0:
                # Unlimited for this period
                result[period_type] = {"count": 0, "limit": limit}
                return

            doc = self.usage_limits.find_one_and_update(
                {
                    "key": key,
                    "feature": feature,
                    "period_type": period_type,
                    "period_value": period_value,
                },
                {
                    "$inc": {"count": 1},
                    "$set": {"last_at": now},
                    "$setOnInsert": {"first_at": now},
                },
                upsert=True,
                return_document=ReturnDocument.AFTER,
            )

            count = int(doc.get("count", 0))
            result[period_type] = {"count": count, "limit": limit}
            if count > limit:
                result["allowed"] = False

        _inc_and_check("daily", date_str, daily_limit)
        _inc_and_check("monthly", month_str, monthly_limit)

        return result

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

    # ---------- Subscription management ----------
    
    def upsert_subscription(self, subscription_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create or update a subscription document
        
        Expected keys in subscription_data:
        - user_id: User ID
        - razorpay_subscription_id: Razorpay subscription ID
        - razorpay_plan_id: Razorpay plan ID
        - plan_name: Plan name (e.g., "Pro")
        - status: Subscription status (e.g., "active", "cancelled", "expired")
        - amount: Subscription amount
        - currency: Currency code
        - current_start: Current billing cycle start
        - current_end: Current billing cycle end
        - next_billing_at: Next billing date
        - created_at: Subscription creation date
        - updated_at: Last update date
        """
        if self.subscriptions is None:
            raise RuntimeError("subscriptions collection not initialised")
        
        from datetime import datetime
        
        razorpay_subscription_id = subscription_data.get("razorpay_subscription_id")
        if not razorpay_subscription_id:
            raise ValueError("razorpay_subscription_id is required")
        
        now = datetime.utcnow()
        
        # Prepare update data
        update_data = {
            **subscription_data,
            "updated_at": now,
        }
        
        # Set created_at only if creating new subscription
        existing = self.subscriptions.find_one(
            {"razorpay_subscription_id": razorpay_subscription_id}
        )
        
        if not existing:
            update_data["created_at"] = subscription_data.get("created_at") or now
        
        # Upsert subscription
        result = self.subscriptions.find_one_and_update(
            {"razorpay_subscription_id": razorpay_subscription_id},
            {"$set": update_data},
            upsert=True,
            return_document=True
        )
        
        if result:
            result["_id"] = str(result["_id"])
            logger.info(f"✅ Upserted subscription: {razorpay_subscription_id}")
            
            # Update user's subscription tier if user_id is present
            user_id = subscription_data.get("user_id")
            status = subscription_data.get("status")
            plan_name = subscription_data.get("plan_name", "Free")
            
            if user_id:
                if status == "active":
                    success = self.update_user_subscription_tier(user_id, plan_name)
                    if success:
                        logger.info(f"✅ Updated user {user_id} subscription tier to {plan_name} via upsert_subscription")
                elif status in ["cancelled", "expired", "paused", "ended"]:
                    success = self.update_user_subscription_tier(user_id, "Free")
                    if success:
                        logger.info(f"✅ Updated user {user_id} subscription tier to Free (status: {status})")
        
        return result
    
    def get_user_subscription(
        self,
        user_id: str,
        status: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get user's active subscription
        
        Args:
            user_id: User ID
            status: Filter by status (e.g., "active"). If None, returns most recent
            
        Returns:
            Subscription document or None
        """
        if self.subscriptions is None:
            return None
        
        query = {"user_id": user_id}
        if status:
            query["status"] = status
        
        subscription = self.subscriptions.find_one(
            query,
            sort=[("created_at", -1)]
        )
        
        if subscription:
            subscription["_id"] = str(subscription["_id"])
        
        return subscription
    
    def update_subscription_status(
        self,
        razorpay_subscription_id: str,
        status: str,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Update subscription status from webhook events
        
        Args:
            razorpay_subscription_id: Razorpay subscription ID
            status: New status
            additional_data: Additional fields to update
            
        Returns:
            Updated subscription document or None
        """
        if self.subscriptions is None:
            return None
        
        from datetime import datetime
        
        update_data = {
            "status": status,
            "updated_at": datetime.utcnow()
        }
        
        if additional_data:
            update_data.update(additional_data)
        
        result = self.subscriptions.find_one_and_update(
            {"razorpay_subscription_id": razorpay_subscription_id},
            {"$set": update_data},
            return_document=True
        )
        
        if result:
            result["_id"] = str(result["_id"])
            logger.info(f"✅ Updated subscription status: {razorpay_subscription_id} -> {status}")
            
            # Update user's subscription tier
            user_id = result.get("user_id")
            if user_id:
                plan_name = result.get("plan_name", "Free")
                if status == "active":
                    self.update_user_subscription_tier(user_id, plan_name)
                elif status in ["cancelled", "expired", "paused"]:
                    self.update_user_subscription_tier(user_id, "Free")
        
        return result
    
    def get_subscription_by_razorpay_id(
        self,
        razorpay_subscription_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get subscription by Razorpay subscription ID
        
        Args:
            razorpay_subscription_id: Razorpay subscription ID
            
        Returns:
            Subscription document or None
        """
        if self.subscriptions is None:
            return None
        
        subscription = self.subscriptions.find_one(
            {"razorpay_subscription_id": razorpay_subscription_id}
        )
        
        if subscription:
            subscription["_id"] = str(subscription["_id"])
        
        return subscription
    
    def create_user(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new user in MongoDB
        
        Args:
            user_data: User data including email, password (hashed), domain_preferences, etc.
            
        Returns:
            Created user document
        """
        if self.users is None:
            raise RuntimeError("users collection not initialised")
        
        from datetime import datetime
        from bson import ObjectId
        
        # Check if user already exists
        existing = self.users.find_one({"email": user_data["email"]})
        if existing:
            raise ValueError("Email already registered")
        
        user_doc = {
            **user_data,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }
        
        result = self.users.insert_one(user_doc)
        user_doc["_id"] = str(result.inserted_id)
        user_doc["id"] = str(result.inserted_id)
        
        logger.info(f"✅ Created user: {user_data['email']}")
        return user_doc
    
    def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """
        Get user by email
        
        Args:
            email: User email
            
        Returns:
            User document or None
        """
        if self.users is None:
            return None
        
        user = self.users.find_one({"email": email})
        if user:
            user["_id"] = str(user["_id"])
            user["id"] = str(user["_id"])
        
        return user
    
    def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get user by ID
        
        Args:
            user_id: User ID
            
        Returns:
            User document or None
        """
        if self.users is None:
            return None
        
        from bson import ObjectId
        
        try:
            user = self.users.find_one({"_id": ObjectId(user_id)})
            if user:
                user["_id"] = str(user["_id"])
                user["id"] = str(user["_id"])
            return user
        except Exception as e:
            logger.error(f"Error getting user by ID: {e}")
            return None
    
    def update_user_subscription_tier(self, user_id: str, subscription_tier: str) -> bool:
        """
        Update user's subscription tier in user collection
        
        Args:
            user_id: User ID
            subscription_tier: Subscription tier (Free, Pro, Enterprise)
            
        Returns:
            True if updated successfully, False otherwise
        """
        if self.users is None:
            return False
        
        from datetime import datetime
        from bson import ObjectId
        
        try:
            result = self.users.update_one(
                {"_id": ObjectId(user_id)},
                {
                    "$set": {
                        "subscription_tier": subscription_tier,
                        "updated_at": datetime.utcnow()
                    }
                }
            )
            if result.modified_count > 0:
                logger.info(f"✅ Updated user {user_id} subscription tier to {subscription_tier}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error updating user subscription tier: {e}")
            return False
    
    # ---------- Educational Modules from weekly_posts ----------
    
    def get_educational_modules_list(self) -> List[Dict[str, Any]]:
        """
        Get list of all educational modules from weekly_posts collection
        Only returns posts that have educational_module field
        
        Returns:
            List of educational module summaries with unique misinformation types
        """
        try:
            if self.weekly_posts is None:
                logger.warning("⚠️ weekly_posts collection not initialized")
                return []
            
            # Find all posts with educational_module field
            posts_with_modules = list(
                self.weekly_posts.find(
                    {
                        "educational_module": {"$exists": True, "$ne": None}
                    },
                    {
                        "educational_module.misinformation_type": 1,
                        "educational_module.trending_score": 1,
                        "post_content.heading": 1,
                        "metadata.tags": 1,
                        "stored_at": 1,
                        "_id": 1
                    }
                )
                .sort("stored_at", -1)
            )
            
            # Group by misinformation_type and get the most recent one for each type
            type_map = {}
            for post in posts_with_modules:
                edu_module = post.get("educational_module", {})
                misinfo_type = edu_module.get("misinformation_type")
                
                if misinfo_type and misinfo_type not in type_map:
                    # Get trending score
                    trending_score = edu_module.get("trending_score", {}).get("$numberInt") if isinstance(edu_module.get("trending_score"), dict) else edu_module.get("trending_score", 0)
                    if isinstance(trending_score, str):
                        trending_score = int(trending_score)
                    
                    type_map[misinfo_type] = {
                        "id": misinfo_type.lower().replace(" ", "_").replace("-", "_"),
                        "title": misinfo_type,
                        "description": edu_module.get("technique_explanation", "")[:150] + "..." if edu_module.get("technique_explanation") else "Learn about this misinformation technique",
                        "trending_score": trending_score,
                        "tags": post.get("metadata", {}).get("tags", []),
                        "example_heading": post.get("post_content", {}).get("heading", "")[:100] if post.get("post_content", {}).get("heading") else "",
                        "post_id": str(post.get("_id", "")),
                        "related_patterns": edu_module.get("related_patterns", []),
                        "estimated_time": "15-20 minutes"
                    }
            
            # Convert to list and sort by trending score (descending)
            modules_list = list(type_map.values())
            modules_list.sort(key=lambda x: x.get("trending_score", 0), reverse=True)
            
            logger.info(f"✅ Retrieved {len(modules_list)} unique educational modules from weekly_posts")
            return modules_list
            
        except Exception as e:
            logger.error(f"❌ Failed to get educational modules list: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def get_educational_module_by_id(self, module_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific educational module by ID (misinformation_type)
        
        Args:
            module_id: The module ID (misinformation_type converted to ID format)
            
        Returns:
            Educational module document or None
        """
        try:
            if self.weekly_posts is None:
                logger.warning("⚠️ weekly_posts collection not initialized")
                return None
            
            # Get all posts with educational modules first
            all_posts = list(self.weekly_posts.find({
                "educational_module": {"$exists": True, "$ne": None}
            }))
            
            # Find matching post by converting module_id to various formats
            post = None
            for candidate_post in all_posts:
                edu_module = candidate_post.get("educational_module", {})
                misinfo_type = edu_module.get("misinformation_type", "")
                
                # Clean both for comparison - normalize spaces, dashes, special chars
                misinfo_id = misinfo_type.lower().replace(" ", "_").replace("-", "_").replace("'", "").replace('"', "").replace(",", "").replace(".", "").strip()
                module_id_clean = module_id.lower().replace(" ", "_").replace("-", "_").replace("'", "").replace('"', "").replace(",", "").replace(".", "").strip()
                
                if misinfo_id == module_id_clean:
                    post = candidate_post
                    break
            
            # If not found, try to find by partial match
            if not post:
                for candidate_post in all_posts:
                    edu_module = candidate_post.get("educational_module", {})
                    misinfo_type = edu_module.get("misinformation_type", "")
                    if module_id.lower() in misinfo_type.lower() or misinfo_type.lower() in module_id.lower():
                        post = candidate_post
                        break
            
            if not post:
                logger.warning(f"⚠️ No educational module found for ID: {module_id}")
                return None
            
            # Extract and format the educational module data
            edu_module = post.get("educational_module", {})
            
            # Handle MongoDB extended JSON format (numbers as objects)
            def clean_value(val):
                if isinstance(val, dict) and "$numberInt" in val:
                    return int(val["$numberInt"])
                if isinstance(val, dict) and "$numberLong" in val:
                    return int(val["$numberLong"])
                if isinstance(val, list):
                    return [clean_value(v) for v in val]
                if isinstance(val, dict):
                    return {k: clean_value(v) for k, v in val.items()}
                return val
            
            edu_module_cleaned = clean_value(edu_module)
            
            # Build response with post context
            result = {
                "id": module_id,
                "title": edu_module_cleaned.get("misinformation_type", "Educational Module"),
                "overview": edu_module_cleaned.get("technique_explanation", ""),
                "misinformation_type": edu_module_cleaned.get("misinformation_type"),
                "technique_explanation": edu_module_cleaned.get("technique_explanation", ""),
                "red_flags": edu_module_cleaned.get("red_flags", []),
                "verification_tips": edu_module_cleaned.get("verification_tips", []),
                "trending_score": edu_module_cleaned.get("trending_score", 0),
                "related_patterns": edu_module_cleaned.get("related_patterns", []),
                "user_action_items": edu_module_cleaned.get("user_action_items", []),
                "sources_of_technique": edu_module_cleaned.get("sources_of_technique", []),
                "example": {
                    "heading": post.get("post_content", {}).get("heading", ""),
                    "body": post.get("post_content", {}).get("body", "")[:500] if post.get("post_content", {}).get("body") else "",
                    "claim": post.get("claim", {}).get("text", "") if isinstance(post.get("claim"), dict) else "",
                    "verdict": post.get("claim", {}).get("verdict_statement", "") if isinstance(post.get("claim"), dict) else "",
                    "tags": post.get("metadata", {}).get("tags", []),
                    "image_url": post.get("metadata", {}).get("image_url"),
                    "source_url": post.get("post_content", {}).get("full_article_url")
                },
                "tags": post.get("metadata", {}).get("tags", []),
                "estimated_time": "15-20 minutes",
                "learning_objectives": [
                    f"Understand how {edu_module_cleaned.get('misinformation_type', 'misinformation')} works",
                    "Identify red flags associated with this technique",
                    "Learn verification strategies to detect this type of misinformation"
                ]
            }
            
            logger.info(f"✅ Retrieved educational module: {module_id}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Failed to get educational module by ID: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            logger.info("🔌 MongoDB connection closed")
