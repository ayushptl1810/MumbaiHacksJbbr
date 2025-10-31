"""
MongoDB Integration for Project Aegis
Stores debunk posts and verification results in MongoDB
"""

import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, DuplicateKeyError
import uuid
from dotenv import load_dotenv
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AegisMongoDB:
    """MongoDB integration for storing Aegis debunk posts and verification results"""
    
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
        self.collections = {}
        
        self._connect()
        self._setup_collections()
    
    def _connect(self):
        """Establish MongoDB connection"""
        try:
            self.client = MongoClient(self.connection_string)
            # Test connection
            self.client.admin.command('ping')
            
            # Use 'aegis' database
            self.db = self.client['aegis']
            logger.info("✅ Successfully connected to MongoDB")
            
        except ConnectionFailure as e:
            logger.error(f"❌ Failed to connect to MongoDB: {e}")
            raise
    
    def _setup_collections(self):
        """Setup MongoDB collections with proper indexes"""
        try:
            # Single collection for debunk posts
            self.collections['debunk_posts'] = self.db['debunk_posts']
            
            # Create indexes for better performance
            self._create_indexes()
            
            logger.info("✅ MongoDB collections setup completed")
            
        except Exception as e:
            logger.error(f"❌ Failed to setup MongoDB collections: {e}")
            raise
    
    def _create_indexes(self):
        """Create database indexes for better query performance"""
        try:
            # Debunk posts indexes
            self.collections['debunk_posts'].create_index("post_id", unique=True)
            self.collections['debunk_posts'].create_index("stored_at")
            self.collections['debunk_posts'].create_index("claim.verdict")
            self.collections['debunk_posts'].create_index("pipeline_run_id")
            
            logger.info("✅ Database indexes created successfully")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to create some indexes: {e}")
    
    def store_debunk_posts(self, debunk_posts: List[Dict[str, Any]], pipeline_run_id: str) -> Dict[str, Any]:
        """Store debunk posts in MongoDB
        
        Args:
            debunk_posts: List of debunk post objects
            pipeline_run_id: Unique identifier for this pipeline run
            
        Returns:
            Dictionary with storage results
        """
        try:
            stored_count = 0
            skipped_count = 0
            error_count = 0
            
            for post in debunk_posts:
                try:
                    # Add MongoDB-specific fields
                    post_doc = {
                        "_id": str(uuid.uuid4()),
                        "post_id": post.get('post_id', str(uuid.uuid4())),
                        "pipeline_run_id": pipeline_run_id,
                        "stored_at": datetime.utcnow(),
                        **post  # Include all original post data
                    }
                    
                    # Try to insert the document
                    result = self.collections['debunk_posts'].insert_one(post_doc)
                    
                    if result.inserted_id:
                        stored_count += 1
                        logger.info(f"✅ Stored debunk post: {post.get('post_id', 'unknown')}")
                    else:
                        error_count += 1
                        logger.warning(f"⚠️ Failed to store debunk post: {post.get('post_id', 'unknown')}")
                        
                except DuplicateKeyError:
                    skipped_count += 1
                    logger.info(f"⏭️ Skipped duplicate debunk post: {post.get('post_id', 'unknown')}")
                except Exception as e:
                    error_count += 1
                    logger.error(f"❌ Error storing debunk post {post.get('post_id', 'unknown')}: {e}")
            
            result = {
                "success": True,
                "stored_count": stored_count,
                "skipped_count": skipped_count,
                "error_count": error_count,
                "total_processed": len(debunk_posts),
                "pipeline_run_id": pipeline_run_id
            }
            
            logger.info(f"📊 Debunk posts storage completed: {stored_count} stored, {skipped_count} skipped, {error_count} errors")
            return result
            
        except Exception as e:
            logger.error(f"❌ Failed to store debunk posts: {e}")
            return {
                "success": False,
                "error": str(e),
                "stored_count": 0,
                "skipped_count": 0,
                "error_count": len(debunk_posts)
            }
    
    
    
    def upload_results_file(self, results_file_path: str) -> Dict[str, Any]:
        """Upload debunk posts from a JSON file to MongoDB
        
        Args:
            results_file_path: Path to the results JSON file
            
        Returns:
            Dictionary with upload results
        """
        try:
            # Read the results file
            with open(results_file_path, 'r', encoding='utf-8') as f:
                results_data = json.load(f)
            
            logger.info(f"📁 Reading results file: {results_file_path}")
            
            # Extract pipeline run ID
            workflow_id = results_data.get('google_agents_workflow', {}).get('workflow_id', str(uuid.uuid4()))
            
            # Extract and store debunk posts
            debunk_posts = results_data.get('debunk_posts', [])
            debunk_result = self.store_debunk_posts(debunk_posts, workflow_id)
            
            # Return simplified results
            upload_result = {
                "success": True,
                "pipeline_run_id": workflow_id,
                "file_path": results_file_path,
                "debunk_posts": debunk_result,
                "summary": {
                    "debunk_posts_stored": debunk_result.get("stored_count", 0),
                    "total_processed": debunk_result.get("total_processed", 0)
                }
            }
            
            logger.info(f"✅ Successfully uploaded debunk posts from {results_file_path}")
            logger.info(f"📊 Summary: {upload_result['summary']}")
            
            return upload_result
            
        except FileNotFoundError:
            error_msg = f"❌ Results file not found: {results_file_path}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg
            }
        except json.JSONDecodeError as e:
            error_msg = f"❌ Invalid JSON in results file: {e}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg
            }
        except Exception as e:
            error_msg = f"❌ Failed to upload results file: {e}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg
            }
    
    def get_recent_posts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent debunk posts from MongoDB
        
        Args:
            limit: Maximum number of posts to return
            
        Returns:
            List of recent debunk posts
        """
        try:
            posts = list(self.collections['debunk_posts']
                        .find()
                        .sort("stored_at", -1)
                        .limit(limit))
            
            logger.info(f"📋 Retrieved {len(posts)} recent debunk posts")
            return posts
            
        except Exception as e:
            logger.error(f"❌ Failed to get recent posts: {e}")
            return []
    
    def get_posts_by_verdict(self, verdict: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get debunk posts by verification verdict
        
        Args:
            verdict: Verification verdict (true, false, uncertain, error)
            limit: Maximum number of posts to return
            
        Returns:
            List of debunk posts with specified verdict
        """
        try:
            posts = list(self.collections['debunk_posts']
                        .find({"claim.verdict": verdict})
                        .sort("stored_at", -1)
                        .limit(limit))
            
            logger.info(f"📋 Retrieved {len(posts)} posts with verdict '{verdict}'")
            return posts
            
        except Exception as e:
            logger.error(f"❌ Failed to get posts by verdict: {e}")
            return []
    
    def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            logger.info("🔌 MongoDB connection closed")


def test_mongodb_integration():
    """Test the MongoDB integration with sample data"""
    try:
        logger.info("🧪 Testing MongoDB integration...")
        
        # Initialize MongoDB connection
        mongo = AegisMongoDB()
        
        # Test connection
        logger.info("✅ MongoDB connection successful")
        
        # Test with a sample results file if it exists
        results_dir = "/Users/ayushpatel/Documents/GitHub/hack2skill/agent/orchestrator_results"
        import glob
        
        # Find the most recent results file
        results_files = glob.glob(f"{results_dir}/*.json")
        if results_files:
            latest_file = max(results_files, key=os.path.getctime)
            logger.info(f"📁 Found latest results file: {latest_file}")
            
            # Upload the debunk posts
            upload_result = mongo.upload_results_file(latest_file)
            
            if upload_result["success"]:
                logger.info("✅ Test upload successful!")
                logger.info(f"📊 Upload summary: {upload_result['summary']}")
                
                # Test queries
                recent_posts = mongo.get_recent_posts(5)
                logger.info(f"📋 Retrieved {len(recent_posts)} recent posts")
                
                uncertain_posts = mongo.get_posts_by_verdict("uncertain", 3)
                logger.info(f"📋 Retrieved {len(uncertain_posts)} uncertain posts")
                
            else:
                logger.error(f"❌ Test upload failed: {upload_result.get('error', 'Unknown error')}")
        else:
            logger.warning("⚠️ No results files found for testing")
        
        # Close connection
        mongo.close()
        
        logger.info("🎉 MongoDB integration test completed!")
        
    except Exception as e:
        logger.error(f"❌ MongoDB integration test failed: {e}")
        raise


if __name__ == "__main__":
    # Run the test when script is executed directly
    test_mongodb_integration()
