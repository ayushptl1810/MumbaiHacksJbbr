"""
Script to create test users for Plus and Pro tiers.
Run with: python3 create_test_users.py
"""
import os
import sys
from datetime import datetime
import hashlib
from dotenv import load_dotenv

load_dotenv()

# Add parent directory to path to import MongoDBService
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.mongodb_service import MongoDBService

def hash_password(password: str) -> str:
    """Hash password using SHA256 (matches backend login logic)"""
    return hashlib.sha256(password.encode()).hexdigest()

def create_test_users():
    """Create Plus and Pro test users"""
    try:
        mongodb = MongoDBService()
        
        # Test user credentials
        test_users = [
            {
                "email": "plus_tester@projectaegis.com",
                "plain_password": "PlusTester!123",
                "password": hash_password("PlusTester!123"),
                "name": "Plus Test User",
                "subscription_tier": "Pro",  # Maps to Plus tier
                "domain_preferences": ["Technology", "Science"],
                "phone_number": "+1234567890",
                "age": 25,
            },
            {
                "email": "pro_tester@projectaegis.com",
                "plain_password": "ProTester!123",
                "password": hash_password("ProTester!123"),
                "name": "Pro Test User",
                "subscription_tier": "Enterprise",  # Maps to Pro tier
                "domain_preferences": ["Technology", "Science", "Politics", "Health"],
                "phone_number": "+1234567891",
                "age": 30,
            }
        ]
        
        print("🔍 Creating test users...")
        
        for user_data in test_users:
            email = user_data["email"]
            
            # Check if user already exists
            existing = mongodb.users.find_one({"email": email})
            if existing:
                print(f"⚠️  User {email} already exists. Updating subscription tier...")
                mongodb.update_user_subscription_tier(
                    str(existing["_id"]),
                    user_data["subscription_tier"]
                )
                print(f"✅ Updated {email} to {user_data['subscription_tier']} tier")
            else:
                # Create new user - remove plain_password before inserting
                user_insert_data = {k: v for k, v in user_data.items() if k != "plain_password"}
                user_insert_data["created_at"] = datetime.utcnow()
                user_insert_data["updated_at"] = datetime.utcnow()
                
                result = mongodb.users.insert_one(user_insert_data)
                user_data["_id"] = str(result.inserted_id)
                user_data["id"] = str(result.inserted_id)
                
                print(f"✅ Created {email} with tier: {user_data['subscription_tier']}")
        
        print("\n✅ Test users created/updated successfully!")
        print("\n📋 Login credentials:")
        print("=" * 60)
        for user_data in test_users:
            print(f"\nEmail: {user_data['email']}")
            print(f"Password: {user_data['plain_password']}")
            if user_data['subscription_tier'] == "Pro":
                print("Tier: Plus (subscription_tier: Pro)")
            elif user_data['subscription_tier'] == "Enterprise":
                print("Tier: Pro (subscription_tier: Enterprise)")
        print("=" * 60)
        
        mongodb.close()
        
    except Exception as e:
        print(f"❌ Error creating test users: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    create_test_users()

