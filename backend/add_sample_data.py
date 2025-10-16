#!/usr/bin/env python3
"""
Script to add sample rumour data to MongoDB for testing real-time updates
"""

import os
import sys
import asyncio
from datetime import datetime, timedelta
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_mongo_client():
    """Get MongoDB client connection"""
    connection_string = os.getenv('MONGO_CONNECTION_STRING')
    if not connection_string:
        raise ValueError("MONGO_CONNECTION_STRING environment variable not set")
    
    client = MongoClient(connection_string)
    # Test connection
    client.admin.command('ping')
    return client

def add_sample_rumours():
    """Add sample rumour data to MongoDB"""
    
    client = get_mongo_client()
    db = client['aegis']
    collection = db['debunk_posts']
    
    # Sample rumour data with unique post_ids
    sample_rumours = [
        {
            "post_id": "sample_rumour_001",
            "claim": "Scientists have discovered a new planet that could support human life",
            "summary": "Recent astronomical observations suggest the possibility of a habitable exoplanet",
            "platform": "Twitter",
            "Post_link": "https://twitter.com/example/status/123456789",
            "verification": {
                "verdict": "true",
                "message": "This claim is accurate based on NASA's recent findings",
                "reasoning": "The discovery was confirmed by multiple telescopes and peer-reviewed research",
                "verification_date": datetime.now() - timedelta(hours=2),
                "sources": {
                    "count": 3,
                    "links": [
                        "https://www.nasa.gov/feature/nasa-discovers-new-exoplanet",
                        "https://www.nature.com/articles/space-discovery-2024",
                        "https://www.scientificamerican.com/article/new-habitable-planet"
                    ],
                    "titles": [
                        "NASA Discovers New Exoplanet",
                        "Nature: Space Discovery 2024",
                        "Scientific American: New Habitable Planet Found"
                    ]
                }
            },
            "stored_at": datetime.now() - timedelta(hours=2)
        },
        {
            "post_id": "sample_rumour_002", 
            "claim": "Breaking: Major tech company announces they're shutting down all services",
            "summary": "A viral post claims a major technology company is discontinuing all its services",
            "platform": "Facebook",
            "Post_link": "https://facebook.com/example/posts/987654321",
            "verification": {
                "verdict": "false",
                "message": "This is completely false and has been debunked by the company",
                "reasoning": "The company's official channels have confirmed this is a hoax. No such announcement was made.",
                "verification_date": datetime.now() - timedelta(hours=1, minutes=30),
                "sources": {
                    "count": 2,
                    "links": [
                        "https://company.com/official-statement",
                        "https://techcrunch.com/company-denies-shutdown-rumors"
                    ],
                    "titles": [
                        "Official Company Statement",
                        "TechCrunch: Company Denies Shutdown Rumors"
                    ]
                }
            },
            "stored_at": datetime.now() - timedelta(hours=1, minutes=30)
        },
        {
            "post_id": "sample_rumour_003",
            "claim": "New study shows that coffee increases life expectancy by 5 years",
            "summary": "A recent research paper claims significant health benefits from coffee consumption",
            "platform": "Instagram",
            "Post_link": "https://instagram.com/p/coffee-study-2024",
            "verification": {
                "verdict": "mostly true",
                "message": "While coffee does have health benefits, the 5-year claim is exaggerated",
                "reasoning": "Studies show moderate coffee consumption has health benefits, but the specific 5-year claim is not supported by the research cited.",
                "verification_date": datetime.now() - timedelta(minutes=45),
                "sources": {
                    "count": 4,
                    "links": [
                        "https://www.nejm.org/journal/coffee-health-study",
                        "https://www.mayoclinic.org/coffee-health-benefits",
                        "https://www.hsph.harvard.edu/coffee-research",
                        "https://www.healthline.com/coffee-life-expectancy-study"
                    ],
                    "titles": [
                        "NEJM: Coffee Health Study",
                        "Mayo Clinic: Coffee Health Benefits",
                        "Harvard: Coffee Research",
                        "Healthline: Coffee Life Expectancy Study"
                    ]
                }
            },
            "stored_at": datetime.now() - timedelta(minutes=45)
        },
        {
            "post_id": "sample_rumour_004",
            "claim": "Local restaurant caught serving expired food to customers",
            "summary": "Social media posts allege a popular local restaurant is serving expired ingredients",
            "platform": "Reddit",
            "Post_link": "https://reddit.com/r/localnews/expired-food-restaurant",
            "verification": {
                "verdict": "disputed",
                "message": "The claims are under investigation by health authorities",
                "reasoning": "Health department inspection is ongoing. Some allegations have been confirmed, others are disputed by the restaurant management.",
                "verification_date": datetime.now() - timedelta(minutes=20),
                "sources": {
                    "count": 3,
                    "links": [
                        "https://healthdept.gov/inspection-reports",
                        "https://localnews.com/restaurant-investigation",
                        "https://restaurant.com/official-response"
                    ],
                    "titles": [
                        "Health Department Inspection Reports",
                        "Local News: Restaurant Investigation",
                        "Restaurant Official Response"
                    ]
                }
            },
            "stored_at": datetime.now() - timedelta(minutes=20)
        },
        {
            "post_id": "sample_rumour_005",
            "claim": "Mysterious lights spotted in the sky over the city last night",
            "summary": "Multiple reports of unusual lights in the night sky",
            "platform": "TikTok",
            "Post_link": "https://tiktok.com/@user/video/mysterious-lights-city",
            "verification": {
                "verdict": "unverified",
                "message": "Unable to verify the source or authenticity of these reports",
                "reasoning": "No official explanation has been provided. Could be various phenomena including aircraft, drones, or natural occurrences.",
                "verification_date": datetime.now() - timedelta(minutes=10),
                "sources": {
                    "count": 2,
                    "links": [
                        "https://weather.gov/sky-conditions-report",
                        "https://faa.gov/flight-tracker-archive"
                    ],
                    "titles": [
                        "Weather Service: Sky Conditions Report",
                        "FAA: Flight Tracker Archive"
                    ]
                }
            },
            "stored_at": datetime.now() - timedelta(minutes=10)
        }
    ]
    
    print("🔄 Adding sample rumour data to MongoDB...")
    
    added_count = 0
    skipped_count = 0
    
    for rumour in sample_rumours:
        try:
            # Try to insert the document
            result = collection.insert_one(rumour)
            print(f"✅ Added rumour: {rumour['post_id']} - {rumour['claim'][:50]}...")
            added_count += 1
            
        except DuplicateKeyError:
            print(f"⚠️  Skipped rumour (already exists): {rumour['post_id']}")
            skipped_count += 1
            
        except Exception as e:
            print(f"❌ Error adding rumour {rumour['post_id']}: {e}")
    
    print(f"\n📊 Summary:")
    print(f"   ✅ Added: {added_count} rumours")
    print(f"   ⚠️  Skipped: {skipped_count} rumours")
    print(f"   📝 Total in database: {collection.count_documents({})} rumours")
    
    # Close connection
    client.close()
    print("\n🔌 MongoDB connection closed")

def test_realtime_update():
    """Add a new rumour to test real-time updates"""
    
    client = get_mongo_client()
    db = client['aegis']
    collection = db['debunk_posts']
    
    # Create a new rumour with current timestamp
    new_rumour = {
        "post_id": f"test_realtime_{int(datetime.now().timestamp())}",
        "claim": "Test real-time update: This is a new rumour added for testing WebSocket functionality",
        "summary": "This rumour was added to test the real-time WebSocket update system",
        "platform": "Test Platform",
        "Post_link": "https://example.com/test-realtime-update",
        "verification": {
            "verdict": "true",
            "message": "This is a test rumour for real-time updates",
            "reasoning": "Added programmatically to verify WebSocket functionality",
            "verification_date": datetime.now(),
            "sources": {
                "count": 1,
                "links": ["https://example.com/test-source"],
                "titles": ["Test Source"]
            }
        },
        "stored_at": datetime.now()
    }
    
    print("🔄 Adding test rumour for real-time update...")
    
    try:
        result = collection.insert_one(new_rumour)
        print(f"✅ Test rumour added successfully!")
        print(f"   📝 Post ID: {new_rumour['post_id']}")
        print(f"   📅 Added at: {new_rumour['stored_at']}")
        print(f"   🔍 MongoDB ID: {result.inserted_id}")
        print("\n💡 Check your frontend - you should see this new rumour appear automatically!")
        
    except Exception as e:
        print(f"❌ Error adding test rumour: {e}")
    
    # Close connection
    client.close()
    print("\n🔌 MongoDB connection closed")

if __name__ == "__main__":
    print("🚀 MongoDB Sample Data Script")
    print("=" * 50)
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_realtime_update()
    else:
        add_sample_rumours()
    
    print("\n✨ Script completed!")
    print("\n💡 Usage:")
    print("   python add_sample_data.py          # Add sample rumours")
    print("   python add_sample_data.py test     # Add test rumour for real-time updates")
