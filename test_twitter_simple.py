"""Simple test script to debug Twitter/Twikit authentication and API calls"""

import asyncio
import os
from twikit import Client

async def test_twitter_basic():
    """Test basic Twitter operations with Twikit"""
    
    print("=" * 60)
    print("TWIKIT TWITTER TEST")
    print("=" * 60)
    
    # Create client
    print("\n1. Creating Twikit client...")
    client = Client('en-US')
    print("   ✓ Client created")
    
    # Load cookies
    cookies_file = 'twitter_cookies.json'
    if os.path.exists(cookies_file):
        print(f"\n2. Loading cookies from {cookies_file}...")
        client.load_cookies(cookies_file)
        print("   ✓ Cookies loaded")
    else:
        print(f"\n2. ERROR: {cookies_file} not found!")
        return
    
    # Test 1: Get user info
    print("\n3. Testing: Get user by screen name (@elonmusk)...")
    try:
        user = await client.get_user_by_screen_name('elonmusk')
        print(f"   ✓ User found: {user.name} (@{user.screen_name})")
        print(f"   - Followers: {user.followers_count}")
        print(f"   - Following: {user.following_count}")
    except Exception as e:
        print(f"   ✗ FAILED: {e}")
        if '403' in str(e):
            print("   → 403 Forbidden - Possible causes:")
            print("      • Rate limiting")
            print("      • Cookies expired or invalid")
            print("      • Account suspended/restricted")
            print("      • IP blocked")
    
    # Test 2: Get tweets from user
    print("\n4. Testing: Get tweets from user...")
    try:
        user = await client.get_user_by_screen_name('elonmusk')
        tweets = await user.get_tweets('Tweets', count=5)
        print(f"   ✓ Fetched {len(tweets)} tweets")
        if tweets:
            print(f"   First tweet: {tweets[0].text[:100]}...")
    except Exception as e:
        print(f"   ✗ FAILED: {e}")
        if '403' in str(e):
            print("   → 403 Forbidden (see causes above)")
    
    # Test 3: Search tweets
    print("\n5. Testing: Search tweets for 'python'...")
    try:
        search_results = await client.search_tweet('python', 'Latest', count=5)
        print(f"   ✓ Found {len(search_results)} tweets")
        if search_results:
            print(f"   First result: {search_results[0].text[:100]}...")
    except Exception as e:
        print(f"   ✗ FAILED: {e}")
        if '403' in str(e):
            print("   → 403 Forbidden (see causes above)")
    
    # Test 4: Get trending topics
    print("\n6. Testing: Get trending topics...")
    try:
        trends = await client.get_trends('trending')
        print(f"   ✓ Fetched {len(trends)} trends")
        if trends:
            for i, trend in enumerate(trends[:5], 1):
                topic_name = trend.name if hasattr(trend, 'name') else str(trend)
                print(f"   {i}. {topic_name}")
    except Exception as e:
        print(f"   ✗ FAILED: {e}")
        if '403' in str(e):
            print("   → 403 Forbidden (see causes above)")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


async def test_multiple_clients():
    """Test creating multiple fresh clients (simulating sequential scans)"""
    
    print("\n\n" + "=" * 60)
    print("MULTIPLE CLIENT TEST (Sequential Scans)")
    print("=" * 60)
    
    cookies_file = 'twitter_cookies.json'
    
    # First scan
    print("\n--- SCAN 1: @elonmusk ---")
    client1 = Client('en-US')
    if os.path.exists(cookies_file):
        client1.load_cookies(cookies_file)
        print("✓ Client 1 created and authenticated")
        
        try:
            user = await client1.get_user_by_screen_name('elonmusk')
            print(f"✓ Fetched user: {user.name}")
        except Exception as e:
            print(f"✗ Failed: {e}")
    
    # Second scan (fresh client)
    print("\n--- SCAN 2: Search 'python' ---")
    client2 = Client('en-US')
    if os.path.exists(cookies_file):
        client2.load_cookies(cookies_file)
        print("✓ Client 2 created and authenticated")
        
        try:
            tweets = await client2.search_tweet('python', 'Latest', count=3)
            print(f"✓ Found {len(tweets)} tweets")
        except Exception as e:
            print(f"✗ Failed: {e}")
    
    print("\n" + "=" * 60)
    print("MULTIPLE CLIENT TEST COMPLETE")
    print("=" * 60)


def main():
    """Run all tests"""
    print("\n🔍 Testing Twitter/Twikit Integration\n")
    
    # Test 1: Basic operations
    asyncio.run(test_twitter_basic())
    
    # Test 2: Multiple clients
    asyncio.run(test_multiple_clients())
    
    print("\n\n💡 TROUBLESHOOTING TIPS:")
    print("=" * 60)
    print("If you see 403 Forbidden errors:")
    print("1. Check if twitter_cookies.json exists and is valid")
    print("2. Try logging into Twitter web and exporting fresh cookies")
    print("3. Check if your IP is rate-limited (try different network)")
    print("4. Verify Twitter account is not suspended/restricted")
    print("5. Wait 15-30 minutes if rate-limited")
    print("\nCookie export tools:")
    print("- Chrome: 'Get cookies.txt LOCALLY' extension")
    print("- Firefox: 'cookies.txt' extension")
    print("- Convert cookies.txt to JSON format")
    print("=" * 60)


if __name__ == "__main__":
    main()
