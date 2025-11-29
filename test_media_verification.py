"""
Test script to verify if media verification is working
"""

import asyncio
import sys
from claim_verifier.tools import TextFactChecker

async def test_media_verification():
    """Test if media verifiers are loaded and working"""
    
    print("=" * 60)
    print("MEDIA VERIFICATION TEST")
    print("=" * 60)
    
    try:
        # Initialize fact checker
        print("\n1. Initializing TextFactChecker...")
        checker = TextFactChecker()
        print("   ✅ TextFactChecker initialized")
        
        # Check if image verifier is available
        print("\n2. Checking Image Verifier...")
        if checker.image_verifier:
            print("   ✅ Image verifier is AVAILABLE")
            print(f"   Type: {type(checker.image_verifier).__name__}")
        else:
            print("   ❌ Image verifier is NOT AVAILABLE")
            print("   Reason: SerpAPI key missing or import failed")
        
        # Check if video verifier is available
        print("\n3. Checking Video Verifier...")
        if checker.video_verifier:
            print("   ✅ Video verifier is AVAILABLE")
            print(f"   Type: {type(checker.video_verifier).__name__}")
        else:
            print("   ❌ Video verifier is NOT AVAILABLE")
            print("   Reason: SerpAPI key missing or import failed")
        
        # Test media detection
        print("\n4. Testing Media Detection...")
        
        test_cases = [
            {
                "name": "Image URL in claim data",
                "context": "This image shows a protest",
                "data": {"url": "https://i.imgur.com/example.jpg"}
            },
            {
                "name": "Video URL in claim data",
                "context": "This video shows an event",
                "data": {"url": "https://youtube.com/watch?v=example"}
            },
            {
                "name": "Image keyword in text",
                "context": "This viral image shows a protest in 2020",
                "data": {}
            },
            {
                "name": "No media",
                "context": "This is just text without any media",
                "data": {}
            }
        ]
        
        for test in test_cases:
            print(f"\n   Test: {test['name']}")
            media_info = checker._detect_media_in_claim(test['context'], test['data'])
            
            if media_info['has_media']:
                print(f"      ✅ Media detected: {media_info['media_type']}")
                if media_info['media_urls']:
                    print(f"      URLs: {media_info['media_urls']}")
                if media_info['media_description']:
                    print(f"      Description: {media_info['media_description']}")
            else:
                print(f"      ❌ No media detected")
        
        # Test actual verification with media (if verifiers available)
        if checker.image_verifier or checker.video_verifier:
            print("\n5. Testing Actual Verification with Media...")
            
            test_claim_data = {
                'text_input': 'Viral image shows protest in New York 2020',
                'claim_context': 'This image shows a protest in New York during 2020',
                'claim_date': '2020-01-01',
                'url': 'https://i.redd.it/example.jpg'  # Example image URL
            }
            
            print("   Running verification (this may take a few seconds)...")
            try:
                result = await checker.verify(**test_claim_data)
                
                print("\n   Verification completed!")
                print(f"   - Verdict: {result.get('verdict')}")
                print(f"   - Confidence: {result.get('confidence')}")
                
                if 'media_analysis' in result:
                    print("\n   🎉 MEDIA ANALYSIS INCLUDED!")
                    print(f"   - Media Type: {result['media_analysis'].get('type')}")
                    print(f"   - Media Verdict: {result['media_analysis'].get('verdict')}")
                    print(f"   - Media Summary: {result['media_analysis'].get('summary', '')[:100]}...")
                else:
                    print("\n   ⚠️  No media analysis in result")
                    print("   (This might be normal if media verification failed or was skipped)")
                
            except Exception as e:
                print(f"   ⚠️  Verification test failed: {e}")
        else:
            print("\n5. Skipping actual verification test (no verifiers available)")
        
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        
        if checker.image_verifier and checker.video_verifier:
            print("✅ FULL MEDIA VERIFICATION ENABLED")
            print("   Both image and video verifiers are working")
        elif checker.image_verifier:
            print("⚠️  PARTIAL MEDIA VERIFICATION")
            print("   Only image verifier is available")
        elif checker.video_verifier:
            print("⚠️  PARTIAL MEDIA VERIFICATION")
            print("   Only video verifier is available")
        else:
            print("❌ MEDIA VERIFICATION DISABLED")
            print("   No media verifiers available")
            print("\nTo enable media verification:")
            print("1. Set SERP_API_KEY in your environment")
            print("2. Ensure image_verifier.py and video_verifier.py exist")
            print("3. Install required packages: pip install Pillow opencv-python google-search-results")
        
        print("\n" + "=" * 60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(test_media_verification())
