"""Login to Twitter and save fresh cookies"""

import asyncio
from twikit import Client
import os

async def login_and_save_cookies():
    """Login to Twitter and save cookies"""
    
    print("=" * 60)
    print("TWITTER LOGIN - SAVE FRESH COOKIES")
    print("=" * 60)
    
    # Get credentials
    print("\nEnter your Twitter credentials:")
    username = input("Username: ").strip()
    email = input("Email (optional, press Enter to skip): ").strip() or None
    password = input("Password: ").strip()
    
    # Create client
    print("\n1. Creating Twitter client...")
    client = Client('en-US')
    
    # Login
    print("2. Logging in...")
    try:
        if email:
            await client.login(
                auth_info_1=username,
                auth_info_2=email,
                password=password
            )
        else:
            await client.login(
                auth_info_1=username,
                password=password
            )
        
        print("   ✓ Login successful!")
        
        # Save cookies
        cookies_file = 'twitter_cookies.json'
        client.save_cookies(cookies_file)
        print(f"\n3. Cookies saved to: {cookies_file}")
        print("   ✓ You can now use these cookies for authentication")
        
        # Test the cookies
        print("\n4. Testing cookies...")
        user = await client.get_user_by_screen_name('twitter')
        print(f"   ✓ Test successful! Fetched: @{user.screen_name}")
        
        print("\n" + "=" * 60)
        print("SUCCESS! Cookies are ready to use")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ LOGIN FAILED: {e}")
        print("\nPossible issues:")
        print("• Wrong username/password")
        print("• 2FA enabled (use browser cookie export instead)")
        print("• Account locked/suspended")
        print("• Rate limited - wait and try again")

if __name__ == "__main__":
    print("\n⚠️  NOTE: If your account has 2FA enabled,")
    print("   use browser cookie export method instead!")
    print()
    
    asyncio.run(login_and_save_cookies())
