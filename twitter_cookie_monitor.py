"""Monitor Twitter cookie health and auto-refresh if needed"""

import asyncio
import os
import json
from datetime import datetime, timedelta
from twikit import Client
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TwitterCookieManager:
    """Manage Twitter cookies with health checks and auto-refresh"""
    
    def __init__(self, cookies_file='twitter_cookies.json'):
        self.cookies_file = cookies_file
        self.last_check = None
        self.check_interval = timedelta(hours=6)  # Check every 6 hours
    
    async def check_cookie_health(self) -> bool:
        """
        Test if current cookies are still valid
        Returns: True if cookies work, False if expired/invalid
        """
        if not os.path.exists(self.cookies_file):
            logger.error(f"Cookie file not found: {self.cookies_file}")
            return False
        
        try:
            # Try to make a simple API call
            client = Client('en-US')
            client.load_cookies(self.cookies_file)
            
            # Test with a simple user lookup
            user = await client.get_user_by_screen_name('twitter')
            
            logger.info(f"✓ Cookies are valid (tested with @{user.screen_name})")
            self.last_check = datetime.now()
            return True
            
        except Exception as e:
            error_msg = str(e)
            
            if '403' in error_msg or 'Forbidden' in error_msg:
                logger.error("✗ Cookies expired or invalid (403 Forbidden)")
                return False
            elif '401' in error_msg or 'Unauthorized' in error_msg:
                logger.error("✗ Cookies unauthorized (401)")
                return False
            elif '429' in error_msg or 'rate limit' in error_msg.lower():
                logger.warning("⚠ Rate limited - cookies might be valid")
                return True  # Assume valid, just rate limited
            else:
                logger.error(f"✗ Cookie check failed: {e}")
                return False
    
    def get_cookie_age(self) -> timedelta:
        """Get age of cookie file"""
        if not os.path.exists(self.cookies_file):
            return timedelta.max
        
        mtime = os.path.getmtime(self.cookies_file)
        file_time = datetime.fromtimestamp(mtime)
        age = datetime.now() - file_time
        return age
    
    def should_refresh_cookies(self) -> bool:
        """Check if cookies should be refreshed based on age"""
        age = self.get_cookie_age()
        
        # Refresh if older than 30 days
        if age > timedelta(days=30):
            logger.warning(f"⚠ Cookies are {age.days} days old - recommend refresh")
            return True
        
        return False
    
    async def auto_check_and_alert(self):
        """Periodically check cookie health and alert if issues"""
        logger.info("🔍 Starting cookie health monitor...")
        
        while True:
            try:
                # Check age first
                age = self.get_cookie_age()
                logger.info(f"Cookie file age: {age.days} days, {age.seconds//3600} hours")
                
                if self.should_refresh_cookies():
                    logger.warning("⚠️  COOKIES NEED REFRESH - OLDER THAN 30 DAYS")
                    logger.warning("   Action: Export fresh cookies from browser")
                    logger.warning("   Run: python convert_cookies.py x.com_cookies.txt twitter_cookies.json")
                
                # Check validity
                is_valid = await self.check_cookie_health()
                
                if not is_valid:
                    logger.error("❌ COOKIE VALIDATION FAILED")
                    logger.error("   Scanner will fail until cookies are refreshed!")
                    logger.error("   Steps:")
                    logger.error("   1. Login to twitter.com in browser")
                    logger.error("   2. Export cookies using 'Get cookies.txt LOCALLY' extension")
                    logger.error("   3. Run: python convert_cookies.py x.com_cookies.txt twitter_cookies.json")
                    
                    # Send alert (email, Slack, etc.)
                    self.send_alert("Twitter cookies expired - manual refresh needed")
                
                # Wait before next check
                await asyncio.sleep(self.check_interval.total_seconds())
                
            except Exception as e:
                logger.error(f"Monitor error: {e}")
                await asyncio.sleep(60)  # Retry in 1 minute
    
    def send_alert(self, message: str):
        """Send alert when cookies fail (implement your notification method)"""
        logger.critical(f"🚨 ALERT: {message}")
        
        # TODO: Implement your notification method:
        # - Send email
        # - Send Slack message
        # - Send SMS
        # - Write to monitoring system
        pass
    
    def get_status_report(self) -> dict:
        """Get comprehensive status report"""
        age = self.get_cookie_age()
        
        return {
            'cookie_file': self.cookies_file,
            'exists': os.path.exists(self.cookies_file),
            'age_days': age.days,
            'needs_refresh': self.should_refresh_cookies(),
            'last_check': self.last_check.isoformat() if self.last_check else None,
            'next_check': (datetime.now() + self.check_interval).isoformat() if self.last_check else 'pending'
        }


async def main():
    """Run cookie health check"""
    manager = TwitterCookieManager()
    
    # One-time check
    print("=" * 60)
    print("TWITTER COOKIE HEALTH CHECK")
    print("=" * 60)
    
    is_valid = await manager.check_cookie_health()
    
    status = manager.get_status_report()
    print(f"\n📊 Status Report:")
    print(f"   Cookie File: {status['cookie_file']}")
    print(f"   Exists: {status['exists']}")
    print(f"   Age: {status['age_days']} days")
    print(f"   Needs Refresh: {status['needs_refresh']}")
    print(f"   Valid: {is_valid}")
    
    if not is_valid:
        print("\n❌ ACTION REQUIRED:")
        print("   1. Open browser, go to twitter.com and login")
        print("   2. Export cookies with 'Get cookies.txt LOCALLY' extension")
        print("   3. Run: python convert_cookies.py x.com_cookies.txt twitter_cookies.json")
    elif status['needs_refresh']:
        print("\n⚠️  RECOMMENDED:")
        print("   Cookies are old. Refresh soon to avoid scanner failures.")
    else:
        print("\n✓ Cookies are healthy!")
    
    print("\n" + "=" * 60)
    
    # Ask if user wants continuous monitoring
    print("\n💡 TIP: Run this script periodically or as a background service")
    print("   to monitor cookie health automatically")


if __name__ == "__main__":
    asyncio.run(main())
