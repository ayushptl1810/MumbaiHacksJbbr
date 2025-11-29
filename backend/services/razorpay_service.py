"""
Razorpay Service for Subscription Management
Handles Razorpay API interactions for subscription payments
"""

import logging
import hmac
import hashlib
from typing import Dict, Any, Optional
import razorpay
from config import config

logger = logging.getLogger(__name__)


class RazorpayService:
    """Service for handling Razorpay subscription operations"""
    
    def __init__(self):
        """Initialize Razorpay client"""
        if not config.RAZORPAY_ID or not config.RAZORPAY_KEY:
            logger.warning("⚠️ Razorpay credentials not configured. Subscription features will not work.")
            self.client = None
        else:
            try:
                # Initialize Razorpay client with explicit base URL
                # Test mode uses different base URL, but SDK handles this automatically
                self.client = razorpay.Client(auth=(config.RAZORPAY_ID, config.RAZORPAY_KEY))
                logger.info(f"✅ Razorpay client initialized with Key ID: {config.RAZORPAY_ID[:8]}...")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Razorpay client: {e}")
                self.client = None
    
    def create_plan(
        self,
        name: str,
        amount: int,
        currency: str = "INR",
        interval: int = 1,
        period: str = "monthly",
        description: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a subscription plan in Razorpay
        
        Args:
            name: Plan name
            amount: Amount in smallest currency unit (paise for INR)
            currency: Currency code (default: INR)
            interval: Billing interval (default: 1)
            period: Billing period - 'daily', 'weekly', 'monthly', 'yearly' (default: monthly)
            description: Plan description
            
        Returns:
            Dict containing plan details from Razorpay
        """
        if not self.client:
            raise ValueError("Razorpay client not initialized. Check RAZORPAY_ID and RAZORPAY_KEY.")
        
        try:
            plan_data = {
                "period": period,
                "interval": interval,
                "item": {
                    "name": name,
                    "amount": amount,
                    "currency": currency,
                    "description": description or f"{name} subscription plan"
                }
            }
            
            logger.debug(f"Creating plan with data: {plan_data}")
            # Try creating plan - note: some accounts may need subscriptions enabled first
            plan = self.client.plan.create(plan_data)
            logger.info(f"✅ Created Razorpay plan: {plan.get('id')}")
            return plan
        except razorpay.errors.BadRequestError as e:
            error_msg = str(e)
            logger.error(f"❌ BadRequestError creating plan '{name}': {error_msg}")
            # Check if it's a "URL not found" error which indicates subscriptions might not be enabled
            if "not found" in error_msg.lower() or "url" in error_msg.lower():
                logger.error(f"   This error typically means:")
                logger.error(f"   1. Subscriptions feature is NOT enabled on your Razorpay account")
                logger.error(f"   2. You need to enable subscriptions in Razorpay Dashboard")
                logger.error(f"   3. Go to: Razorpay Dashboard > Settings > Subscriptions")
                logger.error(f"   4. Or contact Razorpay support to enable subscriptions")
            # Check if plan already exists
            elif "already exists" in error_msg.lower() or "duplicate" in error_msg.lower():
                logger.warning(f"⚠️ Plan '{name}' may already exist")
            raise
        except razorpay.errors.ServerError as e:
            logger.error(f"❌ ServerError creating plan '{name}': {e}")
            raise
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            logger.error(f"❌ Failed to create Razorpay plan '{name}' ({error_type}): {error_msg}")
            # Log more details if available
            if hasattr(e, 'status_code'):
                logger.error(f"   Status code: {e.status_code}")
            if hasattr(e, 'error'):
                logger.error(f"   Error details: {e.error}")
            raise
    
    def create_subscription(
        self,
        plan_id: str,
        customer_notify: int = 1,
        total_count: Optional[int] = None,
        start_at: Optional[int] = None,
        end_at: Optional[int] = None,
        notes: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Create a subscription for a user
        
        Args:
            plan_id: Razorpay plan ID
            customer_notify: Whether to notify customer (1 or 0)
            total_count: Total number of billing cycles (None for infinite - will use end_at instead)
            start_at: Unix timestamp for subscription start (None for immediate)
            end_at: Unix timestamp for subscription end (used if total_count is None for infinite subscriptions)
            notes: Additional notes/metadata
            
        Returns:
            Dict containing subscription details from Razorpay
        """
        if not self.client:
            raise ValueError("Razorpay client not initialized. Check RAZORPAY_ID and RAZORPAY_KEY.")
        
        try:
            subscription_data = {
                "plan_id": plan_id,
                "customer_notify": customer_notify,
            }
            
            # Razorpay requires either total_count or end_at
            # If end_at is provided, start_at is also required
            # start_at must be in the future (add 60 seconds buffer to account for clock differences)
            import time
            current_time = int(time.time())
            # Add 60 seconds buffer to ensure start_at is always in the future
            future_start_time = current_time + 60
            
            if total_count is not None:
                subscription_data["total_count"] = total_count
            elif end_at is not None:
                subscription_data["end_at"] = end_at
                # If end_at is set but start_at is not, set start_at to 60 seconds in the future
                if start_at is None:
                    subscription_data["start_at"] = future_start_time
            else:
                # Set both start_at and end_at for infinite subscriptions
                subscription_data["start_at"] = future_start_time
                subscription_data["end_at"] = future_start_time + (10 * 365 * 24 * 60 * 60)  # 10 years
                logger.info("ℹ️ No total_count or end_at provided, setting start_at to 60 seconds in future and end_at to 10 years from start (infinite subscription)")
            
            # Override start_at if explicitly provided (but ensure it's in the future)
            if start_at is not None:
                if start_at <= current_time:
                    # If provided start_at is in the past, add 60 seconds buffer
                    subscription_data["start_at"] = current_time + 60
                    logger.warning(f"⚠️ Provided start_at was in the past, adjusted to {subscription_data['start_at']}")
                else:
                    subscription_data["start_at"] = start_at
            
            if notes:
                subscription_data["notes"] = notes
            
            subscription = self.client.subscription.create(subscription_data)
            logger.info(f"✅ Created Razorpay subscription: {subscription.get('id')}")
            return subscription
        except Exception as e:
            logger.error(f"❌ Failed to create Razorpay subscription: {e}")
            raise
    
    def get_subscription(self, subscription_id: str) -> Dict[str, Any]:
        """
        Get subscription details from Razorpay
        
        Args:
            subscription_id: Razorpay subscription ID
            
        Returns:
            Dict containing subscription details
        """
        if not self.client:
            raise ValueError("Razorpay client not initialized. Check RAZORPAY_ID and RAZORPAY_KEY.")
        
        try:
            subscription = self.client.subscription.fetch(subscription_id)
            return subscription
        except Exception as e:
            logger.error(f"❌ Failed to fetch subscription {subscription_id}: {e}")
            raise
    
    def cancel_subscription(
        self,
        subscription_id: str,
        cancel_at_cycle_end: bool = False
    ) -> Dict[str, Any]:
        """
        Cancel a subscription
        
        Args:
            subscription_id: Razorpay subscription ID
            cancel_at_cycle_end: If True, cancel at end of current cycle
            
        Returns:
            Dict containing updated subscription details
        """
        if not self.client:
            raise ValueError("Razorpay client not initialized. Check RAZORPAY_ID and RAZORPAY_KEY.")
        
        try:
            if cancel_at_cycle_end:
                subscription = self.client.subscription.cancel(
                    subscription_id,
                    {"cancel_at_cycle_end": 1}
                )
            else:
                subscription = self.client.subscription.cancel(subscription_id)
            
            logger.info(f"✅ Cancelled subscription: {subscription_id}")
            return subscription
        except Exception as e:
            logger.error(f"❌ Failed to cancel subscription {subscription_id}: {e}")
            raise
    
    def verify_webhook_signature(
        self,
        payload: str,
        signature: str
    ) -> bool:
        """
        Verify Razorpay webhook signature
        
        Args:
            payload: Raw webhook payload (string)
            signature: Webhook signature from X-Razorpay-Signature header
            
        Returns:
            True if signature is valid, False otherwise
        """
        if not config.RAZORPAY_WEBHOOK_SECRET:
            logger.warning("⚠️ RAZORPAY_WEBHOOK_SECRET not set. Webhook verification skipped.")
            return True  # Allow if secret not configured (for development)
        
        try:
            expected_signature = hmac.new(
                config.RAZORPAY_WEBHOOK_SECRET.encode('utf-8'),
                payload.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            return hmac.compare_digest(expected_signature, signature)
        except Exception as e:
            logger.error(f"❌ Webhook signature verification failed: {e}")
            return False
    
    def get_plan(self, plan_id: str) -> Dict[str, Any]:
        """
        Get plan details from Razorpay
        
        Args:
            plan_id: Razorpay plan ID
            
        Returns:
            Dict containing plan details
        """
        if not self.client:
            raise ValueError("Razorpay client not initialized. Check RAZORPAY_ID and RAZORPAY_KEY.")
        
        try:
            plan = self.client.plan.fetch(plan_id)
            return plan
        except Exception as e:
            logger.error(f"❌ Failed to fetch plan {plan_id}: {e}")
            raise
    
    def list_plans(self, count: int = 10, skip: int = 0) -> Dict[str, Any]:
        """
        List all plans
        
        Args:
            count: Number of plans to fetch
            skip: Number of plans to skip
            
        Returns:
            Dict containing list of plans
        """
        if not self.client:
            raise ValueError("Razorpay client not initialized. Check RAZORPAY_ID and RAZORPAY_KEY.")
        
        try:
            # Try to list plans - this may fail if no plans exist or API endpoint is different
            plans = self.client.plan.all({"count": count, "skip": skip})
            return plans
        except razorpay.errors.BadRequestError as e:
            error_msg = str(e).lower()
            logger.error(f"❌ BadRequestError listing plans: {e}")
            # Check if it's a "not found" error which might mean subscriptions aren't enabled
            if "not found" in error_msg or "url" in error_msg:
                logger.warning("⚠️ Subscriptions API endpoint not found. This might mean:")
                logger.warning("   1. Subscriptions feature is not enabled on your Razorpay account")
                logger.warning("   2. Your API keys don't have subscription permissions")
                logger.warning("   3. You need to enable subscriptions in Razorpay Dashboard")
            # Return empty structure if it's a "not found" type error
            return {"items": [], "count": 0}
        except razorpay.errors.ServerError as e:
            logger.error(f"❌ ServerError listing plans: {e}")
            raise
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            logger.error(f"❌ Failed to list plans ({error_type}): {error_msg}")
            # If it's a "not found" error, return empty list instead of raising
            if "not found" in error_msg.lower() or "404" in error_msg:
                logger.warning("⚠️ No plans found or endpoint not available, returning empty list")
                return {"items": [], "count": 0}
            raise

