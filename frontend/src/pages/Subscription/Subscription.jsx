import { motion } from "framer-motion";
import { Check } from "lucide-react";
import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { subscriptionService } from "../../services/api";
import { useAuth } from "../../contexts/AuthContext";

const Subscription = () => {
  const { isAuthenticated, user, refreshUser } = useAuth();
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [razorpayLoaded, setRazorpayLoaded] = useState(false);
  const [razorpayKeyId, setRazorpayKeyId] = useState(null);

  useEffect(() => {
    if (!isAuthenticated) {
      navigate("/login");
    }
  }, [isAuthenticated, navigate]);

  // Load Razorpay Checkout script
  useEffect(() => {
    const script = document.createElement("script");
    script.src = "https://checkout.razorpay.com/v1/checkout.js";
    script.async = true;
    script.onload = () => {
      setRazorpayLoaded(true);
      console.log("✅ Razorpay Checkout script loaded");
    };
    script.onerror = () => {
      console.error("❌ Failed to load Razorpay Checkout script");
      setError("Failed to load payment gateway");
    };
    document.body.appendChild(script);

    // Get Razorpay Key ID from backend
    subscriptionService.getConfig()
      .then((response) => {
        if (response.data.success && response.data.razorpay_key_id) {
          setRazorpayKeyId(response.data.razorpay_key_id);
        }
      })
      .catch((err) => {
        console.error("Failed to get Razorpay config:", err);
        // Fallback to env variable if backend fails
        const keyId = import.meta.env?.VITE_RAZORPAY_KEY_ID;
        if (keyId) {
          setRazorpayKeyId(keyId);
        }
      });

    return () => {
      // Cleanup script on unmount
      const existingScript = document.querySelector('script[src="https://checkout.razorpay.com/v1/checkout.js"]');
      if (existingScript) {
        document.body.removeChild(existingScript);
      }
    };
  }, []);

  const handlePlanClick = async (plan) => {
    if (plan.name === "Free") {
      // Already on the free tier – no checkout needed
      return;
    }

    if (plan.name === "Pro") {
      await handleProSubscription();
      return;
    }

    if (plan.name === "Plus") {
      await handlePlusSubscription();
    }
  };

  const handlePlusSubscription = async () => {
    if (!razorpayLoaded) {
      setError("Payment gateway is still loading. Please wait...");
      return;
    }

    if (!razorpayKeyId) {
      setError("Razorpay Key ID not configured. Please contact support.");
      return;
    }

    setLoading(true);
    setError(null);

    if (!isAuthenticated || !user) {
      setError("You must be logged in to subscribe");
      navigate("/login");
      return;
    }

    try {
      const userId = user.id;

      // Get available plans and find Plus plan (backed by a Razorpay subscription)
      const plansResponse = await subscriptionService.getPlans();
      const plans = plansResponse.data.plans?.items || [];

      // Find Plus plan (by name or explicit env ID)
      let plusPlan =
        plans.find((p) =>
          p.item?.name?.toLowerCase().includes("plus")
        ) ||
        plans.find(
          (p) => p.id === import.meta.env?.VITE_PLUS_PLAN_ID
        );

      // If Plus plan doesn't exist, create it
      if (!plusPlan) {
        // Create Plus plan: $4.99 = 49900 paise (for INR) or 499 cents (for USD)
        // Note: Adjust amount based on your currency
        const planResponse = await subscriptionService.createPlan({
          name: "Plus Plan",
          amount: 49900, // 499.00 in smallest unit (adjust for your currency)
          currency: "INR", // Change to USD if needed
          interval: 1,
          period: "monthly",
          description: "Plus Plan - Monthly Subscription",
        });
        plusPlan = planResponse.data.plan;
      }

      const PLUS_PLAN_ID = plusPlan.id;

      // Create subscription
      const response = await subscriptionService.createSubscription(
        PLUS_PLAN_ID,
        userId,
        { plan_name: "Plus", source: "web" }
      );

      if (response.data.success && response.data.subscription_id) {
        // Initialize Razorpay Checkout
        const options = {
          key: razorpayKeyId,
          subscription_id: response.data.subscription_id,
          name: "Aegis Fact Checker",
          description: "Plus Plan - Monthly Subscription",
          prefill: {
            name: localStorage.getItem("user_name") || "",
            email: localStorage.getItem("user_email") || "",
            contact: localStorage.getItem("user_phone") || "",
          },
          theme: {
            color: "#06b6d4", // cyan-500
          },
          handler: function (response) {
            console.log("✅ Payment successful:", response);
            setLoading(false);
            // Refresh user data to get updated subscription tier
            if (refreshUser) {
              refreshUser();
            }
            // Show success message or redirect
            alert("Subscription activated successfully! Welcome to Plus plan.");
            // Optionally reload or redirect
            window.location.reload();
          },
          modal: {
            ondismiss: function () {
              console.log("Payment modal closed");
              setLoading(false);
            },
          },
        };

        const rzp = new window.Razorpay(options);
        rzp.on("payment.failed", function (response) {
          console.error("❌ Payment failed - Full response:", JSON.stringify(response, null, 2));
          
          // Extract error message from various possible structures
          let errorMsg = "Unknown payment error";
          if (response.error) {
            if (typeof response.error === "string") {
              errorMsg = response.error;
            } else if (response.error.description) {
              errorMsg = response.error.description;
            } else if (response.error.reason) {
              errorMsg = response.error.reason;
            } else if (response.error.code) {
              errorMsg = `Error code: ${response.error.code}`;
            } else {
              errorMsg = JSON.stringify(response.error);
            }
          } else if (response.metadata && response.metadata.error) {
            errorMsg = response.metadata.error;
          }
          
          setError(`Payment failed: ${errorMsg}`);
          setLoading(false);
        });
        
        rzp.on("payment.authorized", function (response) {
          console.log("✅ Payment authorized:", response);
        });
        
        rzp.on("payment.captured", function (response) {
          console.log("✅ Payment captured:", response);
        });

        rzp.open();
      } else {
        throw new Error("Failed to create subscription");
      }
    } catch (err) {
      console.error("❌ Subscription error:", err);
      setError(
        err.response?.data?.detail ||
        err.message ||
        "Failed to initiate subscription. Please try again."
      );
      setLoading(false);
    }
  };

  const handleProSubscription = async () => {
    if (!razorpayLoaded) {
      setError("Payment gateway is still loading. Please wait...");
      return;
    }

    if (!razorpayKeyId) {
      setError("Razorpay Key ID not configured. Please contact support.");
      return;
    }

    setLoading(true);
    setError(null);

    if (!isAuthenticated || !user) {
      setError("You must be logged in to subscribe");
      navigate("/login");
      return;
    }

    try {
      const userId = user.id;

      // Get available plans and find Pro plan (backed by a Razorpay subscription)
      const plansResponse = await subscriptionService.getPlans();
      const plans = plansResponse.data.plans?.items || [];

      // Find Pro plan (by name or explicit env ID)
      // Note: Backend maps "Enterprise" to "Pro" tier
      let proPlan =
        plans.find((p) =>
          p.item?.name?.toLowerCase().includes("pro") ||
          p.item?.name?.toLowerCase().includes("enterprise")
        ) ||
        plans.find(
          (p) => p.id === import.meta.env?.VITE_PRO_PLAN_ID
        );

      // If Pro plan doesn't exist, create it
      if (!proPlan) {
        // Create Pro plan: $9.99 = 99900 paise (for INR) or 999 cents (for USD)
        // Note: Adjust amount based on your currency
        const planResponse = await subscriptionService.createPlan({
          name: "Pro Plan",
          amount: 99900, // 999.00 in smallest unit (adjust for your currency)
          currency: "INR", // Change to USD if needed
          interval: 1,
          period: "monthly",
          description: "Pro Plan - Monthly Subscription",
        });
        proPlan = planResponse.data.plan;
      }

      const PRO_PLAN_ID = proPlan.id;

      // Create subscription
      // Backend maps "Enterprise" plan_name to "Pro" tier
      const response = await subscriptionService.createSubscription(
        PRO_PLAN_ID,
        userId,
        { plan_name: "Enterprise", source: "web" }
      );

      if (response.data.success && response.data.subscription_id) {
        // Initialize Razorpay Checkout
        const options = {
          key: razorpayKeyId,
          subscription_id: response.data.subscription_id,
          name: "Aegis Fact Checker",
          description: "Pro Plan - Monthly Subscription",
          prefill: {
            name: localStorage.getItem("user_name") || "",
            email: localStorage.getItem("user_email") || "",
            contact: localStorage.getItem("user_phone") || "",
          },
          theme: {
            color: "#06b6d4", // cyan-500
          },
          handler: function (response) {
            console.log("✅ Payment successful:", response);
            setLoading(false);
            // Refresh user data to get updated subscription tier
            if (refreshUser) {
              refreshUser();
            }
            // Show success message or redirect
            alert("Subscription activated successfully! Welcome to Pro plan.");
            // Optionally reload or redirect
            window.location.reload();
          },
          modal: {
            ondismiss: function () {
              console.log("Payment modal closed");
              setLoading(false);
            },
          },
        };

        const rzp = new window.Razorpay(options);
        rzp.on("payment.failed", function (response) {
          console.error("❌ Payment failed - Full response:", JSON.stringify(response, null, 2));
          
          // Extract error message from various possible structures
          let errorMsg = "Unknown payment error";
          if (response.error) {
            if (typeof response.error === "string") {
              errorMsg = response.error;
            } else if (response.error.description) {
              errorMsg = response.error.description;
            } else if (response.error.reason) {
              errorMsg = response.error.reason;
            } else if (response.error.code) {
              errorMsg = `Error code: ${response.error.code}`;
            } else {
              errorMsg = JSON.stringify(response.error);
            }
          } else if (response.metadata && response.metadata.error) {
            errorMsg = response.metadata.error;
          }
          
          setError(`Payment failed: ${errorMsg}`);
          setLoading(false);
        });
        
        rzp.on("payment.authorized", function (response) {
          console.log("✅ Payment authorized:", response);
        });
        
        rzp.on("payment.captured", function (response) {
          console.log("✅ Payment captured:", response);
        });

        rzp.open();
      } else {
        throw new Error("Failed to create subscription");
      }
    } catch (err) {
      console.error("❌ Subscription error:", err);
      setError(
        err.response?.data?.detail ||
        err.message ||
        "Failed to initiate subscription. Please try again."
      );
      setLoading(false);
    }
  };

  const plans = [
    {
      name: "Free",
      price: "$0",
      period: "forever",
      description: "Perfect for trying out our fact-checking service.",
      features: [
        {
          title: "5 verifications per day",
          description: "25 verifications per month",
        },
        {
          title: "1 chat session",
          description: "Up to 10 messages per session",
        },
        {
          title: "All verification types",
          description: "Text, images, videos, and URLs",
        },
        {
          title: "Educational modules access",
          description: "Learn how to spot misinformation",
        },
      ],
      ctaLabel: "Use Free Tier",
      guaranteeText: "No credit card required",
    },
    {
      name: "Plus",
      price: "$4.99",
      period: "per month",
      description: "Ideal for regular users who need more capacity.",
      features: [
        {
          title: "10 verifications per day",
          description: "50 verifications per month",
        },
        {
          title: "5 chat sessions",
          description: "Up to 50 messages per session",
        },
        {
          title: "All verification types",
          description: "Text, images, videos, and URLs",
        },
        {
          title: "Full educational modules access",
          description: "Complete learning resources",
        },
      ],
      highlighted: true,
      badgeText: "RECOMMENDED",
      ctaLabel: "Upgrade To Plus",
      guaranteeText: "Cancel anytime",
    },
    {
      name: "Pro",
      price: "$9.99",
      period: "per month",
      description: "For power users and organizations needing high volume.",
      features: [
        {
          title: "25 verifications per day",
          description: "200 verifications per month",
        },
        {
          title: "20 chat sessions",
          description: "Up to 200 messages per session",
        },
        {
          title: "Priority processing",
          description: "Faster verification turnaround",
        },
        {
          title: "All premium features",
          description: "Complete access to all capabilities",
        },
      ],
      ctaLabel: "Upgrade To Pro",
      guaranteeText: "Cancel anytime",
    },
  ];

  return (
    <div className="min-h-screen bg-black py-12">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-12"
        >
          <h1 className="text-4xl font-bold text-white mb-4">Choose Your Plan</h1>
          <p className="text-gray-400 text-lg">
            Select the perfect plan for your fact-checking needs
          </p>
        </motion.div>

        {error && (
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 mb-6">
            <div className="bg-red-500/10 border border-red-500/50 rounded-lg p-4 text-red-400">
              <p className="font-medium">Error</p>
              <p className="text-sm mt-1">{error}</p>
            </div>
          </div>
        )}

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 justify-items-center items-stretch">
          {plans.map((plan, index) => (
            <motion.div
              key={plan.name}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              className="flex h-full w-full justify-center"
            >
              <div className="group relative w-full max-w-sm h-full">
                <div className="relative h-full overflow-hidden rounded-2xl bg-gradient-to-b from-slate-950 to-slate-900 p-[1px] shadow-2xl transition-all duration-300 hover:-translate-y-2 hover:shadow-cyan-500/25">
                  <div
                    className={`absolute inset-0 bg-gradient-to-b ${
                      plan.highlighted
                        ? "from-cyan-500 to-blue-500 opacity-30"
                        : "from-slate-800 to-slate-700 opacity-20"
                    }`}
                  />
                  <div className="relative flex h-full flex-col rounded-2xl bg-gradient-to-b from-slate-950 to-slate-900 p-6">
                    <div className="absolute -left-16 -top-16 h-32 w-32 rounded-full bg-gradient-to-br from-cyan-500/20 to-blue-500/0 blur-2xl transition-all duration-500 group-hover:scale-150 group-hover:opacity-70" />
                    <div className="absolute -bottom-16 -right-16 h-32 w-32 rounded-full bg-gradient-to-br from-blue-500/20 to-cyan-500/0 blur-2xl transition-all duration-500 group-hover:scale-150 group-hover:opacity-70" />

                    {plan.badgeText && (
                      <div className="absolute -right-[1px] -top-[1px] overflow-hidden rounded-tr-2xl">
                        <div className="absolute h-20 w-20 bg-gradient-to-r from-cyan-500 to-blue-500" />
                        <div className="absolute h-20 w-20 bg-slate-950/90" />
                        <div className="absolute right-0 top-[22px] h-[2px] w-[56px] rotate-45 bg-gradient-to-r from-cyan-500 to-blue-500" />
                        <span className="absolute right-1 top-1 text-[10px] font-semibold text-white">
                          {plan.badgeText}
                        </span>
                      </div>
                    )}

                    <div className="relative">
                      <h3 className="text-sm font-medium uppercase tracking-wider text-cyan-500">
                        {plan.name}
                      </h3>
                      <div className="mt-2 flex items-baseline gap-2">
                        <span className="text-3xl font-bold text-white">
                          {plan.price}
                        </span>
                        {plan.period && (
                          <span className="text-sm text-slate-400">
                            /{plan.period}
                          </span>
                        )}
                      </div>
                      {plan.description && (
                        <p className="mt-2 min-h-[48px] text-sm text-slate-400">
                          {plan.description}
                        </p>
                      )}
                    </div>

                    <div className="relative mt-6 flex-1 space-y-4">
                      {plan.features.map((feature, featureIndex) => {
                        const normalizedFeature =
                          typeof feature === "string"
                            ? { title: feature }
                            : feature;
                        return (
                          <div
                            key={`${plan.name}-feature-${featureIndex}`}
                            className="flex items-start gap-3"
                          >
                            <div className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-cyan-500/10">
                              <Check className="h-4 w-4 text-cyan-500" />
                            </div>
                            <div>
                              <p className="text-sm font-medium text-white">
                                {normalizedFeature.title}
                              </p>
                              {normalizedFeature.description && (
                                <p className="text-xs text-slate-400">
                                  {normalizedFeature.description}
                                </p>
                              )}
                            </div>
                          </div>
                        );
                      })}
                    </div>

                    <div className="relative mt-auto pt-6">
                      <button
                        onClick={() => handlePlanClick(plan)}
                        disabled={loading || ((plan.name === "Plus" || plan.name === "Pro") && !razorpayLoaded)}
                        className="group/btn relative w-full overflow-hidden rounded-xl bg-gradient-to-r from-cyan-500 to-blue-500 p-px font-semibold text-white shadow-[0_1000px_0_0_hsl(0_0%_100%_/_0%)_inset] transition-colors hover:shadow-[0_1000px_0_0_hsl(0_0%_100%_/_2%)_inset] disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                        <div className="relative rounded-xl bg-slate-950/50 px-4 py-3 transition-colors group-hover/btn:bg-transparent">
                          <span className="relative flex items-center justify-center gap-2">
                            {loading && (plan.name === "Plus" || plan.name === "Pro") ? (
                              <>
                                <svg
                                  className="animate-spin h-4 w-4"
                                  xmlns="http://www.w3.org/2000/svg"
                                  fill="none"
                                  viewBox="0 0 24 24"
                                >
                                  <circle
                                    className="opacity-25"
                                    cx="12"
                                    cy="12"
                                    r="10"
                                    stroke="currentColor"
                                    strokeWidth="4"
                                  ></circle>
                                  <path
                                    className="opacity-75"
                                    fill="currentColor"
                                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                                  ></path>
                                </svg>
                                Processing...
                              </>
                            ) : (
                              <>
                                {plan.ctaLabel}
                                <svg
                                  stroke="currentColor"
                                  viewBox="0 0 24 24"
                                  fill="none"
                                  className="h-4 w-4 transition-transform duration-300 group-hover/btn:translate-x-1"
                                >
                                  <path
                                    d="M17 8l4 4m0 0l-4 4m4-4H3"
                                    strokeWidth={2}
                                    strokeLinejoin="round"
                                    strokeLinecap="round"
                                  />
                                </svg>
                              </>
                            )}
                          </span>
                        </div>
                      </button>
                    </div>

                    {plan.guaranteeText && (
                      <div className="mt-4 flex items-center justify-center gap-2 text-slate-400">
                        <svg
                          stroke="currentColor"
                          viewBox="0 0 24 24"
                          fill="none"
                          className="h-4 w-4"
                        >
                          <path
                            d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"
                            strokeWidth={2}
                            strokeLinejoin="round"
                            strokeLinecap="round"
                          />
                        </svg>
                        <span className="text-xs font-medium">
                          {plan.guaranteeText}
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default Subscription;

