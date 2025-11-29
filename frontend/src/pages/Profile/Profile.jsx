import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  User,
  Mail,
  Phone,
  Calendar,
  Tag,
  CreditCard,
  Clock,
  CheckCircle,
  XCircle,
  ShieldCheck,
  Sparkles,
  Zap,
  TrendingUp,
  BarChart3,
  Crown,
  Settings,
  ArrowRight,
  Copy,
  Check,
} from "lucide-react";
import { useAuth } from "../../contexts/AuthContext";
import { subscriptionService } from "../../services/api";
import { useNavigate } from "react-router-dom";

const Profile = () => {
  const { user, isAuthenticated, refreshUser } = useAuth();
  const navigate = useNavigate();
  const [subscription, setSubscription] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (!isAuthenticated) {
      navigate("/login");
      return;
    }

    const fetchSubscription = async () => {
      try {
        if (user?.id) {
          const response = await subscriptionService.getSubscriptionStatus(
            user.id
          );
          if (response.data.success) {
            setSubscription(response.data.subscription);
          }
        }
      } catch (err) {
        console.error("Failed to fetch subscription:", err);
        setError("Failed to load subscription information");
      } finally {
        setLoading(false);
      }
    };

    fetchSubscription();
    if (refreshUser) {
      refreshUser();
    }
  }, [user, isAuthenticated, navigate, refreshUser]);

  const getTierColor = (tier) => {
    switch (tier?.toLowerCase()) {
      case "pro":
        return {
          gradient: "from-cyan-500 via-blue-500 to-cyan-600",
          light: "from-cyan-500/20 via-blue-500/20 to-cyan-600/20",
          border: "border-cyan-500/50",
          text: "text-cyan-300",
        };
      case "enterprise":
        return {
          gradient: "from-purple-600 via-indigo-600 to-purple-700",
          light: "from-purple-600/20 via-indigo-600/20 to-purple-700/20",
          border: "border-purple-600/50",
          text: "text-purple-300",
        };
      default:
        return {
          gradient: "from-gray-500 to-gray-600",
          light: "from-gray-500/20 to-gray-600/20",
          border: "border-gray-500/50",
          text: "text-gray-300",
        };
    }
  };

  const getTierBadgeColor = (tier) => {
    switch (tier?.toLowerCase()) {
      case "pro":
        return "bg-cyan-500/20 text-cyan-300 border-cyan-500/50";
      case "enterprise":
        return "bg-purple-600/20 text-purple-300 border-purple-600/50";
      default:
        return "bg-gray-500/20 text-gray-300 border-gray-500/50";
    }
  };

  const getTierIconColor = (tier) => {
    switch (tier?.toLowerCase()) {
      case "pro":
        return "text-cyan-400";
      case "enterprise":
        return "text-purple-400";
      default:
        return "text-gray-400";
    }
  };

  const formatDate = (dateString) => {
    if (!dateString) return "N/A";
    try {
      const date = new Date(dateString);
      return date.toLocaleDateString("en-US", {
        year: "numeric",
        month: "long",
        day: "numeric",
      });
    } catch {
      return "Invalid date";
    }
  };

  const getDaysUntilRenewal = (dateString) => {
    if (!dateString) return null;
    try {
      const renewalDate = new Date(dateString);
      const today = new Date();
      const diffTime = renewalDate - today;
      const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
      return diffDays;
    } catch {
      return null;
    }
  };

  const copyToClipboard = async (text) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error("Failed to copy:", err);
    }
  };

  const currentTier =
    user?.subscription_tier || subscription?.plan_name || "Free";
  const renewalDate = subscription?.next_billing_at
    ? new Date(subscription.next_billing_at * 1000).toISOString()
    : null;
  const daysUntilRenewal = renewalDate
    ? getDaysUntilRenewal(renewalDate)
    : null;
  const isActive = subscription?.status === "active";
  const tierColors = getTierColor(currentTier);

  if (loading) {
    return (
      <div className="min-h-screen bg-black flex items-center justify-center">
        <motion.div
          animate={{ rotate: 360 }}
          transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
          className="w-12 h-12 border-4 border-blue-500/30 border-t-blue-500 rounded-full"
        />
      </div>
    );
  }

  const initialLetter = (user?.name || user?.email || "?")
    .charAt(0)
    .toUpperCase();

  return (
    <div className="min-h-screen bg-black relative overflow-hidden">
      {/* Enhanced Animated Background */}
      <div
        className="fixed inset-0 overflow-hidden pointer-events-none"
        style={{ zIndex: 0 }}
      >
        {[...Array(3)].map((_, i) => (
          <motion.div
            key={i}
            className={`absolute w-[30rem] h-[30rem] rounded-full blur-3xl opacity-20`}
            animate={{
              x: [0, 100 * (i + 1), 0],
              y: [0, 50 * (i + 1), 0],
              scale: [1, 1.2 + i * 0.1, 1],
            }}
            transition={{
              duration: 15 + i * 5,
              repeat: Infinity,
              ease: "easeInOut",
            }}
            style={{
              top: `${10 + i * 30}%`,
              left: `${5 + i * 20}%`,
              background: `radial-gradient(circle, ${
                i === 0
                  ? "rgba(59, 130, 246, 0.4)"
                  : i === 1
                  ? "rgba(139, 92, 246, 0.4)"
                  : "rgba(6, 182, 212, 0.4)"
              }, transparent)`,
            }}
          />
        ))}
        <div
          className="absolute inset-0 opacity-[0.03]"
          style={{
            backgroundImage: `
              linear-gradient(rgba(59, 130, 246, 0.5) 1px, transparent 1px),
              linear-gradient(90deg, rgba(59, 130, 246, 0.5) 1px, transparent 1px)
            `,
            backgroundSize: "60px 60px",
          }}
        />
      </div>

      <div className="relative z-10 py-8 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          {/* Epic Header */}
          <motion.div
            initial={{ opacity: 0, y: -30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, ease: "easeOut" }}
            className="mb-10"
          >
            <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-6">
              {/* Left: Avatar & Name */}
              <div className="flex items-center gap-6">
                <div className="relative group">
                  <motion.div
                    animate={{
                      boxShadow: [
                        "0 0 0 0 rgba(59, 130, 246, 0.4)",
                        "0 0 0 8px rgba(59, 130, 246, 0)",
                        "0 0 0 0 rgba(59, 130, 246, 0)",
                      ],
                    }}
                    transition={{ duration: 2, repeat: Infinity }}
                    className="absolute -inset-2 rounded-full"
                  />
                  <div className="relative w-20 h-20 rounded-2xl bg-gradient-to-br from-blue-500 via-cyan-400 to-purple-500 p-[2px]">
                    <div className="w-full h-full rounded-2xl bg-black flex items-center justify-center">
                      <span className="text-2xl font-bold text-transparent bg-clip-text bg-gradient-to-br from-blue-400 to-purple-400">
                        {initialLetter}
                      </span>
                    </div>
                  </div>
                  <div className="absolute -bottom-1 -right-1 w-7 h-7 rounded-full bg-gradient-to-br from-emerald-400 to-cyan-400 flex items-center justify-center border-2 border-black shadow-lg">
                    <ShieldCheck className="w-4 h-4 text-black" />
                  </div>
                </div>
                <div>
                  <motion.p
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.2 }}
                    className="text-xs uppercase tracking-[0.3em] text-blue-400/80 mb-2 font-medium"
                  >
                    Welcome Back
                  </motion.p>
                  <motion.h1
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.3 }}
                    className="text-4xl sm:text-5xl font-bold text-white mb-2 bg-gradient-to-r from-white via-blue-100 to-cyan-100 bg-clip-text text-transparent"
                  >
                    {user?.name || "Your Profile"}
                  </motion.h1>
                  <motion.p
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.4 }}
                    className="text-gray-400 text-sm flex items-center gap-2"
                  >
                    <span>{user?.email}</span>
                    <button
                      onClick={() => copyToClipboard(user?.email || "")}
                      className="p-1 hover:bg-white/10 rounded transition-colors"
                    >
                      {copied ? (
                        <Check className="w-3 h-3 text-emerald-400" />
                      ) : (
                        <Copy className="w-3 h-3 text-gray-500" />
                      )}
                    </button>
                  </motion.p>
                </div>
              </div>

              {/* Right: Tier Badge */}
              <motion.div
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.5, type: "spring" }}
                className={`inline-flex items-center gap-3 px-6 py-3 rounded-2xl border-2 backdrop-blur-xl ${getTierBadgeColor(
                  currentTier
                )} bg-black/40 shadow-xl`}
              >
                {currentTier === "Enterprise" && (
                  <Crown className="w-5 h-5 animate-pulse" />
                )}
                <div>
                  <p className="text-xs uppercase tracking-wider text-gray-400 mb-0.5">
                    Current Plan
                  </p>
                  <p className="text-lg font-bold flex items-center gap-2">
                    {currentTier} Tier
                    {isActive && (
                      <span className="inline-flex h-2 w-2 rounded-full bg-emerald-400 shadow-[0_0_8px_rgba(16,185,129,0.6)]" />
                    )}
                  </p>
                </div>
              </motion.div>
            </div>
          </motion.div>

          {/* Main Content Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
            {/* Subscription Card - Left Side (2/3) */}
            <motion.div
              initial={{ opacity: 0, x: -30 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.2 }}
              className="lg:col-span-2 relative group"
            >
              <div
                className={`relative overflow-hidden rounded-2xl border-2 ${tierColors.border} bg-gradient-to-br ${tierColors.light} backdrop-blur-2xl p-8 shadow-2xl`}
              >
                {/* Animated gradient overlay */}
                <motion.div
                  animate={{
                    backgroundPosition: ["0% 50%", "100% 50%", "0% 50%"],
                  }}
                  transition={{ duration: 5, repeat: Infinity }}
                  className="absolute inset-0 opacity-50"
                  style={{
                    background: `linear-gradient(90deg, transparent, ${tierColors.gradient
                      .split(" ")[0]
                      .replace("from-", "rgba(")}, 0.1), transparent)`,
                    backgroundSize: "200% 200%",
                  }}
                />

                <div className="relative z-10">
                  {/* Header */}
                  <div className="flex items-start justify-between mb-6">
                    <div className="flex items-start gap-4">
                      <div
                        className={`p-4 rounded-2xl bg-gradient-to-br ${tierColors.gradient} shadow-lg`}
                      >
                        <CreditCard className="w-6 h-6 text-white" />
                      </div>
                      <div>
                        <h2 className="text-2xl font-bold text-white mb-1 flex items-center gap-2">
                          Subscription Overview
                          <Sparkles className="w-5 h-5 text-yellow-400" />
                        </h2>
                        <p className="text-sm text-gray-300">
                          Manage your plan, billing, and premium features
                        </p>
                      </div>
                    </div>
                    {currentTier !== "Free" && (
                      <motion.button
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                        onClick={() => navigate("/subscription")}
                        className="px-5 py-2.5 bg-white/10 hover:bg-white/20 text-white rounded-xl text-sm font-medium transition-all border border-white/20 backdrop-blur-sm"
                      >
                        Manage
                      </motion.button>
                    )}
                  </div>

                  {/* Status Badges */}
                  <div className="flex flex-wrap items-center gap-3 mb-6">
                    <motion.span
                      initial={{ scale: 0 }}
                      animate={{ scale: 1 }}
                      transition={{ delay: 0.4, type: "spring" }}
                      className={`inline-flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-semibold border backdrop-blur-sm ${getTierBadgeColor(
                        currentTier
                      )}`}
                    >
                      {isActive ? (
                        <CheckCircle className="w-4 h-4" />
                      ) : (
                        <XCircle className="w-4 h-4" />
                      )}
                      {currentTier} {isActive ? "Active" : "Inactive"}
                    </motion.span>
                    {subscription?.status && (
                      <span className="px-3 py-2 rounded-xl bg-black/40 text-xs uppercase tracking-wide text-gray-300 border border-white/10">
                        {subscription.status}
                      </span>
                    )}
                  </div>

                  {/* Subscription Details Grid */}
                  {currentTier !== "Free" && renewalDate && (
                    <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                      {[
                        {
                          icon: Clock,
                          label: "Next Renewal",
                          value: formatDate(renewalDate),
                          sub:
                            daysUntilRenewal !== null
                              ? daysUntilRenewal > 0
                                ? `${daysUntilRenewal} day${
                                    daysUntilRenewal !== 1 ? "s" : ""
                                  } remaining`
                                : daysUntilRenewal === 0
                                ? "Renews today"
                                : "Expired"
                              : null,
                        },
                        {
                          icon: CreditCard,
                          label: "Plan",
                          value: subscription?.plan_name || currentTier,
                          sub: subscription?.razorpay_subscription_id
                            ? `ID: ${subscription.razorpay_subscription_id.slice(
                                0,
                                8
                              )}...`
                            : null,
                        },
                        {
                          icon: ShieldCheck,
                          label: "Protection",
                          value: "Active",
                          sub: "Full coverage enabled",
                        },
                      ].map((item, idx) => (
                        <motion.div
                          key={idx}
                          initial={{ opacity: 0, y: 10 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ delay: 0.5 + idx * 0.1 }}
                          className="bg-black/40 rounded-xl p-4 border border-white/10 hover:border-white/20 transition-all"
                        >
                          <div className="flex items-center gap-2 text-gray-400 mb-2">
                            <item.icon className="w-4 h-4" />
                            <span className="text-[10px] uppercase tracking-wider">
                              {item.label}
                            </span>
                          </div>
                          <p className="text-sm font-semibold text-white mb-1">
                            {item.value}
                          </p>
                          {item.sub && (
                            <p className="text-xs text-gray-400">{item.sub}</p>
                          )}
                        </motion.div>
                      ))}
                    </div>
                  )}

                  {/* Free Tier CTA */}
                  {currentTier === "Free" && (
                    <motion.div
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.5 }}
                      className="mt-6 p-6 bg-black/40 rounded-xl border-2 border-dashed border-white/20"
                    >
                      <p className="text-gray-200 text-sm mb-4">
                        Unlock the full potential with{" "}
                        <span className="text-cyan-400 font-semibold">Pro</span>{" "}
                        or{" "}
                        <span className="text-purple-400 font-semibold">
                          Enterprise
                        </span>
                        . Get priority verification, unlimited requests, and
                        advanced AI explainability.
                      </p>
                      <motion.button
                        whileHover={{ scale: 1.02 }}
                        whileTap={{ scale: 0.98 }}
                        onClick={() => navigate("/subscription")}
                        className="inline-flex items-center gap-2 px-5 py-2.5 bg-gradient-to-r from-cyan-500 to-blue-500 text-white rounded-xl text-sm font-semibold hover:from-cyan-600 hover:to-blue-600 transition-all shadow-lg shadow-cyan-500/30"
                      >
                        Explore Plans
                        <ArrowRight className="w-4 h-4" />
                      </motion.button>
                    </motion.div>
                  )}
                </div>
              </div>
            </motion.div>

            {/* Personal Info Card - Right Side (1/3) */}
            <motion.div
              initial={{ opacity: 0, x: 30 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.3 }}
              className="rounded-2xl border border-white/10 bg-black/40 backdrop-blur-2xl p-5 shadow-xl"
            >
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                  <div className="p-2 rounded-lg bg-white/5">
                    <User className="w-4 h-4 text-gray-300" />
                  </div>
                  <h2 className="text-base font-semibold text-white">
                    Personal Information
                  </h2>
                </div>
                <span className="text-[10px] uppercase tracking-[0.18em] text-gray-500">
                  Identity
                </span>
              </div>

              <div className="grid grid-cols-1 gap-3">
                {[
                  { icon: User, label: "Full Name", value: user?.name },
                  { icon: Mail, label: "Email", value: user?.email },
                  { icon: Phone, label: "Phone", value: user?.phone_number },
                  {
                    icon: Calendar,
                    label: "Age",
                    value: user?.age ? `${user.age} years` : null,
                  },
                ].map((field, idx) => (
                  <motion.div
                    key={idx}
                    initial={{ opacity: 0, x: 10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.4 + idx * 0.05 }}
                    className="flex items-center justify-between gap-3 px-3 py-2.5 rounded-xl bg-black/40 border border-white/5 hover:border-white/15 transition-all"
                  >
                    <div className="flex items-center gap-2 min-w-0">
                      <field.icon className="w-3.5 h-3.5 text-gray-400 flex-shrink-0" />
                      <div className="flex flex-col min-w-0">
                        <span className="text-[10px] uppercase tracking-wide text-gray-500">
                          {field.label}
                        </span>
                        <span className="text-xs text-white font-medium truncate">
                          {field.value || "Not provided"}
                        </span>
                      </div>
                    </div>
                  </motion.div>
                ))}
              </div>
            </motion.div>
          </div>

          {/* Interests & Actions Row */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Interests */}
            {user?.domain_preferences && user.domain_preferences.length > 0 && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.5 }}
                className="rounded-2xl border border-white/10 bg-black/40 backdrop-blur-xl p-6 shadow-xl"
              >
                <h2 className="text-xl font-bold text-white mb-5 flex items-center gap-2">
                  <Tag className="w-5 h-5" />
                  Interests & Preferences
                </h2>
                <div className="flex flex-wrap gap-2">
                  {user.domain_preferences.map((domain, index) => (
                    <motion.span
                      key={index}
                      initial={{ opacity: 0, scale: 0.8 }}
                      animate={{ opacity: 1, scale: 1 }}
                      transition={{ delay: 0.6 + index * 0.05 }}
                      whileHover={{ scale: 1.05 }}
                      className="px-4 py-2 bg-gradient-to-r from-cyan-500/20 to-blue-500/20 text-cyan-300 border border-cyan-500/50 rounded-xl text-sm font-medium hover:from-cyan-500/30 hover:to-blue-500/30 transition-all cursor-default backdrop-blur-sm"
                    >
                      {domain}
                    </motion.span>
                  ))}
                </div>
              </motion.div>
            )}

            {/* Account Actions */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.6 }}
              className="rounded-2xl border border-white/10 bg-black/40 backdrop-blur-xl p-6 shadow-xl"
            >
              <h2 className="text-xl font-bold text-white mb-5 flex items-center gap-2">
                <Settings className="w-5 h-5" />
                Quick Actions
              </h2>
              <div className="flex flex-col gap-3">
                <motion.button
                  whileHover={{ scale: 1.02, x: 5 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={() => navigate("/subscription")}
                  className="flex items-center justify-between p-4 bg-gradient-to-r from-cyan-500/20 to-blue-500/20 border border-cyan-500/30 rounded-xl text-white hover:from-cyan-500/30 hover:to-blue-500/30 transition-all group"
                >
                  <span className="font-semibold">Manage Subscription</span>
                  <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
                </motion.button>
                <motion.button
                  whileHover={{ scale: 1.02, x: 5 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={() => navigate("/")}
                  className="flex items-center justify-between p-4 bg-white/5 border border-white/10 rounded-xl text-white hover:bg-white/10 transition-all group"
                >
                  <span className="font-medium">Back to Home</span>
                  <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
                </motion.button>
              </div>
            </motion.div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Profile;
