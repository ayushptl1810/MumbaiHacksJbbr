import { motion, AnimatePresence } from "framer-motion";
import { X, AlertTriangle, ArrowRight, Crown } from "lucide-react";
import { useNavigate } from "react-router-dom";

const TierLimitModal = ({ isOpen, onClose, limitInfo }) => {
  const navigate = useNavigate();

  if (!isOpen || !limitInfo) return null;

  const { tier, limits, usage, feature = "verification" } = limitInfo;
  
  // Extract limits - can be nested or flat
  const dailyLimit = limits?.daily || limits?.tier_limits?.daily;
  const monthlyLimit = limits?.monthly || limits?.tier_limits?.monthly;
  
  // Extract usage - should have count and limit for each period
  const dailyUsage = usage?.daily || { count: 0, limit: dailyLimit };
  const monthlyUsage = usage?.monthly || { count: 0, limit: monthlyLimit };
  
  const dailyCount = dailyUsage?.count || 0;
  const monthlyCount = monthlyUsage?.count || 0;
  
  const isDailyLimit = dailyLimit && dailyCount >= dailyLimit;
  const isMonthlyLimit = monthlyLimit && monthlyCount >= monthlyLimit;
  const limitType = isDailyLimit ? "daily" : "monthly";
  const remainingCount = isDailyLimit 
    ? 0 
    : isMonthlyLimit 
    ? 0 
    : dailyLimit ? Math.max(0, dailyLimit - dailyCount) : 0;

  const getUpgradeTier = () => {
    if (tier === "Free") return "Plus";
    if (tier === "Plus") return "Pro";
    return null;
  };

  const upgradeTier = getUpgradeTier();

  const handleUpgrade = () => {
    onClose();
    navigate("/subscription");
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50"
          />

          {/* Modal */}
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: 20 }}
            className="fixed inset-0 z-50 flex items-center justify-center p-4"
          >
            <div className="relative w-full max-w-md">
              <div className="rounded-2xl border border-red-500/30 bg-gradient-to-br from-black/90 via-slate-900/90 to-black/90 backdrop-blur-xl p-6 shadow-2xl">
                {/* Close button */}
                <button
                  onClick={onClose}
                  className="absolute right-4 top-4 p-1.5 rounded-lg hover:bg-white/10 transition-colors"
                >
                  <X className="w-5 h-5 text-gray-400" />
                </button>

                {/* Icon */}
                <div className="flex justify-center mb-4">
                  <div className="p-3 rounded-full bg-red-500/20 border border-red-500/50">
                    <AlertTriangle className="w-8 h-8 text-red-400" />
                  </div>
                </div>

                {/* Title */}
                <h2 className="text-2xl font-bold text-white text-center mb-2">
                  {limitType === "daily" ? "Daily Limit Reached" : "Monthly Limit Reached"}
                </h2>

                {/* Description */}
                <p className="text-gray-300 text-center mb-6">
                  You've reached your {limitType} verification limit for the <span className="font-semibold text-white">{tier}</span> tier.
                </p>

                {/* Usage stats */}
                <div className="bg-black/40 rounded-xl p-4 mb-6 space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-400">Daily Usage</span>
                    <span className="text-sm font-semibold text-white">
                      {dailyCount} / {dailyLimit || "∞"}
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-400">Monthly Usage</span>
                    <span className="text-sm font-semibold text-white">
                      {monthlyCount} / {monthlyLimit || "∞"}
                    </span>
                  </div>
                  {remainingCount > 0 && (
                    <div className="pt-2 border-t border-white/10">
                      <span className="text-xs text-gray-400">
                        {remainingCount} more {limitType === "daily" ? "today" : "this month"}
                      </span>
                    </div>
                  )}
                </div>

                {/* Upgrade CTA */}
                {upgradeTier && (
                  <div className="space-y-3">
                    <p className="text-sm text-gray-300 text-center">
                      Upgrade to <span className="font-semibold text-cyan-400">{upgradeTier}</span> to get higher limits:
                    </p>
                    <div className="bg-gradient-to-r from-cyan-500/10 to-blue-500/10 border border-cyan-500/30 rounded-xl p-4 mb-4">
                      <div className="flex items-center gap-2 mb-2">
                        {upgradeTier === "Pro" && <Crown className="w-4 h-4 text-yellow-400" />}
                        <span className="text-sm font-semibold text-cyan-300">{upgradeTier} Tier Benefits</span>
                      </div>
                      <ul className="text-xs text-gray-300 space-y-1 ml-6">
                        {upgradeTier === "Plus" && (
                          <>
                            <li>• 50 verifications per month</li>
                            <li>• 3 verifications per day</li>
                            <li>• All media types enabled</li>
                          </>
                        )}
                        {upgradeTier === "Pro" && (
                          <>
                            <li>• 200 verifications per month</li>
                            <li>• 10 verifications per day</li>
                            <li>• Priority processing</li>
                            <li>• All premium features</li>
                          </>
                        )}
                      </ul>
                    </div>
                    <button
                      onClick={handleUpgrade}
                      className="w-full flex items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-500 px-6 py-3 font-semibold text-white hover:from-cyan-600 hover:to-blue-600 transition-all shadow-lg hover:shadow-cyan-500/50"
                    >
                      Upgrade to {upgradeTier}
                      <ArrowRight className="w-5 h-5" />
                    </button>
                  </div>
                )}

                {/* Free tier message */}
                {!upgradeTier && (
                  <p className="text-sm text-gray-400 text-center">
                    You're already on the highest tier. Limits will reset {limitType === "daily" ? "tomorrow" : "next month"}.
                  </p>
                )}

                {/* Try again later */}
                <button
                  onClick={onClose}
                  className="w-full mt-3 text-sm text-gray-400 hover:text-gray-300 transition-colors"
                >
                  I'll try again later
                </button>
              </div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
};

export default TierLimitModal;

