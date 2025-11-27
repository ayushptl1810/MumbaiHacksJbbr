import { motion } from "framer-motion";
import { Check } from "lucide-react";

const Subscription = () => {
  const plans = [
    {
      name: "Free",
      price: "$0",
      period: "forever",
      description: "Great for individuals validating occasional claims.",
      features: [
        { title: "Basic fact-checking" },
        { title: "5 verifications per day", description: "Fair-use limits" },
        { title: "Community support" },
      ],
      ctaLabel: "Start For Free",
      guaranteeText: "No credit card required",
    },
    {
      name: "Pro",
      price: "$9.99",
      period: "per month",
      description: "Built for teams that rely on trusted information daily.",
      features: [
        {
          title: "Unlimited verifications",
          description: "Remove usage caps entirely",
        },
        {
          title: "Priority processing",
          description: "Jump the verification queue",
        },
        {
          title: "Advanced AI analysis",
          description: "Deeper cross-source checks",
        },
        { title: "Email support" },
        { title: "Detailed reports", description: "Download and share" },
      ],
      highlighted: true,
      badgeText: "POPULAR",
      ctaLabel: "Upgrade To Pro",
      guaranteeText: "30-day money-back guarantee",
    },
    {
      name: "Enterprise",
      price: "Custom",
      period: "",
      description: "Tailored verification pipelines for large organizations.",
      features: [
        { title: "Everything in Pro" },
        { title: "API access", description: "Embed verification workflows" },
        { title: "Custom integrations" },
        {
          title: "Dedicated support",
          description: "Named specialists & playbooks",
        },
        { title: "SLA guarantee" },
      ],
      ctaLabel: "Talk To Us",
      guaranteeText: "Custom SLAs & onboarding",
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
                      <button className="group/btn relative w-full overflow-hidden rounded-xl bg-gradient-to-r from-cyan-500 to-blue-500 p-px font-semibold text-white shadow-[0_1000px_0_0_hsl(0_0%_100%_/_0%)_inset] transition-colors hover:shadow-[0_1000px_0_0_hsl(0_0%_100%_/_2%)_inset]">
                        <div className="relative rounded-xl bg-slate-950/50 px-4 py-3 transition-colors group-hover/btn:bg-transparent">
                          <span className="relative flex items-center justify-center gap-2">
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

