// eslint-disable-next-line no-unused-vars
import { motion, AnimatePresence } from "framer-motion";
import { CheckCircle } from "lucide-react";
import MotionText from "./MotionText";
import MotionCard from "./MotionCard";
import SectionHeader from "./SectionHeader";

const ContentSection = ({
  section,
  isExpanded,
  onToggle,
  isDarkMode,
}) => {
  return (
    <MotionCard
      className="rounded-3xl border border-white/5 bg-gradient-to-br from-[#11182a]/90 via-[#0d1324]/85 to-[#0b0f1b]/90 p-1"
      isDarkMode={isDarkMode}
      darkBackgroundColor="#0f172a"
      lightBackgroundColor="#0f172a"
      darkBorderColor="#1e293b"
      lightBorderColor="#1e293b"
      initial={{
        opacity: 0,
        y: 20,
        backgroundColor: "#0f172a",
      }}
      animate={{
        opacity: 1,
        y: 0,
        backgroundColor: "#0f172a",
      }}
      transition={{ duration: 0.6, ease: "easeInOut" }}
    >
      <div className="rounded-[26px] bg-black/40 p-6 sm:p-8">
        <SectionHeader
          title={section.title}
          isExpanded={isExpanded}
          onToggle={onToggle}
          isDarkMode={isDarkMode}
        />

        <AnimatePresence>
          {isExpanded && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ duration: 0.6, ease: "easeInOut" }}
              className="mt-6 space-y-4"
            >
              <MotionText
                className="text-base leading-relaxed text-slate-200"
                isDarkMode={isDarkMode}
                darkColor="#e2e8f0"
                lightColor="#0f172a"
              >
                {section.content}
              </MotionText>

              {section.key_points && (
                <div>
                  <MotionText
                    className="text-lg font-medium mb-3"
                    isDarkMode={isDarkMode}
                    darkColor="#ffffff"
                    lightColor="#111827"
                  >
                    Key Points
                  </MotionText>
                  <ul className="space-y-2">
                    {section.key_points.map((point, pointIndex) => (
                      <li
                        key={pointIndex}
                        className="flex items-start space-x-3 text-slate-300"
                      >
                        <CheckCircle className="w-5 h-5 text-emerald-300 mt-0.5 flex-shrink-0" />
                        <MotionText
                          className="text-sm leading-relaxed"
                          isDarkMode={isDarkMode}
                          darkColor="#cbd5f5"
                          lightColor="#1f2933"
                        >
                          {point}
                        </MotionText>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {section.visual_indicators && (
                <div>
                  <MotionText
                    className="text-lg font-medium mb-3"
                    isDarkMode={isDarkMode}
                    darkColor="#ffffff"
                    lightColor="#111827"
                  >
                    Visual Indicators
                  </MotionText>
                  <div className="flex flex-wrap gap-2">
                    {section.visual_indicators.map(
                      (indicator, indicatorIndex) => (
                        <motion.span
                          key={indicatorIndex}
                          className="px-4 py-1.5 rounded-full text-xs font-semibold uppercase tracking-wide border border-amber-400/30 bg-amber-400/10 text-amber-200"
                          transition={{ duration: 0.3, ease: "easeInOut" }}
                        >
                          {indicator}
                        </motion.span>
                      )
                    )}
                  </div>
                </div>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </MotionCard>
  );
};

export default ContentSection;
