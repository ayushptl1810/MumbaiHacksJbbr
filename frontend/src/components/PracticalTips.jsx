import { motion } from "framer-motion";
import { Lightbulb } from "lucide-react";
import MotionText from "./MotionText";
import MotionCard from "./MotionCard";

const PracticalTips = ({ tips, isDarkMode }) => {
  if (!tips || tips.length === 0) return null;

  return (
    <MotionCard
      className="p-6 rounded-3xl border border-emerald-400/20 bg-gradient-to-br from-emerald-600/20 via-emerald-500/10 to-emerald-600/5"
      isDarkMode={isDarkMode}
      darkBackgroundColor="#064e3b"
      lightBackgroundColor="#064e3b"
      darkBorderColor="#065f46"
      lightBorderColor="#065f46"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.5, duration: 0.6, ease: "easeInOut" }}
    >
      <MotionText
        className="text-xl font-semibold mb-4 text-emerald-100"
        isDarkMode={isDarkMode}
        darkColor="#bbf7d0"
        lightColor="#166534"
      >
        Practical Tips
      </MotionText>
      <div className="space-y-3">
        {tips.map((tip, index) => (
          <div key={index} className="flex items-start space-x-3">
            <Lightbulb className="w-5 h-5 text-emerald-200 mt-0.5 flex-shrink-0" />
            <MotionText
              className="text-sm leading-relaxed text-emerald-50"
              isDarkMode={isDarkMode}
              darkColor="#bbf7d0"
              lightColor="#15803d"
            >
              {tip}
            </MotionText>
          </div>
        ))}
      </div>
    </MotionCard>
  );
};

export default PracticalTips;
