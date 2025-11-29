// eslint-disable-next-line no-unused-vars
import { motion } from "framer-motion";
import { ChevronRight, ChevronDown } from "lucide-react";
import MotionText from "./MotionText";

const SectionHeader = ({ title, isExpanded, onToggle, isDarkMode }) => {
  return (
    <motion.button
      onClick={onToggle}
      className="w-full flex items-center justify-between py-4 rounded-lg text-left cursor-pointer"
      whileTap={{ scale: 0.98 }}
      transition={{ duration: 0.6, ease: "easeInOut" }}
    >
      <div className="flex-1">
        <MotionText
          className="text-xl font-semibold block"
          isDarkMode={isDarkMode}
          darkColor="#ffffff"
          lightColor="#111827"
        >
          {title}
        </MotionText>
      </div>
      <div className="flex-shrink-0 ml-4">
        <motion.div
          animate={{
            color: isDarkMode ? "#9ca3af" : "#6b7280",
            rotate: isExpanded ? 90 : 0,
          }}
          transition={{ duration: 0.6, ease: "easeInOut" }}
        >
          <ChevronRight className="w-6 h-6" />
        </motion.div>
      </div>
    </motion.button>
  );
};

export default SectionHeader;
