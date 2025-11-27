import { motion } from "framer-motion";
import MotionText from "./MotionText";

const ProgressCard = ({
  label,
  value,
  icon: Icon,
  isDarkMode,
  colorScheme = "blue",
}) => {
  const colorSchemes = {
    blue: {
      bg: isDarkMode ? "bg-blue-900" : "bg-blue-50",
      labelColor: isDarkMode ? "#bfdbfe" : "#1e40af",
      valueColor: isDarkMode ? "#60a5fa" : "#2563eb",
      iconColor: "#3b82f6",
    },
    green: {
      bg: isDarkMode ? "bg-green-900" : "bg-green-50",
      labelColor: isDarkMode ? "#bbf7d0" : "#166534",
      valueColor: isDarkMode ? "#4ade80" : "#16a34a",
      iconColor: "#22c55e",
    },
    yellow: {
      bg: isDarkMode ? "bg-yellow-900" : "bg-yellow-50",
      labelColor: isDarkMode ? "#fde68a" : "#92400e",
      valueColor: isDarkMode ? "#fbbf24" : "#d97706",
      iconColor: "#eab308",
    },
    purple: {
      bg: isDarkMode ? "bg-purple-900" : "bg-purple-50",
      labelColor: isDarkMode ? "#e9d5ff" : "#6b21a8",
      valueColor: isDarkMode ? "#a855f7" : "#9333ea",
      iconColor: "#8b5cf6",
    },
  };

  const colors = colorSchemes[colorScheme];

  return (
    <motion.div
      className={`p-6 rounded-lg ${colors.bg}`}
      animate={{
        backgroundColor: isDarkMode
          ? colors.bg.includes("blue")
            ? "#1e3a8a"
            : colors.bg.includes("green")
            ? "#14532d"
            : colors.bg.includes("yellow")
            ? "#78350f"
            : "#6b21a8"
          : colors.bg.includes("blue")
          ? "#eff6ff"
          : colors.bg.includes("green")
          ? "#f0fdf4"
          : colors.bg.includes("yellow")
          ? "#fefce8"
          : "#faf5ff",
      }}
      transition={{
        duration: 0.6,
        ease: "easeInOut",
      }}
    >
      <div className="flex items-center space-x-3 mb-2">
        {Icon && (
          <Icon className="w-6 h-6" style={{ color: colors.iconColor }} />
        )}
        <MotionText
          className="text-lg font-semibold"
          isDarkMode={isDarkMode}
          darkColor={colors.labelColor}
          lightColor={colors.labelColor}
        >
          {label}
        </MotionText>
      </div>
      <MotionText
        className="text-3xl font-bold"
        isDarkMode={isDarkMode}
        darkColor={colors.valueColor}
        lightColor={colors.valueColor}
      >
        {value}
      </MotionText>
    </motion.div>
  );
};

export default ProgressCard;
