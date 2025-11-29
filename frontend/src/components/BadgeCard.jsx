// eslint-disable-next-line no-unused-vars
import { motion } from "framer-motion";
import MotionText from "./MotionText";

const BadgeCard = ({ badge, isDarkMode }) => {
  return (
    <motion.div
      className={`p-4 rounded-lg text-center ${
        isDarkMode ? "bg-yellow-900" : "bg-yellow-50"
      }`}
      animate={{
        backgroundColor: isDarkMode ? "#78350f" : "#fefce8",
      }}
      transition={{
        duration: 0.6,
        ease: "easeInOut",
      }}
    >
      <div className="text-3xl mb-2">{badge.icon}</div>
      <MotionText
        className="text-sm font-medium"
        isDarkMode={isDarkMode}
        darkColor="#fde68a"
        lightColor="#92400e"
      >
        {badge.name}
      </MotionText>
    </motion.div>
  );
};

export default BadgeCard;
