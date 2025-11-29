// eslint-disable-next-line no-unused-vars
import { motion } from "framer-motion";

const MotionCard = ({
  children,
  className = "",
  isDarkMode,
  darkBackgroundColor,
  lightBackgroundColor,
  darkBorderColor,
  lightBorderColor,
  ...props
}) => {
  return (
    <motion.div
      className={className}
      animate={{
        backgroundColor: isDarkMode
          ? darkBackgroundColor
          : lightBackgroundColor,
        borderColor: isDarkMode ? darkBorderColor : lightBorderColor,
      }}
      transition={{
        duration: 0.6,
        ease: "easeInOut",
      }}
      {...props}
    >
      {children}
    </motion.div>
  );
};

export default MotionCard;
