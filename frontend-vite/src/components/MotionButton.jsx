import { motion } from "framer-motion";

const MotionButton = ({
  children,
  className = "",
  isDarkMode,
  backgroundColor,
  darkBackgroundColor,
  lightBackgroundColor,
  color,
  darkColor,
  lightColor,
  onClick,
  ...props
}) => {
  return (
    <motion.button
      className={className}
      onClick={onClick}
      animate={{
        backgroundColor: isDarkMode
          ? darkBackgroundColor
          : lightBackgroundColor,
        color: isDarkMode ? darkColor : lightColor,
      }}
      transition={{
        duration: 0.6,
        ease: "easeInOut",
      }}
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
      {...props}
    >
      {children}
    </motion.button>
  );
};

export default MotionButton;
