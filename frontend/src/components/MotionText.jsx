import { motion } from "framer-motion";

const MotionText = ({
  children,
  className = "",
  isDarkMode,
  color,
  darkColor,
  lightColor,
  type = "div",
  ...props
}) => {
  const MotionElement = motion[type];

  return (
    <MotionElement
      className={className}
      animate={{
        color: isDarkMode ? darkColor : lightColor,
      }}
      transition={{
        duration: 0.6,
        ease: "easeInOut",
      }}
      {...props}
    >
      {children}
    </MotionElement>
  );
};

export default MotionText;
