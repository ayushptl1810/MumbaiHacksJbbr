import { motion } from "framer-motion";
import ModulesGrid from "../sections/ModulesGrid";

const ModulesView = ({
  modules,
  userProgress,
  isDarkMode,
  onModuleClick,
  getModuleIcon,
}) => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, ease: "easeInOut" }}
      className="max-w-6xl mx-auto px-4 sm:px-6"
    >
      <ModulesGrid
        modules={modules}
        userProgress={userProgress}
        isDarkMode={isDarkMode}
        onModuleClick={onModuleClick}
        getModuleIcon={getModuleIcon}
      />
    </motion.div>
  );
};

export default ModulesView;
