// eslint-disable-next-line no-unused-vars
import { motion } from "framer-motion";
import ModuleCard from "./ModuleCard";

const ModulesGrid = ({
  modules,
  userProgress,
  isDarkMode,
  onModuleClick,
  getModuleIcon,
}) => {
  return (
    <motion.div
      className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.6, ease: "easeInOut" }}
    >
      {modules.map((module, index) => (
        <motion.div
          key={module.id}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{
            duration: 0.6,
            ease: "easeInOut",
            delay: index * 0.1,
          }}
        >
          <ModuleCard
            module={module}
            isDarkMode={isDarkMode}
            isCompleted={userProgress.completedModules.includes(module.id)}
            onClick={() => onModuleClick(module.id, userProgress.level)}
            getModuleIcon={getModuleIcon}
          />
        </motion.div>
      ))}
    </motion.div>
  );
};

export default ModulesGrid;
