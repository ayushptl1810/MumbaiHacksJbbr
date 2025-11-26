import { motion } from "framer-motion";
import ChatbotView from "./views/ChatbotView";
import ModulesView from "./views/ModulesView";
import ContentView from "./views/ContentView";
import ProgressView from "./views/ProgressView";
import LoadingSpinner from "./ui/LoadingSpinner";

const ViewContainer = ({
  currentView,
  modules,
  moduleContent,
  selectedModule,
  userProgress,
  expandedSections,
  isDarkMode,
  onModuleClick,
  onToggleSection,
  onCompleteModule,
  getModuleIcon,
  setIsDarkMode,
  onLearnClick,
  loading,
}) => {
  const renderView = () => {
    switch (currentView) {
      case "chatbot":
        return (
          <ChatbotView
            isDarkMode={isDarkMode}
            setIsDarkMode={setIsDarkMode}
            onLearnClick={onLearnClick}
          />
        );
      case "modules":
        return (
          <div className="p-4 sm:p-6">
            <ModulesView
              modules={modules}
              userProgress={userProgress}
              isDarkMode={isDarkMode}
              onModuleClick={onModuleClick}
              getModuleIcon={getModuleIcon}
            />
            {loading && (
              <div className="flex justify-center mt-6">
                <LoadingSpinner />
              </div>
            )}
          </div>
        );
      case "content":
        return (
          <div className="p-4 sm:p-6">
            <ContentView
              moduleContent={moduleContent}
              selectedModule={selectedModule}
              userProgress={userProgress}
              expandedSections={expandedSections}
              isDarkMode={isDarkMode}
              onToggleSection={onToggleSection}
              onCompleteModule={onCompleteModule}
            />
            {loading && (
              <div className="flex justify-center mt-6">
                <LoadingSpinner />
              </div>
            )}
          </div>
        );
      case "progress":
        return (
          <ProgressView
            userProgress={userProgress}
            modules={modules}
            isDarkMode={isDarkMode}
          />
        );
      default:
        return (
          <div className="p-4 sm:p-6">
            <ModulesView
              modules={modules}
              userProgress={userProgress}
              isDarkMode={isDarkMode}
              onModuleClick={onModuleClick}
              getModuleIcon={getModuleIcon}
            />
            {loading && (
              <div className="flex justify-center mt-6">
                <LoadingSpinner />
              </div>
            )}
          </div>
        );
    }
  };

  return (
    <motion.div
      key={currentView}
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.6, ease: "easeInOut" }}
      className="h-full"
    >
      {renderView()}
    </motion.div>
  );
};

export default ViewContainer;
