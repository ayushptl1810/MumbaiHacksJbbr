import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { BookOpen, Shield, AlertTriangle, Target, Brain } from "lucide-react";

// Import all the components
import SidebarHeader from "./sections/SidebarHeader";
import SidebarNavigation from "./sections/SidebarNavigation";
import CurrentRumours from "./sections/CurrentRumours";
import PageHeader from "./sections/PageHeader";
import LoadingSpinner from "./ui/LoadingSpinner";
import ViewContainer from "./ViewContainer";
import RumourModal from "./ui/RumourModal";
import InfoModal from "./ui/InfoModal";
import { getApiBaseUrl } from "../config/api";

const MainApp = ({ isDarkMode, setIsDarkMode }) => {
  // State management
  const [currentView, setCurrentView] = useState("modules");
  const [modules, setModules] = useState([]);
  const [selectedModule, setSelectedModule] = useState(null);
  const [moduleContent, setModuleContent] = useState(null);
  const [userProgress, setUserProgress] = useState({
    level: "beginner",
    completedModules: [],
    points: 0,
    streak: 0,
    totalTimeSpent: 0,
    badges: [],
  });
  const [expandedSections, setExpandedSections] = useState({});
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [isLargeScreen, setIsLargeScreen] = useState(true);
  const [loading, setLoading] = useState(false);
  const [selectedRumour, setSelectedRumour] = useState(null);
  const [isRumourModalOpen, setIsRumourModalOpen] = useState(false);
  const [isInfoModalOpen, setIsInfoModalOpen] = useState(false);

  // Handle responsive sidebar default state and load modules
  useEffect(() => {
    const media = window.matchMedia("(min-width: 1024px)");
    const handleMediaChange = (e) => {
      setIsLargeScreen(e.matches);
      setSidebarOpen(e.matches); // open on large screens, closed on small by default
    };
    handleMediaChange(media);
    media.addEventListener("change", handleMediaChange);

    loadModules();

    return () => media.removeEventListener("change", handleMediaChange);
  }, []);

  const loadModules = async () => {
    try {
      setLoading(true);
      console.log("Loading educational modules...");
      const response = await fetch(`${getApiBaseUrl()}/educational/modules`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      console.log("Modules loaded:", data);
      setModules(data.modules || []);
    } catch (error) {
      console.error("Failed to load modules:", error);
      // Set fallback modules if API fails
      setModules([
        {
          id: "red_flags",
          title: "How to Spot Red Flags",
          description: "Learn to identify warning signs in misinformation",
          difficulty_levels: ["beginner", "intermediate", "advanced"],
          estimated_time: "10-15 minutes",
        },
        {
          id: "source_credibility",
          title: "Evaluating Source Credibility",
          description: "Understand how to assess source reliability",
          difficulty_levels: ["beginner", "intermediate", "advanced"],
          estimated_time: "15-20 minutes",
        },
        {
          id: "manipulation_techniques",
          title: "Common Manipulation Techniques",
          description: "Learn about various misinformation techniques",
          difficulty_levels: ["intermediate", "advanced"],
          estimated_time: "20-25 minutes",
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const loadModuleContent = async (moduleId, difficultyLevel = "beginner") => {
    try {
      setLoading(true);
      console.log(`Loading content for ${moduleId} (${difficultyLevel})...`);
      const response = await fetch(
        `${getApiBaseUrl()}/educational/modules/${moduleId}?difficulty_level=${difficultyLevel}`
      );
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const content = await response.json();
      console.log("Module content loaded:", content);
      setModuleContent(content);
      setSelectedModule(moduleId);
      setCurrentView("content");
    } catch (error) {
      console.error("Failed to load module content:", error);
      // Set fallback content
      setModuleContent({
        title: `Educational Module: ${moduleId}`,
        overview: "Learn about misinformation detection",
        learning_objectives: ["Understand basic concepts"],
        content_sections: [],
        practical_tips: [],
        common_mistakes: [],
        difficulty_level: difficultyLevel,
      });
      setSelectedModule(moduleId);
      setCurrentView("content");
    } finally {
      setLoading(false);
    }
  };

  const toggleSection = (sectionId) => {
    setExpandedSections((prev) => ({
      ...prev,
      [sectionId]: !prev[sectionId],
    }));
  };

  const completeModule = (moduleId) => {
    setUserProgress((prev) => {
      // Check if module is already completed
      if (prev.completedModules.includes(moduleId)) {
        return prev; // Return unchanged state if already completed
      }

      // Add module to completed list and award points
      return {
        ...prev,
        completedModules: [...prev.completedModules, moduleId],
        points: prev.points + 10,
      };
    });
  };

  const getModuleIcon = (moduleId) => {
    switch (moduleId) {
      case "red_flags":
        return AlertTriangle;
      case "source_credibility":
        return Shield;
      case "manipulation_techniques":
        return Target;
      default:
        return BookOpen;
    }
  };

  const getPageTitle = () => {
    switch (currentView) {
      case "chatbot":
        return "Project Aegis";
      case "modules":
        return "Educational Modules";
      case "content":
        return moduleContent?.title || "Module Content";
      case "progress":
        return "Your Progress";
      default:
        return "Educational Modules";
    }
  };

  const getPageSubtitle = () => {
    switch (currentView) {
      case "chatbot":
        return "AI-powered verification assistant";
      case "modules":
        return "Choose a module to start learning";
      case "content":
        return "Interactive learning experience";
      case "progress":
        return "Track your learning journey";
      default:
        return "Choose a module to start learning";
    }
  };

  const handleBackToModules = () => {
    if (selectedModule) {
      setSelectedModule(null);
      setModuleContent(null);
      setExpandedSections({});
    }
    setCurrentView("modules");
  };

  const handleFactCheckClick = () => {
    setCurrentView("chatbot");
  };

  const handleLearnClick = () => {
    setCurrentView("modules");
  };

  const handleRumourClick = (rumour) => {
    setSelectedRumour(rumour);
    setIsRumourModalOpen(true);
  };

  const handleCloseRumourModal = () => {
    setIsRumourModalOpen(false);
    setSelectedRumour(null);
  };

  const handleInfoClick = () => {
    setIsInfoModalOpen(true);
  };

  const handleCloseInfoModal = () => {
    setIsInfoModalOpen(false);
  };

  return (
    <motion.div
      className={`h-screen flex overflow-hidden`}
      animate={{
        backgroundColor: isDarkMode ? "#111827" : "#f9fafb",
      }}
      transition={{
        duration: 0.6,
        ease: "easeInOut",
      }}
    >
      {/* Sidebar */}
      <motion.div
        className={`
          ${isLargeScreen ? "relative" : "fixed inset-y-0 left-0 z-40"}
          ${sidebarOpen ? "translate-x-0" : "-translate-x-full"}
          transform transition-transform duration-300 ease-in-out
          ${isLargeScreen ? (sidebarOpen ? "w-80" : "w-16") : "w-80"}
          ${
            isDarkMode
              ? "bg-gray-800 border-gray-700"
              : "bg-white border-gray-200"
          }
          border-r flex flex-col h-full overflow-hidden scrollbar-hide
        `}
        animate={{
          backgroundColor: isDarkMode ? "#1f2937" : "#ffffff",
        }}
        transition={{
          duration: 0.6,
          ease: "easeInOut",
        }}
      >
        <SidebarHeader
          title="Project Aegis"
          subtitle="Misinformation Detection"
          isDarkMode={isDarkMode}
          sidebarOpen={sidebarOpen}
          onToggle={() => setSidebarOpen(!sidebarOpen)}
        />

        {sidebarOpen && (
          <SidebarNavigation
            currentView={currentView}
            setCurrentView={setCurrentView}
            isDarkMode={isDarkMode}
            onFactCheckClick={handleFactCheckClick}
          />
        )}

        {sidebarOpen && (
          <CurrentRumours
            isDarkMode={isDarkMode}
            onRumourClick={handleRumourClick}
          />
        )}
      </motion.div>

      {/* Backdrop for mobile when sidebar open */}
      {!isLargeScreen && sidebarOpen && (
        <div
          className="fixed inset-0 z-30 bg-black bg-opacity-40"
          onClick={() => setSidebarOpen(false)}
          aria-hidden="true"
        />
      )}

      {/* Main Content */}
      <div className="flex-1 flex flex-col min-h-0">
        {/* Header - Always show */}
        <PageHeader
          title={getPageTitle()}
          subtitle={getPageSubtitle()}
          isDarkMode={isDarkMode}
          setIsDarkMode={setIsDarkMode}
          showBackButton={currentView === "content"}
          onBack={handleBackToModules}
          showMenuButton={!isLargeScreen}
          onToggleSidebar={() => setSidebarOpen((prev) => !prev)}
          showInfoButton={true}
          onInfoClick={handleInfoClick}
        />

        {/* Content Area */}
        <div
          className={`flex-1 min-h-0 ${
            currentView === "chatbot" || currentView === "content"
              ? "overflow-y-auto scrollbar-hide"
              : ""
          }`}
        >
          <ViewContainer
            currentView={currentView}
            modules={modules}
            moduleContent={moduleContent}
            selectedModule={selectedModule}
            userProgress={userProgress}
            expandedSections={expandedSections}
            isDarkMode={isDarkMode}
            onModuleClick={loadModuleContent}
            onToggleSection={toggleSection}
            onCompleteModule={completeModule}
            getModuleIcon={getModuleIcon}
            setIsDarkMode={setIsDarkMode}
            onLearnClick={handleLearnClick}
            loading={loading}
          />
        </div>
      </div>

      {/* Rumour Modal - Outside all containers for full-screen display */}
      <RumourModal
        post={selectedRumour}
        isOpen={isRumourModalOpen}
        onClose={handleCloseRumourModal}
        isDarkMode={isDarkMode}
      />

      {/* Info Modal - Outside all containers for full-screen display */}
      <InfoModal
        isOpen={isInfoModalOpen}
        onClose={handleCloseInfoModal}
        isDarkMode={isDarkMode}
      />
    </motion.div>
  );
};

export default MainApp;
