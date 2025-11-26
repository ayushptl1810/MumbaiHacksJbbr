import { motion } from "framer-motion";
import { ArrowLeft, Sun, Moon, Info } from "lucide-react";
import MotionText from "../ui/MotionText";

const PageHeader = ({
  title,
  subtitle,
  isDarkMode,
  setIsDarkMode,
  showBackButton = false,
  onBack,
  showMenuButton = false,
  onToggleSidebar,
  showInfoButton = false,
  onInfoClick,
}) => {
  return (
    <motion.div
      className={`px-4 sm:px-6 py-4 sm:py-6 border-b ${
        isDarkMode ? "border-gray-700" : "border-gray-200"
      }`}
      animate={{
        backgroundColor: isDarkMode ? "#1f2937" : "#ffffff",
        borderColor: isDarkMode ? "#374151" : "#e5e7eb",
      }}
      transition={{
        duration: 0.6,
        ease: "easeInOut",
      }}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          {showMenuButton && (
            <motion.button
              onClick={onToggleSidebar}
              className={`p-2 rounded-lg ${
                isDarkMode ? "hover:bg-gray-700" : "hover:bg-gray-100"
              } lg:hidden`}
              whileHover={{
                backgroundColor: isDarkMode ? "#374151" : "#f3f4f6",
              }}
              transition={{
                duration: 0.6,
                ease: "easeInOut",
              }}
              aria-label="Open menu"
            >
              {/* Simple hamburger icon */}
              <div
                className={`space-y-1 ${
                  isDarkMode ? "text-gray-300" : "text-gray-700"
                }`}
              >
                <div className="w-5 h-0.5 bg-current" />
                <div className="w-5 h-0.5 bg-current" />
                <div className="w-5 h-0.5 bg-current" />
              </div>
            </motion.button>
          )}
          {showBackButton && (
            <motion.button
              onClick={onBack}
              className={`p-2 rounded-lg ${
                isDarkMode ? "hover:bg-gray-700" : "hover:bg-gray-100"
              }`}
              whileHover={{
                backgroundColor: isDarkMode ? "#374151" : "#f3f4f6",
              }}
              transition={{
                duration: 0.6,
                ease: "easeInOut",
              }}
            >
              <ArrowLeft className="w-5 h-5 text-blue-500" />
            </motion.button>
          )}
          <div>
            <MotionText
              className="text-2xl font-bold"
              isDarkMode={isDarkMode}
              darkColor="#ffffff"
              lightColor="#111827"
            >
              {title}
            </MotionText>
            <MotionText
              className="text-sm"
              isDarkMode={isDarkMode}
              darkColor="#9ca3af"
              lightColor="#4b5563"
            >
              {subtitle}
            </MotionText>
          </div>
        </div>
        <div className="flex items-center space-x-4">
          {showInfoButton && (
            <motion.button
              onClick={onInfoClick}
              className={`p-2 rounded-lg ${
                isDarkMode ? "hover:bg-gray-700" : "hover:bg-gray-100"
              }`}
              animate={{
                backgroundColor: isDarkMode ? "#1f2937" : "#ffffff",
              }}
              whileHover={{
                backgroundColor: isDarkMode ? "#374151" : "#f3f4f6",
              }}
              transition={{
                duration: 0.6,
                ease: "easeInOut",
              }}
              aria-label="Show information"
            >
              <motion.div
                animate={{
                  color: isDarkMode ? "#9ca3af" : "#6b7280",
                }}
                transition={{
                  duration: 0.6,
                  ease: "easeInOut",
                }}
              >
                <Info className="w-5 h-5" />
              </motion.div>
            </motion.button>
          )}
          <motion.button
            onClick={() => setIsDarkMode(!isDarkMode)}
            className={`p-2 rounded-lg ${
              isDarkMode ? "hover:bg-gray-700" : "hover:bg-gray-100"
            }`}
            animate={{
              backgroundColor: isDarkMode ? "#1f2937" : "#ffffff",
            }}
            whileHover={{
              backgroundColor: isDarkMode ? "#374151" : "#f3f4f6",
            }}
            transition={{
              duration: 0.6,
              ease: "easeInOut",
            }}
          >
            <motion.div
              animate={{
                color: isDarkMode ? "#fbbf24" : "#374151",
              }}
              transition={{
                duration: 0.6,
                ease: "easeInOut",
              }}
            >
              {isDarkMode ? (
                <Sun className="w-5 h-5" />
              ) : (
                <Moon className="w-5 h-5" />
              )}
            </motion.div>
          </motion.button>
        </div>
      </div>
    </motion.div>
  );
};

export default PageHeader;
