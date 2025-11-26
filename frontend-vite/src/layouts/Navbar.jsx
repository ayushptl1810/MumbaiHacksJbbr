import { useState, useEffect } from "react";
import { Link, useLocation } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import { Menu, X, AlertTriangle } from "lucide-react";
import logo from "../assets/logo.png";
import RumoursSidebar from "../components/RumoursSidebar";
import RumourModal from "../components/RumourModal";
import { useRumoursFeed } from "../hooks/useRumoursFeed";

const Navbar = () => {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [rumoursSidebarOpen, setRumoursSidebarOpen] = useState(false);
  const [selectedRumour, setSelectedRumour] = useState(null);
  const [isRumourModalOpen, setIsRumourModalOpen] = useState(false);
  const location = useLocation();
  const { rumours } = useRumoursFeed();

  useEffect(() => {
    if (rumoursSidebarOpen) {
      const originalOverflow = document.body.style.overflow;
      document.body.style.overflow = "hidden";
      return () => {
        document.body.style.overflow = originalOverflow;
      };
    } else {
      document.body.style.overflow = "";
    }
  }, [rumoursSidebarOpen]);

  const navItems = [
    { path: "/", label: "Home" },
    { path: "/verify", label: "Verify" },
    { path: "/modules", label: "Modules" },
    { path: "/subscription", label: "Subscription" },
  ];

  const handleRumourClick = (rumour) => {
    setSelectedRumour(rumour);
    setIsRumourModalOpen(true);
    setRumoursSidebarOpen(false);
  };

  return (
    <>
      <nav className="sticky top-0 z-50 bg-black/95 backdrop-blur-sm border-b border-gray-900">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            {/* Logo */}
            <Link to="/" className="flex items-center space-x-3">
              <img src={logo} alt="Logo" className="h-8 w-8" />
              <span className="text-xl font-bold text-white">
                Project Aegis
              </span>
            </Link>

            {/* Desktop Navigation */}
            <div className="hidden md:flex items-center space-x-1">
              {navItems.map((item) => {
                const isActive = location.pathname === item.path;
                return (
                  <Link
                    key={item.path}
                    to={item.path}
                    className={`relative px-4 py-2 text-sm font-medium transition-colors ${
                      isActive ? "text-white" : "text-gray-300 hover:text-white"
                    }`}
                  >
                    {item.label}
                    {/* White line below - always visible if active, on hover if not active */}
                    <motion.div
                      className="absolute bottom-0 left-0 right-0 h-0.5 bg-white"
                      initial={{ scaleX: 0 }}
                      animate={{ scaleX: isActive ? 1 : 0 }}
                      whileHover={{ scaleX: isActive ? 1 : 1 }}
                      transition={{ duration: 0.2 }}
                    />
                  </Link>
                );
              })}

              {/* Rumours Notification Icon */}
              <div className="relative ml-2">
                <button
                  onClick={() => setRumoursSidebarOpen(!rumoursSidebarOpen)}
                  className="relative p-2 rounded-lg text-gray-300 hover:bg-gray-700 hover:text-white transition-colors"
                >
                  <AlertTriangle className="w-5 h-5" />
                  {rumours.length > 0 && (
                    <span className="absolute top-0 right-0 w-2 h-2 bg-red-500 rounded-full"></span>
                  )}
                </button>

                {/* sidebar handled outside nav */}
              </div>

              {/* Login Button */}
              <Link
                to="/login"
                className="ml-4 px-4 py-2 bg-blue-600 text-white rounded-lg text-sm font-medium hover:bg-blue-700 transition-colors"
              >
                Login
              </Link>
            </div>

            {/* Mobile menu button */}
            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="md:hidden p-2 rounded-lg text-gray-300 hover:bg-gray-700"
            >
              {mobileMenuOpen ? (
                <X className="w-6 h-6" />
              ) : (
                <Menu className="w-6 h-6" />
              )}
            </button>
          </div>
        </div>

        {/* Mobile Navigation */}
        <AnimatePresence>
          {mobileMenuOpen && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              className="md:hidden border-t border-gray-900 bg-black"
            >
              <div className="px-4 py-2 space-y-1">
                {navItems.map((item) => (
                  <Link
                    key={item.path}
                    to={item.path}
                    onClick={() => setMobileMenuOpen(false)}
                    className={`block px-4 py-2 rounded-lg text-sm font-medium ${
                      location.pathname === item.path
                        ? "bg-blue-600 text-white"
                        : "text-gray-300 hover:bg-gray-700"
                    }`}
                  >
                    {item.label}
                  </Link>
                ))}
                <Link
                  to="/login"
                  onClick={() => setMobileMenuOpen(false)}
                  className="block px-4 py-2 bg-blue-600 text-white rounded-lg text-sm font-medium hover:bg-blue-700"
                >
                  Login
                </Link>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </nav>

      {/* Rumours Sidebar */}
      <AnimatePresence>
        {rumoursSidebarOpen && (
          <>
            <motion.div
              key="rumour-overlay"
              initial={{ opacity: 0 }}
              animate={{ opacity: 0.4 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 z-50 bg-black"
              onClick={() => setRumoursSidebarOpen(false)}
            />
            <RumoursSidebar
              key="rumour-sidebar"
              rumours={rumours}
              onRumourClick={handleRumourClick}
              onClose={() => setRumoursSidebarOpen(false)}
            />
          </>
        )}
      </AnimatePresence>

      {/* Rumour Modal */}
      <RumourModal
        post={selectedRumour}
        isOpen={isRumourModalOpen}
        onClose={() => {
          setIsRumourModalOpen(false);
          setSelectedRumour(null);
        }}
        isDarkMode={true}
      />
    </>
  );
};

export default Navbar;
