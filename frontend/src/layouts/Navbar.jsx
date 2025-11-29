import { useState, useEffect } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
// eslint-disable-next-line no-unused-vars
import { motion, AnimatePresence } from "framer-motion";
import { Menu, X, AlertTriangle, LogOut, UserCircle } from "lucide-react";
import logo from "../assets/logo.png";
import RumoursSidebar from "../components/RumoursSidebar";
import RumourModal from "../components/RumourModal";
import { useRumoursFeed } from "../hooks/useRumoursFeed";
import { useAuth } from "../contexts/AuthContext";

const Navbar = () => {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [rumoursSidebarOpen, setRumoursSidebarOpen] = useState(false);
  const [selectedRumour, setSelectedRumour] = useState(null);
  const [isRumourModalOpen, setIsRumourModalOpen] = useState(false);
  const location = useLocation();
  const navigate = useNavigate();
  const { rumours } = useRumoursFeed();
  const { isAuthenticated, user, logout } = useAuth();

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
    ...(isAuthenticated ? [{ path: "/subscription", label: "Subscription" }] : []),
  ];

  const handleLogout = () => {
    logout();
    navigate("/");
  };

  const handleRumourClick = (rumour) => {
    setSelectedRumour(rumour);
    setIsRumourModalOpen(true);
    setRumoursSidebarOpen(false);
  };

  return (
    <>
      <nav className="sticky top-0 z-50 bg-black/95 backdrop-blur-sm border-b border-gray-900">
        <div className="layout-container">
          <div className="flex items-center justify-between h-14 sm:h-16">
            {/* Logo */}
            <Link to="/" className="flex items-center space-x-2 sm:space-x-3">
              <img src={logo} alt="Logo" className="h-7 w-7 sm:h-8 sm:w-8" />
              <span className="text-lg sm:text-xl font-bold text-white">
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

              {/* Login/Logout Button */}
              {isAuthenticated ? (
                <div className="ml-4 flex items-center gap-3">
                  <button
                    onClick={() => navigate("/profile")}
                    className={`p-2 rounded-lg transition-all duration-200 ${
                      user?.subscription_tier === "Pro"
                        ? "text-cyan-400 hover:bg-cyan-500/20 hover:scale-110"
                        : user?.subscription_tier === "Enterprise"
                        ? "text-purple-400 hover:bg-purple-500/20 hover:scale-110"
                        : "text-gray-400 hover:bg-gray-700 hover:scale-110"
                    }`}
                    title={`Profile - ${user?.subscription_tier || "Free"} Tier`}
                  >
                    <UserCircle className="w-5 h-5" />
                  </button>
                  <button
                    onClick={handleLogout}
                    className="px-4 py-2 bg-gray-700 text-white rounded-lg text-sm font-medium hover:bg-gray-600 transition-colors flex items-center gap-2"
                  >
                    <LogOut className="w-4 h-4" />
                    Logout
                  </button>
                </div>
              ) : (
                <Link
                  to="/login"
                  className="ml-4 px-4 py-2 bg-blue-600 text-white rounded-lg text-sm font-medium hover:bg-blue-700 transition-colors"
                >
                  Login
                </Link>
              )}
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
      </nav>

      {/* Mobile Navigation Overlay (outside nav so it doesn't affect layout) */}
      {mobileMenuOpen && (
        <div className="md:hidden fixed inset-0 z-40 bg-black/95 backdrop-blur-md">
          <div className="pt-14 h-full overflow-y-auto border-t border-gray-900">
            <div className="px-4 py-4 space-y-2">
              <div className="flex items-center justify-between mb-2 text-xs uppercase tracking-[0.18em] text-gray-500">
                <span>Navigation</span>
                <span className="h-px flex-1 ml-3 bg-gradient-to-r from-gray-700 to-transparent" />
              </div>
              {navItems.map((item) => (
                <Link
                  key={item.path}
                  to={item.path}
                  onClick={() => setMobileMenuOpen(false)}
                  className={`block px-4 py-3 rounded-xl text-sm font-medium ${
                    location.pathname === item.path
                      ? "bg-blue-600 text-white"
                      : "text-gray-200 hover:bg-gray-800/80 border border-transparent hover:border-gray-700"
                  }`}
                >
                  {item.label}
                </Link>
              ))}
              {isAuthenticated ? (
                <>
                  <Link
                    to="/profile"
                    onClick={() => setMobileMenuOpen(false)}
                    className="block px-4 py-3 rounded-xl text-sm font-medium text-gray-200 hover:bg-gray-800/80 flex items-center gap-2 border border-transparent hover:border-gray-700"
                  >
                    <UserCircle className="w-4 h-4" />
                    Profile
                  </Link>
                  <button
                    onClick={() => {
                      handleLogout();
                      setMobileMenuOpen(false);
                    }}
                    className="block w-full text-left px-4 py-3 bg-gray-700 text-white rounded-xl text-sm font-medium hover:bg-gray-600 flex items-center gap-2"
                  >
                    <LogOut className="w-4 h-4" />
                    Logout
                  </button>
                </>
              ) : (
                <Link
                  to="/login"
                  onClick={() => setMobileMenuOpen(false)}
                  className="block px-4 py-3 bg-blue-600 text-white rounded-xl text-sm font-medium hover:bg-blue-700 text-center"
                >
                  Login
                </Link>
              )}
            </div>
          </div>
        </div>
      )}

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
