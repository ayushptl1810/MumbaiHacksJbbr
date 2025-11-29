import { useEffect } from "react";
import { Routes, Route, useLocation } from "react-router-dom";
import { AuthProvider } from "./contexts/AuthContext";
import Navbar from "./layouts/Navbar";
import Footer from "./layouts/Footer";
import Home from "./pages/Home/Home";
import Verify from "./pages/Verify/Verify";
import Modules from "./pages/Modules/Modules";
import Login from "./pages/Auth/Login";
import Signup from "./pages/Auth/Signup";
import Subscription from "./pages/Subscription/Subscription";
import Profile from "./pages/Profile/Profile";
import PrivacyPolicy from "./pages/Legal/PrivacyPolicy";
import TermsOfService from "./pages/Legal/TermsOfService";
import ProtectedRoute from "./components/ProtectedRoute";
import "./App.css";

function App() {
  const location = useLocation();
  const footerHiddenRoutes = new Set(["/verify", "/login", "/signup"]);

  useEffect(() => {
    document.documentElement.classList.add("dark");
    document.documentElement.style.colorScheme = "dark";
  }, []);

  // Scroll to top on route change
  useEffect(() => {
    window.scrollTo({ top: 0, left: 0, behavior: "smooth" });
  }, [location.pathname]);

  return (
    <AuthProvider>
      <div className="min-h-screen bg-black text-gray-100 flex flex-col">
        <Navbar />
        <main className="flex-1">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/verify" element={<Verify />} />
            <Route path="/modules" element={<Modules />} />
            <Route path="/modules/:id" element={<Modules />} />
            <Route path="/login" element={<Login />} />
            <Route path="/signup" element={<Signup />} />
            <Route
              path="/subscription"
              element={
                <ProtectedRoute>
                  <Subscription />
                </ProtectedRoute>
              }
            />
            <Route
              path="/profile"
              element={
                <ProtectedRoute>
                  <Profile />
                </ProtectedRoute>
              }
            />
            <Route path="/privacy" element={<PrivacyPolicy />} />
            <Route path="/terms" element={<TermsOfService />} />
          </Routes>
        </main>
        {!footerHiddenRoutes.has(location.pathname) && <Footer />}
      </div>
    </AuthProvider>
  );
}

export default App;
