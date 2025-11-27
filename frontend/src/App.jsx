import { useState, useEffect } from "react";
import { Routes, Route, useLocation } from "react-router-dom";
import Navbar from "./layouts/Navbar";
import Footer from "./layouts/Footer";
import Home from "./pages/Home/Home";
import Verify from "./pages/Verify/Verify";
import Modules from "./pages/Modules/Modules";
import Login from "./pages/Auth/Login";
import Signup from "./pages/Auth/Signup";
import Subscription from "./pages/Subscription/Subscription";
import "./App.css";

function App() {
  const [isDarkMode] = useState(true); // Always dark theme
  const location = useLocation();
  const footerHiddenRoutes = new Set(["/verify"]);

  useEffect(() => {
    document.documentElement.classList.add("dark");
    document.documentElement.style.colorScheme = "dark";
  }, []);

  return (
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
          <Route path="/subscription" element={<Subscription />} />
        </Routes>
      </main>
      {!footerHiddenRoutes.has(location.pathname) && <Footer />}
    </div>
  );
}

export default App;
