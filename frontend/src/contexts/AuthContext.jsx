import { createContext, useContext, useState, useEffect } from "react";
import { authService } from "../services/api";

const AuthContext = createContext(null);

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  const refreshUser = async () => {
    try {
      const token = localStorage.getItem("auth_token");
      if (token) {
        const response = await authService.me();
        if (response?.data?.id) {
          setUser(response.data);
          return response.data;
        } else {
          localStorage.removeItem("auth_token");
          setUser(null);
        }
      }
    } catch (error) {
      console.error("Failed to refresh user:", error);
      localStorage.removeItem("auth_token");
      setUser(null);
    }
    return null;
  };

  useEffect(() => {
    // Check if user is logged in on mount
    const checkAuth = async () => {
      try {
        const token = localStorage.getItem("auth_token");
        if (token) {
          const response = await authService.me();
          if (response?.data?.id) {
            setUser(response.data);
          } else {
            localStorage.removeItem("auth_token");
            setUser(null);
          }
        }
      } catch (error) {
        console.error("Auth check failed:", error);
        localStorage.removeItem("auth_token");
        setUser(null);
      } finally {
        setLoading(false);
      }
    };

    checkAuth();
  }, []);

  const login = (userData, token) => {
    localStorage.setItem("auth_token", token);
    setUser(userData);
  };

  const logout = () => {
    localStorage.removeItem("auth_token");
    setUser(null);
  };

  const value = {
    user,
    loading,
    isAuthenticated: !!user,
    login,
    logout,
    refreshUser,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within AuthProvider");
  }
  return context;
};

