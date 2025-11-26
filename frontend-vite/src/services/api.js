import axios from "axios";

export const getApiBaseUrl = () => {
  const env = import.meta.env?.VITE_API_BASE_URL;
  if (env && typeof env === "string" && env.trim()) {
    return env.replace(/\/$/, "");
  }
  return import.meta.env?.VITE_API_BASE_URL || "http://127.0.0.1:8000"; // default local backend
};

export const getWsUrl = () => {
  const envWs = import.meta.env?.VITE_WS_URL;
  if (envWs && typeof envWs === "string" && envWs.trim()) {
    return envWs.endsWith("/ws") ? envWs : `${envWs.replace(/\/$/, "")}/ws`;
  }
  const api = getApiBaseUrl();
  const protocol = api.startsWith("https") ? "wss" : "ws";
  const host = api.replace(/^https?:\/\//, "");
  return `${protocol}://${host}/ws`;
};

export const apiClient = axios.create({
  baseURL: getApiBaseUrl(),
  withCredentials: true,
});

apiClient.interceptors.request.use(
  (config) => {
    const token =
      typeof window !== "undefined" ? localStorage.getItem("auth_token") : null;
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

export const authService = {
  login: (payload) => apiClient.post("/auth/login", payload),
  signup: (payload) => apiClient.post("/auth/signup", payload),
  me: () => apiClient.get("/auth/me"),
};
