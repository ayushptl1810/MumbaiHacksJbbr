import axios from "axios";

export const getApiBaseUrl = () => {
  const env = import.meta.env?.VITE_API_BASE_URL;
  if (env && typeof env === "string" && env.trim()) {
    const url = env.replace(/\/$/, "");
    console.log("🌐 Using API base URL from env:", url);
    return url;
  }
  const defaultUrl = "http://127.0.0.1:8000";
  console.log("🌐 Using default API base URL:", defaultUrl);
  return defaultUrl;
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

const baseURL = getApiBaseUrl();
console.log("🔧 Initializing axios with baseURL:", baseURL);

export const apiClient = axios.create({
  baseURL: baseURL,
  withCredentials: true,
  timeout: 30000, // 30 second timeout
});

apiClient.interceptors.request.use(
  (config) => {
    const token =
      typeof window !== "undefined" ? localStorage.getItem("auth_token") : null;
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    const fullUrl = config.baseURL ? `${config.baseURL}${config.url}` : config.url;
    console.log("📤 API Request:", config.method?.toUpperCase(), fullUrl, config.params);
    return config;
  },
  (error) => {
    console.error("❌ Request interceptor error:", error);
    return Promise.reject(error);
  }
);

apiClient.interceptors.response.use(
  (response) => {
    console.log("✅ API Response:", response.config.method?.toUpperCase(), response.config.url, response.status, response.data);
    return response;
  },
  (error) => {
    console.error("❌ API Error:", {
      message: error.message,
      code: error.code,
      url: error.config?.url,
      method: error.config?.method,
      status: error.response?.status,
      data: error.response?.data,
      request: error.request,
    });
    return Promise.reject(error);
  }
);

export const authService = {
  login: (payload) => apiClient.post("/auth/login", payload),
  signup: (payload) => apiClient.post("/auth/signup", payload),
  me: () => apiClient.get("/auth/me"),
};

export const chatService = {
  listSessions: ({ anonymousId, userId }) => {
    const params = {};
    if (anonymousId) params.anonymous_id = anonymousId;
    if (userId) params.user_id = userId;
    return apiClient.get("/chat/sessions", { params });
  },
  upsertSession: (payload) => apiClient.post("/chat/sessions", payload),
  appendMessages: (payload) => apiClient.post("/chat/messages", payload),
  getMessages: (sessionId) =>
    apiClient.get(`/chat/messages/${encodeURIComponent(sessionId)}`),
};
