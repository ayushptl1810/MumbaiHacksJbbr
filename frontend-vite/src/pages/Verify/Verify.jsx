import { useState, useMemo, useRef, useEffect } from "react";
import { motion } from "framer-motion";
import { PenSquare, Search } from "lucide-react";
import logoImg from "../../assets/logo.png";
import ChatbotView from "./ChatbotView";
import "./Verify.css";
import { chatService, authService } from "../../services/api";

const Verify = () => {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [chatHistory, setChatHistory] = useState([]);
  const [activeChatId, setActiveChatId] = useState(null);
  const [messagesBySession, setMessagesBySession] = useState({});
  const [anonymousId, setAnonymousId] = useState(null);
  const [userId, setUserId] = useState(null);
  const hoverTimeoutRef = useRef(null);

  useEffect(() => {
    // Establish anonymous id for this browser
    if (typeof window !== "undefined") {
      const existing =
        window.localStorage.getItem("aegis_anonymous_id") || null;
      if (existing) {
        setAnonymousId(existing);
      } else {
        const generated = crypto.randomUUID
          ? crypto.randomUUID()
          : Math.random().toString(36).slice(2);
        window.localStorage.setItem("aegis_anonymous_id", generated);
        setAnonymousId(generated);
      }
    }
  }, []);

  useEffect(() => {
    // Try to discover logged-in user (best effort, backend is mock)
    const loadUser = async () => {
      try {
        const res = await authService.me();
        if (res?.data?.id) {
          setUserId(res.data.id);
        }
      } catch {
        // ignore, anonymous session is fine
      }
    };
    loadUser();
  }, []);

  useEffect(() => {
    const loadSessions = async () => {
      // Only load sessions for logged-in users (anonymous sessions aren't persisted)
      if (!userId) {
        console.log(
          "⏭️ No userId, skipping session load (anonymous sessions not persisted)"
        );
        setChatHistory([]);
        return;
      }
      try {
        console.log("🔍 Loading chat sessions for user:", userId);
        const res = await chatService.listSessions({
          anonymousId: null, // Don't send anonymousId for logged-in users
          userId,
        });
        console.log("✅ Chat sessions response:", res?.data);
        const sessions = res?.data?.sessions || [];
        const mapped = sessions.map((s) => ({
          id: s.session_id,
          title: s.title || "New Chat",
          timestamp: s.updated_at ? new Date(s.updated_at) : new Date(),
        }));
        setChatHistory(mapped);
        if (!activeChatId && mapped.length > 0) {
          setActiveChatId(mapped[0].id);
        }
      } catch (e) {
        console.error("❌ Failed to load chat sessions", e);
        console.error("Error details:", {
          message: e.message,
          response: e.response?.data,
          status: e.response?.status,
          config: e.config,
        });
      }
    };
    loadSessions();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [userId]); // Only depend on userId, not anonymousId

  const handleNewChat = async () => {
    try {
      const res = await chatService.upsertSession({
        title: "New Chat",
        anonymous_id: anonymousId,
        user_id: userId,
      });
      const session = res.data;
      const id = session.session_id;
      const newEntry = {
        id,
        title: session.title || "New Chat",
        timestamp: session.updated_at
          ? new Date(session.updated_at)
          : new Date(),
      };
      setChatHistory((prev) => [newEntry, ...prev]);
      setActiveChatId(id);
      setMessagesBySession((prev) => ({
        ...prev,
        [id]: [],
      }));
    } catch (e) {
      console.error("Failed to create chat session", e);
    }
  };

  const filteredHistory = useMemo(() => chatHistory, [chatHistory]);

  useEffect(() => {
    return () => {
      if (hoverTimeoutRef.current) {
        clearTimeout(hoverTimeoutRef.current);
      }
    };
  }, []);

  const handleSelectChat = async (sessionId) => {
    setActiveChatId(sessionId);
    if (messagesBySession[sessionId]) return;
    try {
      const res = await chatService.getMessages(sessionId);
      const raw = res?.data?.messages || [];
      const mapped = raw.map((m) => ({
        id: m._id,
        type: m.role === "assistant" ? "ai" : "user",
        content: m.content,
        timestamp: m.created_at ? new Date(m.created_at) : new Date(),
        sources: m.sources || [],
      }));
      setMessagesBySession((prev) => ({
        ...prev,
        [sessionId]: mapped,
      }));
    } catch (e) {
      console.error("Failed to load chat messages", e);
    }
  };

  const handlePersistTurn = async (sessionId, userMessage, aiMessage) => {
    // Always update local state for UI
    setMessagesBySession((prev) => {
      const existing = prev[sessionId] || [];
      return {
        ...prev,
        [sessionId]: [...existing, userMessage, aiMessage],
      };
    });

    // Only persist to MongoDB for logged-in users
    if (!userId) {
      console.log("⏭️ Skipping persistence for anonymous user");
      return;
    }

    try {
      await chatService.appendMessages({
        session_id: sessionId,
        user_id: userId,
        anonymous_id: anonymousId,
        messages: [
          {
            role: "user",
            content: userMessage.content,
            created_at: userMessage.timestamp,
          },
          {
            role: "assistant",
            content: aiMessage.content,
            verdict: aiMessage.is_misinformation ? "false" : undefined,
            sources: aiMessage.sources || [],
            created_at: aiMessage.timestamp,
          },
        ],
      });

      // Also bump session metadata
      await chatService.upsertSession({
        session_id: sessionId,
        title: userMessage.content?.slice(0, 80) || "New Chat",
        anonymous_id: anonymousId,
        user_id: userId,
        last_summary: aiMessage.content,
        last_verdict: aiMessage.is_misinformation ? "false" : undefined,
      });
    } catch (e) {
      console.error("Failed to persist chat turn", e);
    }
  };

  return (
    <div className="h-[calc(100vh-4rem)] bg-black">
      <div className="flex h-full w-full overflow-hidden">
        <motion.aside
          onMouseEnter={() => {
            if (hoverTimeoutRef.current) {
              clearTimeout(hoverTimeoutRef.current);
            }
            setSidebarOpen(true);
          }}
          onMouseLeave={() => {
            if (hoverTimeoutRef.current) {
              clearTimeout(hoverTimeoutRef.current);
            }
            hoverTimeoutRef.current = setTimeout(() => {
              setSidebarOpen(false);
            }, 200);
          }}
          initial={{ width: 84 }}
          animate={{ width: sidebarOpen ? 300 : 84 }}
          transition={{ duration: 0.3, ease: "easeInOut" }}
          className="flex h-full flex-col border-r border-white/5 bg-gradient-to-b from-[#0b111c] via-[#060a13] to-black/80 p-3"
        >
          <div className="flex flex-col gap-3">
            <div className="group grid grid-cols-[36px_1fr] items-center rounded-2xl px-3 py-2 text-xs font-semibold text-gray-300 transition-all duration-200">
              <div className="flex h-9 w-9 items-center justify-center rounded-xl">
                <img
                  src={logoImg}
                  alt="Project Aegis"
                  className="h-9 w-9 object-contain"
                />
              </div>
              <div
                className={`ml-3 origin-left overflow-hidden text-left transition-all duration-200 ease-out ${
                  sidebarOpen
                    ? "max-w-[160px] opacity-100 scale-100"
                    : "max-w-0 opacity-0 scale-95"
                }`}
              >
                <span className="text-sm font-semibold whitespace-nowrap transition-colors duration-200 group-hover:text-white">
                  Project Aegis
                </span>
              </div>
            </div>

            <div
              onClick={handleNewChat}
              className="group grid cursor-pointer grid-cols-[36px_1fr] items-center rounded-2xl px-3 py-2 text-xs font-semibold text-gray-300 transition-all duration-200 hover:text-white"
              title="New chat"
            >
              <div className="flex h-9 w-9 items-center justify-center rounded-xl">
                <PenSquare className="h-6 w-6 text-gray-300 transition-colors duration-200 group-hover:text-white" />
              </div>
              <div
                className={`ml-3 origin-left overflow-hidden text-left transition-all duration-200 ease-out ${
                  sidebarOpen
                    ? "max-w-[120px] opacity-100 scale-100"
                    : "max-w-0 opacity-0 scale-95"
                }`}
              >
                <span className="whitespace-nowrap transition-colors duration-200 group-hover:text-white">
                  New Chat
                </span>
              </div>
            </div>

            <button
              onClick={() => setSidebarOpen(true)}
              className="group grid grid-cols-[36px_1fr] items-center rounded-2xl px-3 py-2 text-xs font-semibold text-gray-300 transition-all duration-200 hover:text-white text-left"
              title="Search chats"
            >
              <div className="flex h-9 w-9 items-center justify-center rounded-xl">
                <Search className="h-6 w-6 text-gray-300 transition-colors duration-200 group-hover:text-white" />
              </div>
              <div
                className={`ml-3 origin-left overflow-hidden text-left transition-all duration-200 ease-out ${
                  sidebarOpen
                    ? "max-w-[120px] opacity-100 scale-100"
                    : "max-w-0 opacity-0 scale-95"
                }`}
              >
                <span className="whitespace-nowrap transition-colors duration-200 group-hover:text-white">
                  Search Chats
                </span>
              </div>
            </button>
          </div>

          <motion.div
            className="my-4 h-px bg-white/10"
            initial={false}
            animate={{ width: sidebarOpen ? "100%" : "24px" }}
            transition={{ duration: 0.25, ease: "easeInOut" }}
          />

          {sidebarOpen && (
            <div className="flex-1 space-y-2 overflow-y-auto pr-1">
              {filteredHistory.map((chat) => {
                const isActive = activeChatId === chat.id;
                return (
                  <button
                    key={chat.id}
                    onClick={() => handleSelectChat(chat.id)}
                    className={`flex w-full items-center rounded-2xl border border-transparent px-3 py-3 text-left text-sm transition ${
                      isActive
                        ? "border-blue-500/40 bg-blue-600/20 text-white shadow-inner"
                        : "text-gray-300 hover:border-white/10 hover:bg-white/5"
                    }`}
                  >
                    <div className="flex flex-col">
                      <p className="truncate text-sm font-medium">
                        {chat.title}
                      </p>
                      <p className="text-xs text-gray-500">
                        {chat.timestamp.toLocaleTimeString([], {
                          hour: "2-digit",
                          minute: "2-digit",
                        })}
                      </p>
                    </div>
                  </button>
                );
              })}
            </div>
          )}
        </motion.aside>

        <div className="flex-1 overflow-hidden bg-black/30 backdrop-blur-sm">
          <ChatbotView
            key={activeChatId || "default"}
            isDarkMode={true}
            setIsDarkMode={() => {}}
            sessionId={activeChatId}
            initialMessages={
              (activeChatId && messagesBySession[activeChatId]) || []
            }
            onTurnPersist={handlePersistTurn}
          />
        </div>
      </div>
    </div>
  );
};

export default Verify;
