import { useState, useMemo, useRef, useEffect } from "react";
import { motion } from "framer-motion";
import { PenSquare, Search } from "lucide-react";
import logoImg from "../../assets/logo.png";
import ChatbotView from "./ChatbotView";
import "./Verify.css";

const Verify = () => {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [chatHistory, setChatHistory] = useState([
    { id: 1, title: "New Chat", timestamp: new Date() },
  ]);
  const [activeChatId, setActiveChatId] = useState(1);
  const hoverTimeoutRef = useRef(null);

  const handleNewChat = () => {
    const newChat = {
      id: Date.now(),
      title: "New Chat",
      timestamp: new Date(),
    };
    setChatHistory([newChat, ...chatHistory]);
    setActiveChatId(newChat.id);
  };

  const filteredHistory = useMemo(() => chatHistory, [chatHistory]);

  useEffect(() => {
    return () => {
      if (hoverTimeoutRef.current) {
        clearTimeout(hoverTimeoutRef.current);
      }
    };
  }, []);

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
                    onClick={() => setActiveChatId(chat.id)}
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
          <ChatbotView isDarkMode={true} setIsDarkMode={() => {}} />
        </div>
      </div>
    </div>
  );
};

export default Verify;
