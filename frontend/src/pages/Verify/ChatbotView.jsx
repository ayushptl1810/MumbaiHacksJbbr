import { useState, useRef, useEffect } from "react";
import { useSearchParams } from "react-router-dom";
import { getApiBaseUrl } from "../../services/api";
import { motion, AnimatePresence } from "framer-motion";
import {
  Send,
  Upload,
  Image,
  Video,
  Mic,
  CheckCircle,
  AlertCircle,
  Clock,
  Sun,
  Moon,
  X,
} from "lucide-react";
import logoImg from "../../assets/logo.png";
import TierLimitModal from "../../components/TierLimitModal";

const ChatbotView = ({
  isDarkMode,
  setIsDarkMode,
  onLearnClick,
  sessionId,
  initialMessages = [],
  onTurnPersist,
  anonymousId,
  userId,
}) => {
  // Initialize inputValue from URL query parameter if present
  const getInitialInputValue = () => {
    if (typeof window !== "undefined") {
      const urlParams = new URLSearchParams(window.location.search);
      const textParam = urlParams.get("text");
      if (textParam && textParam.trim()) {
        try {
          return decodeURIComponent(textParam.trim());
        } catch (e) {
          return textParam.trim();
        }
      }
    }
    return "";
  };

  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState(getInitialInputValue);
  const [isLoading, setIsLoading] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const fileInputRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingError, setRecordingError] = useState("");
  const [recordingSeconds, setRecordingSeconds] = useState(0);
  // idle | initializing | recording | stopping
  const [recordingPhase, setRecordingPhase] = useState("idle");
  const recordingTimerRef = useRef(null);
  const [isDragging, setIsDragging] = useState(false);
  const dropZoneRef = useRef(null);
  const [searchParams, setSearchParams] = useSearchParams();
  const [limitError, setLimitError] = useState(null);

  useEffect(() => {
    // Reset messages when session changes
    if (Array.isArray(initialMessages) && initialMessages.length) {
      setMessages(initialMessages);
    } else {
      // Empty messages for new chat - welcome screen will show
      setMessages([]);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId]);

  // Clear the query parameter from URL after component mounts
  useEffect(() => {
    if (inputValue && window.location.search.includes("text=")) {
      // Clear the query parameter from URL after a delay
      setTimeout(() => {
        const url = new URL(window.location.href);
        url.searchParams.delete("text");
        window.history.replaceState({}, "", url.pathname + url.search);
      }, 1000);
    }
  }, [inputValue]);

  useEffect(() => {
    return () => {
      if (
        mediaRecorderRef.current &&
        mediaRecorderRef.current.state !== "inactive"
      ) {
        try {
          mediaRecorderRef.current.stop();
        } catch (_) {}
      }
      if (recordingTimerRef.current) {
        clearInterval(recordingTimerRef.current);
        recordingTimerRef.current = null;
      }
      setRecordingPhase("idle");
    };
  }, []);

  // Drag and drop handlers
  const handleDragEnter = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (!e.currentTarget.contains(e.relatedTarget)) {
      setIsDragging(false);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) {
      setUploadedFiles((prev) => [...prev, ...files]);
    }
  };

  const handleSendMessage = async () => {
    if (!inputValue.trim() && uploadedFiles.length === 0) return;

    const userMessage = {
      id: `user-${Date.now()}-${Math.random()}`,
      type: "user",
      content: inputValue,
      timestamp: new Date(),
      files: uploadedFiles,
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputValue("");
    setUploadedFiles([]);
    if (fileInputRef.current) fileInputRef.current.value = null;
    setIsLoading(true);

    try {
      // Create FormData to match backend expectations
      const formData = new FormData();

      // Add text input if provided
      if (inputValue.trim()) {
        formData.append("text_input", inputValue.trim());
      }

      // Add files if provided
      if (uploadedFiles.length > 0) {
        uploadedFiles.forEach((file) => {
          formData.append("files", file);
        });
      }

      // Attach identity hints for rate limiting and history
      if (anonymousId) {
        formData.append("anonymous_id", anonymousId);
      }
      if (userId) {
        formData.append("user_id", userId);
      }

      const apiBase = getApiBaseUrl();

      const response = await fetch(`${apiBase}/chatbot/verify`, {
        method: "POST",
        body: formData, // Send FormData instead of JSON
      });

      // Parse response once
      let result;
      try {
        result = await response.json();
      } catch (e) {
        // If JSON parsing fails, treat as generic error
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        throw new Error("Failed to parse response");
      }

      // Check for limit errors (429 status)
      if (!response.ok && response.status === 429 && result.error === "verification_limit_reached") {
        setLimitError({
          tier: result.tier || "Free",
          limits: result.limits || {},
          usage: result.usage || {},
          feature: "verification",
        });
        setIsLoading(false);
        return; // Don't add error message, modal will show
      }

      if (!response.ok) {
        throw new Error(result.error || `HTTP error! status: ${response.status}`);
      }

      console.log("Verification result:", result);

      // Check if backend returned an error
      if (result.error && result.error !== "verification_limit_reached") {
        throw new Error(result.error);
      }

      const aiMessage = {
        id: `ai-${Date.now()}-${Math.random()}`,
        type: "ai",
        content:
          result.message || result.verification_result || "Analysis complete",
        timestamp: new Date(),
        sources: result.sources || [],
        confidence: result.confidence,
        is_misinformation: result.is_misinformation,
      };

      setMessages((prev) => [...prev, aiMessage]);

      if (onTurnPersist && sessionId) {
        try {
          await onTurnPersist(sessionId, userMessage, aiMessage);
        } catch (e) {
          console.error("Failed to persist chat turn", e);
        }
      }
    } catch (error) {
      console.error("Main error:", error);
      const errorMessage = {
        id: `error-${Date.now()}-${Math.random()}`,
        type: "ai",
        content:
          "Sorry, I encountered an error while verifying your claim. Please try again.",
        timestamp: new Date(),
        sources: [],
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileUpload = (e) => {
    const files = Array.from(e.target.files);
    setUploadedFiles((prev) => [...prev, ...files]);
  };

  const removeFile = (index) => {
    setUploadedFiles((prev) => prev.filter((_, i) => i !== index));
  };

  const getFileIcon = (file) => {
    if (file.type.startsWith("image/")) return Image;
    if (file.type.startsWith("video/")) return Video;
    return Upload;
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const startRecording = async () => {
    try {
      setRecordingError("");
      setRecordingPhase("initializing");
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: "audio/webm",
      });
      const chunks = [];
      mediaRecorder.onstart = () => {
        // actual recording started
        setRecordingSeconds(0);
        if (recordingTimerRef.current) {
          clearInterval(recordingTimerRef.current);
        }
        recordingTimerRef.current = setInterval(() => {
          setRecordingSeconds((s) => s + 1);
        }, 1000);
        setRecordingPhase("recording");
      };
      mediaRecorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) chunks.push(e.data);
      };
      mediaRecorder.onstop = async () => {
        try {
          const blob = new Blob(chunks, { type: "audio/webm" });
          const file = new File([blob], `voice-${Date.now()}.webm`, {
            type: "audio/webm",
          });
          const formData = new FormData();
          formData.append("audio", file);
          const apiBase = getApiBaseUrl();
          const resp = await fetch(`${apiBase}/speech-to-text`, {
            method: "POST",
            body: formData,
          });
          if (!resp.ok) {
            const text = await resp.text();
            throw new Error(text || `HTTP ${resp.status}`);
          }
          const data = await resp.json();
          const transcript = (data && data.transcript) || "";
          if (transcript)
            setInputValue((prev) =>
              prev ? `${prev} ${transcript}` : transcript
            );
          else setRecordingError("No speech recognized. Try again.");
        } catch (err) {
          console.error("Speech-to-text error:", err);
          setRecordingError("Speech-to-text failed. Check backend logs.");
        } finally {
          // stop all tracks
          stream.getTracks().forEach((t) => t.stop());
          setIsRecording(false);
          if (recordingTimerRef.current) {
            clearInterval(recordingTimerRef.current);
            recordingTimerRef.current = null;
          }
          setRecordingSeconds(0);
          setRecordingPhase("idle");
        }
      };
      mediaRecorder.start();
      mediaRecorderRef.current = mediaRecorder;
      // show banner immediately in initializing phase
      setIsRecording(true);
    } catch (err) {
      console.error("Mic access/recording error:", err);
      setRecordingError("Microphone access denied or unsupported.");
      setIsRecording(false);
      if (recordingTimerRef.current) {
        clearInterval(recordingTimerRef.current);
        recordingTimerRef.current = null;
      }
      setRecordingPhase("idle");
    }
  };

  const stopRecording = () => {
    try {
      setRecordingPhase("stopping");
      if (
        mediaRecorderRef.current &&
        mediaRecorderRef.current.state !== "inactive"
      ) {
        mediaRecorderRef.current.stop();
      }
    } catch (err) {
      console.error("Stop recording error:", err);
    }
  };

  return (
    <motion.div
      ref={dropZoneRef}
      className={`h-full flex flex-col overflow-hidden relative ${
        isDragging ? "ring-2 ring-blue-500 ring-offset-2" : ""
      }`}
      animate={{
        backgroundColor: isDarkMode ? "#000000" : "#f9fafb",
      }}
      transition={{
        duration: 0.6,
        ease: "easeInOut",
      }}
      onDragEnter={handleDragEnter}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      {/* Drag and Drop Overlay */}
      {isDragging && (
        <div className="absolute inset-0 z-50 bg-blue-600/20 backdrop-blur-sm flex items-center justify-center border-2 border-dashed border-blue-500 rounded-lg">
          <div className="text-center">
            <Upload className="w-12 h-12 text-blue-400 mx-auto mb-2" />
            <p className="text-blue-400 font-semibold">
              Drop files here to upload
            </p>
          </div>
        </div>
      )}
      {/* Messages */}
      <div className="flex-1 overflow-y-auto scrollbar-hide px-4 sm:px-6 py-4 sm:py-6 space-y-4">
        {/* Welcome Screen - Show when no messages */}
        {messages.length === 0 && !isLoading && (
          <motion.div
            className="flex flex-col items-center justify-center h-full pt-24"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <motion.div
              className="mb-8"
              initial={{ scale: 0.9 }}
              animate={{ scale: 1 }}
              transition={{ duration: 0.5, delay: 0.2 }}
            >
              <div className="relative">
                <div className="absolute inset-0 bg-gradient-to-r from-blue-500/20 to-cyan-500/20 blur-2xl rounded-full" />
                <div className="relative w-20 h-20 rounded-full flex items-center justify-center">
                  <img
                    src={logoImg}
                    alt="Project Aegis"
                    className="w-20 h-20 object-contain"
                  />
                </div>
              </div>
            </motion.div>

            <motion.h2
              className="text-3xl font-bold mb-3 text-center"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.3 }}
            >
              <span className="bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
                Project Aegis
              </span>
            </motion.h2>

            <motion.p
              className="text-gray-400 text-center mb-8 max-w-md"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.4 }}
            >
              Your AI-powered fact-checking assistant. Verify text claims,
              analyze images, check videos, and detect deepfakes with
              confidence.
            </motion.p>

            {/* Example Prompts */}
            <motion.div
              className="grid grid-cols-1 md:grid-cols-2 gap-3 w-full max-w-2xl"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5 }}
            >
              {[
                {
                  title: "Verify Text Claims",
                  prompt:
                    "Is it true that the CEO of Astronomer cheated on his wife?",
                  color: "from-blue-500/10 to-blue-600/10",
                  borderColor: "border-blue-500/20",
                },
                {
                  title: "Analyze Images",
                  prompt: "Check if this image is real or AI-generated",
                  color: "from-purple-500/10 to-purple-600/10",
                  borderColor: "border-purple-500/20",
                },
                {
                  title: "Verify Videos",
                  prompt: "Is this video authentic or manipulated?",
                  color: "from-cyan-500/10 to-cyan-600/10",
                  borderColor: "border-cyan-500/20",
                },
                {
                  title: "Audio Deepfake Analysis",
                  prompt:
                    "Check if this audio recording is AI-generated or authentic",
                  color: "from-emerald-500/10 to-emerald-600/10",
                  borderColor: "border-emerald-500/20",
                },
              ].map((example, idx) => {
                return (
                  <motion.button
                    key={idx}
                    onClick={() => setInputValue(example.prompt)}
                    className={`group relative overflow-hidden rounded-xl border ${example.borderColor} bg-gradient-to-br ${example.color} p-4 text-left transition-all hover:border-opacity-40 hover:scale-[1.02] flex flex-col h-full`}
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.6 + idx * 0.1 }}
                  >
                    <div className="flex-1 flex flex-col">
                      <h3 className="text-sm font-semibold text-white mb-2 h-5 flex items-center">
                        {example.title}
                      </h3>
                      <p className="text-xs text-gray-400 line-clamp-2 flex-1">
                        {example.prompt}
                      </p>
                    </div>
                    <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/5 to-transparent -translate-x-full group-hover:translate-x-full transition-transform duration-700" />
                  </motion.button>
                );
              })}
            </motion.div>

            <motion.p
              className="text-xs text-gray-500 mt-8 text-center"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 1 }}
            >
              Click a suggestion above or type your own question
            </motion.p>
          </motion.div>
        )}

        {/* Messages List */}
        {messages.length > 0 && (
          <div className="flex flex-col gap-4">
            {messages.map((message, idx) => {
              const isUser = message.type === "user";
              const alignment = isUser ? "items-end" : "items-start";
              const bubbleAlignment = isUser ? "justify-end" : "justify-start";
              return (
                <div
                  key={message.id}
                  className={`mb-2 flex flex-col ${alignment}`}
                >
                  {/* File preview (aligned and width limited like message) */}
                  {message.files && message.files.length > 0 && (
                    <div className="mb-2 flex flex-wrap gap-2 max-w-[65%]">
                      {message.files.map((file, i) => {
                        const FileIcon = getFileIcon(file);
                        const isImage =
                          file.type && file.type.startsWith("image/");
                        if (isImage) {
                          const objectUrl = URL.createObjectURL(file);
                          return (
                            <div
                              key={i}
                              className="relative overflow-hidden rounded-md border border-gray-200 dark:border-gray-700"
                            >
                              <img
                                src={objectUrl}
                                alt={file.name || `upload-${i}`}
                                className="w-full max-w-xs h-auto object-contain"
                                onLoad={() => URL.revokeObjectURL(objectUrl)}
                              />
                            </div>
                          );
                        }
                        return (
                          <div
                            key={i}
                            className={`flex items-center space-x-2 px-3 py-2 rounded-md border text-xs ${
                              isDarkMode
                                ? "bg-gray-800 border-gray-700 text-gray-200"
                                : "bg-gray-50 border-gray-200 text-gray-700"
                            }`}
                          >
                            <FileIcon className="w-4 h-4 text-blue-500" />
                            <span className="truncate" title={file.name}>
                              {file.name}
                            </span>
                          </div>
                        );
                      })}
                    </div>
                  )}
                  {/* Message bubble below file preview */}
                  <div className={`flex ${bubbleAlignment} max-w-[65%]`}>
                    <div
                      className={`rounded-lg px-4 py-3 shadow-sm break-words w-full whitespace-pre-wrap ${
                        message.type === "ai"
                          ? isDarkMode
                            ? "bg-gray-800 text-gray-100"
                            : "bg-gray-100 text-gray-900"
                          : "bg-blue-500 text-white"
                      }`}
                    >
                      {message.content}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {isLoading && (
          <motion.div
            className="flex justify-start"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.6, ease: "easeInOut" }}
          >
            <div className="flex items-end space-x-3">
              {/* Avatar */}
              <div
                className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
                  isDarkMode ? "bg-gray-700" : "bg-gray-200"
                }`}
              >
                <CheckCircle
                  className={`w-4 h-4 ${
                    isDarkMode ? "text-blue-400" : "text-blue-500"
                  }`}
                />
              </div>

              {/* Loading Message Bubble */}
              <div className="relative max-w-xs lg:max-w-md text-left">
                <div
                  className={`px-4 py-3 rounded-lg ${
                    isDarkMode
                      ? "bg-gray-800 border border-gray-700 text-white"
                      : "bg-white border border-gray-200 text-gray-900"
                  }`}
                >
                  <div className="flex items-center space-x-2">
                    <motion.div
                      className="w-2 h-2 bg-gray-400 rounded-full"
                      animate={{ scale: [1, 1.2, 1] }}
                      transition={{ duration: 0.6, repeat: Infinity, delay: 0 }}
                    ></motion.div>
                    <motion.div
                      className="w-2 h-2 bg-gray-400 rounded-full"
                      animate={{ scale: [1, 1.2, 1] }}
                      transition={{
                        duration: 0.6,
                        repeat: Infinity,
                        delay: 0.1,
                      }}
                    ></motion.div>
                    <motion.div
                      className="w-2 h-2 bg-gray-400 rounded-full"
                      animate={{ scale: [1, 1.2, 1] }}
                      transition={{
                        duration: 0.6,
                        repeat: Infinity,
                        delay: 0.2,
                      }}
                    ></motion.div>
                  </div>
                </div>

                {/* Arrow */}
                <div
                  className={`absolute bottom-3 left-[-8px] w-0 h-0 border-r-8 border-r-gray-800 dark:border-r-gray-800 border-t-8 border-t-transparent border-b-8 border-b-transparent`}
                ></div>
              </div>
            </div>
          </motion.div>
        )}
      </div>

      {/* File Preview */}
      {uploadedFiles.length > 0 && (
        <motion.div
          className="px-4 sm:px-6 py-2"
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: "auto" }}
          exit={{ opacity: 0, height: 0 }}
          transition={{ duration: 0.3, ease: "easeInOut" }}
        >
          <div className="flex flex-wrap gap-2 max-w-[65%]">
            {uploadedFiles.map((file, index) => {
              const FileIcon = getFileIcon(file);
              return (
                <motion.div
                  key={index}
                  className={`flex items-center space-x-2 px-3 py-2 rounded-lg ${
                    isDarkMode
                      ? "bg-gray-800 border border-gray-700"
                      : "bg-gray-100 border border-gray-300"
                  }`}
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.8 }}
                  transition={{ duration: 0.2 }}
                >
                  <FileIcon className="w-4 h-4 text-blue-500" />
                  <span className="text-sm text-gray-600 dark:text-gray-300 max-w-32 truncate">
                    {file.name}
                  </span>
                  <button
                    onClick={() => removeFile(index)}
                    className="text-gray-400 hover:text-red-500 transition-colors"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </motion.div>
              );
            })}
          </div>
        </motion.div>
      )}

      {/* Input Area */}
      <motion.div
        className="border-t border-white/5 px-4 sm:px-8 py-4 backdrop-blur-lg"
        animate={{
          backgroundColor: isDarkMode
            ? "rgba(6,10,20,0.8)"
            : "rgba(255,255,255,0.85)",
        }}
        transition={{
          duration: 0.6,
          ease: "easeInOut",
        }}
      >
        <div className="space-y-3">
          <div className="flex flex-wrap items-center gap-2">
            <input
              type="file"
              id="file-upload"
              ref={fileInputRef}
              onChange={handleFileUpload}
              className="hidden"
              disabled={isRecording}
            />
            <motion.label
              htmlFor="file-upload"
              className="flex h-11 w-11 cursor-pointer items-center justify-center rounded-xl border border-white/10 bg-white/5 text-gray-400 transition hover:text-white"
              onClick={() => {
                if (fileInputRef.current) fileInputRef.current.value = null;
              }}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <Upload className="h-5 w-5" />
            </motion.label>
            <motion.button
              onClick={isRecording ? stopRecording : startRecording}
              className={`flex h-11 w-11 items-center justify-center rounded-xl border ${
                isRecording
                  ? "border-red-500/40 bg-red-500/10 text-red-400"
                  : "border-white/10 bg-white/5 text-gray-400 hover:text-white"
              } transition`}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              title={isRecording ? "Stop recording" : "Start voice note"}
            >
              <Mic
                className={`h-5 w-5 ${isRecording ? "animate-pulse" : ""}`}
              />
            </motion.button>
            {isRecording && (
              <motion.div
                className={`flex items-center gap-3 rounded-2xl border px-3 py-2 text-xs font-medium ${
                  isDarkMode
                    ? "border-red-500/40 bg-red-500/5 text-red-200"
                    : "border-red-500/40 bg-red-50 text-red-600"
                }`}
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
              >
                <div className="flex items-center gap-2">
                  <span className="h-2 w-2 rounded-full bg-current animate-pulse" />
                  {recordingPhase === "initializing" && (
                    <span>Preparing mic…</span>
                  )}
                  {recordingPhase === "recording" && <span>Recording…</span>}
                  {recordingPhase === "stopping" && <span>Processing…</span>}
                </div>
                <span className="font-mono text-[11px]">
                  {recordingPhase === "recording"
                    ? `${String(Math.floor(recordingSeconds / 60)).padStart(
                        2,
                        "0"
                      )}:${String(recordingSeconds % 60).padStart(2, "0")}`
                    : "00:00"}
                </span>
              </motion.div>
            )}
            <div className="ml-auto hidden items-center gap-2 rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-xs text-gray-400 sm:flex">
              <Clock className="h-3.5 w-3.5" />
              <span>Shift + Enter for newline</span>
            </div>
          </div>

          <div className="flex items-end gap-3 rounded-2xl border border-white/10 bg-gradient-to-br from-slate-900/80 to-slate-950/80 px-3 py-2">
            <motion.textarea
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask me to verify something..."
              className={`flex-1 resize-none bg-transparent px-3 py-3 text-base focus:outline-none whitespace-pre-wrap ${
                isDarkMode
                  ? "text-white placeholder-gray-500"
                  : "text-gray-900 placeholder-gray-500"
              }`}
              style={{ minHeight: "56px", maxHeight: "200px" }}
              animate={{
                color: isDarkMode ? "#ffffff" : "#111827",
              }}
              transition={{
                duration: 0.6,
                ease: "easeInOut",
              }}
              disabled={isRecording}
            />
            <motion.button
              onClick={handleSendMessage}
              disabled={
                (!inputValue.trim() && uploadedFiles.length === 0) || isLoading
              }
              className="group relative flex items-center justify-center gap-2 rounded-2xl bg-gradient-to-r from-blue-500 to-cyan-500 px-5 py-3 text-sm font-semibold text-white shadow-lg shadow-blue-900/40 transition disabled:cursor-not-allowed disabled:opacity-50"
              whileHover={{ scale: 1.03 }}
              whileTap={{ scale: 0.97 }}
            >
              <Send className="h-4 w-4 transition group-hover:translate-x-0.5" />
              <span>Send</span>
            </motion.button>
          </div>
        </div>
        {recordingError && (
          <div className="mt-2 text-xs text-red-500">{recordingError}</div>
        )}
      </motion.div>

      {/* Tier Limit Modal */}
      <TierLimitModal
        isOpen={!!limitError}
        onClose={() => setLimitError(null)}
        limitInfo={limitError}
      />
    </motion.div>
  );
};

export default ChatbotView;
