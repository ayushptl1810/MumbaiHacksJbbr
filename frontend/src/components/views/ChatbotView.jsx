import { useState, useRef, useEffect } from "react";
import { getApiBaseUrl } from "../../config/api";
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

const ChatbotView = ({ isDarkMode, setIsDarkMode, onLearnClick }) => {
  const [messages, setMessages] = useState([
    {
      id: 1,
      type: "ai",
      content:
        "Hello! I'm your AI fact-checking assistant. I can verify text claims, analyze images, check videos, and listen to audio for accuracy. What would you like me to verify?",
      timestamp: new Date(),
      sources: [],
    },
  ]);
  const [inputValue, setInputValue] = useState("");
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

      const apiBase = getApiBaseUrl();

      const response = await fetch(`${apiBase}/chatbot/verify`, {
        method: "POST",
        body: formData, // Send FormData instead of JSON
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      console.log("Verification result:", result);

      // Check if backend returned an error
      if (result.error) {
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
      className={`h-full flex flex-col overflow-hidden`}
      animate={{
        backgroundColor: isDarkMode ? "#111827" : "#f9fafb",
      }}
      transition={{
        duration: 0.6,
        ease: "easeInOut",
      }}
    >
      {/* Messages */}
      <div className="flex-1 overflow-y-auto scrollbar-hide px-4 sm:px-6 py-4 sm:py-6 space-y-4">
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
                    className={`rounded-lg px-4 py-3 shadow-sm break-words w-full ${
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

      {/* Recording Banner */}
      {isRecording && (
        <div className="px-4 sm:px-6">
          <div
            className="mt-2 mb-2 flex items-center justify-between rounded-md px-3 py-2 text-white"
            style={{ backgroundColor: "#b91c1c" }}
          >
            <div className="flex items-center space-x-3">
              <div className="w-2 h-2 rounded-full bg-white animate-pulse"></div>
              {recordingPhase === "initializing" && (
                <div className="text-sm">
                  <span className="font-medium">Starting…</span>
                  <span className="opacity-90 ml-2">
                    first 0.5s can be missed
                  </span>
                </div>
              )}
              {recordingPhase === "recording" && (
                <span className="text-sm font-medium">Recording…</span>
              )}
              {recordingPhase === "stopping" && (
                <span className="text-sm font-medium">Processing…</span>
              )}
            </div>
            {recordingPhase === "recording" ? (
              <span className="text-sm tabular-nums">
                {String(Math.floor(recordingSeconds / 60)).padStart(2, "0")}:
                {String(recordingSeconds % 60).padStart(2, "0")}
              </span>
            ) : (
              <span className="text-sm"> </span>
            )}
          </div>
        </div>
      )}

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
        className={`border-t px-4 sm:px-6 py-4 ${
          isDarkMode ? "border-gray-700" : "border-gray-200"
        }`}
        animate={{
          backgroundColor: isDarkMode ? "#1f2937" : "#ffffff",
        }}
        transition={{
          duration: 0.6,
          ease: "easeInOut",
        }}
      >
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
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
              className="p-2 rounded-lg cursor-pointer"
              onClick={() => {
                // ensure selecting the same file again fires onChange
                if (fileInputRef.current) fileInputRef.current.value = null;
              }}
              animate={{
                color: isDarkMode ? "#9ca3af" : "#6b7280",
                backgroundColor: "transparent",
              }}
              whileHover={{
                color: isDarkMode ? "#d1d5db" : "#374151",
                backgroundColor: isDarkMode ? "#374151" : "#f3f4f6",
              }}
              transition={{
                duration: 0.6,
                ease: "easeInOut",
              }}
            >
              <Upload className="w-6 h-6" />
            </motion.label>
            <motion.button
              onClick={isRecording ? stopRecording : startRecording}
              className={`p-2 rounded-lg ${isRecording ? "text-red-500" : ""}`}
              animate={{
                color: isDarkMode ? "#9ca3af" : "#6b7280",
                backgroundColor: "transparent",
              }}
              whileHover={{
                color: isDarkMode ? "#d1d5db" : "#374151",
                backgroundColor: isDarkMode ? "#374151" : "#f3f4f6",
              }}
              transition={{
                duration: 0.6,
                ease: "easeInOut",
              }}
            >
              <Mic
                className={`w-6 h-6 ${isRecording ? "animate-pulse" : ""}`}
              />
            </motion.button>
          </div>
          <motion.textarea
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask me to verify something..."
            className={`flex-1 px-4 py-2 rounded-lg border focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none ${
              isDarkMode ? "placeholder-gray-400" : "placeholder-gray-500"
            }`}
            style={{ height: "45px", minHeight: "45px" }}
            animate={{
              backgroundColor: isDarkMode ? "#1f2937" : "#ffffff",
              borderColor: isDarkMode ? "#374151" : "#d1d5db",
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
            className="px-4 text-white rounded-lg disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-2"
            style={{ height: "45px" }}
            animate={{
              backgroundColor: "#3b82f6",
            }}
            whileHover={{
              backgroundColor: "#2563eb",
              scale: 1.02,
            }}
            whileTap={{ scale: 0.98 }}
            transition={{
              duration: 0.6,
              ease: "easeInOut",
            }}
          >
            <Send className="w-4 h-4" />
            <span>Send</span>
          </motion.button>
        </div>
        {recordingError && (
          <div className="mt-2 text-xs text-red-500">{recordingError}</div>
        )}
      </motion.div>
    </motion.div>
  );
};

export default ChatbotView;
