import { motion, AnimatePresence } from "framer-motion";
import {
  X,
  CheckCircle,
  Image,
  Video,
  Mic,
  MessageSquare,
  BookOpen,
  AlertTriangle,
  Shield,
  Zap,
} from "lucide-react";
import logo from "../../assets/logo.png";

const InfoModal = ({ isOpen, onClose, isDarkMode }) => {
  if (!isOpen) return null;

  return (
    <AnimatePresence>
      <motion.div
        className="fixed inset-0 z-50 flex items-center justify-center p-4"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        transition={{ duration: 0.3 }}
      >
        {/* Backdrop */}
        <motion.div
          className="absolute inset-0 bg-black bg-opacity-50"
          onClick={onClose}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
        />

        {/* Modal */}
        <motion.div
          className={`relative w-full max-w-4xl max-h-[90vh] overflow-y-auto scrollbar-hide rounded-lg shadow-xl ${
            isDarkMode ? "bg-gray-800" : "bg-white"
          }`}
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 0.9, opacity: 0 }}
          transition={{ duration: 0.3 }}
        >
          {/* Header */}
          <div
            className={`flex items-center justify-between p-6 border-b ${
              isDarkMode ? "border-gray-700" : "border-gray-200"
            }`}
          >
            <div className="flex items-center space-x-3">
              <div>
                <img src={logo} alt="Project Aegis" className="w-16 h-16" />
              </div>
              <div>
                <h2
                  className={`text-xl font-bold ${
                    isDarkMode ? "text-white" : "text-gray-900"
                  }`}
                >
                  Project Aegis - Fact Check Assistant
                </h2>
                <p
                  className={`text-sm ${
                    isDarkMode ? "text-gray-400" : "text-gray-600"
                  }`}
                >
                  Your AI-powered misinformation detection system
                </p>
              </div>
            </div>
            <motion.button
              onClick={onClose}
              className={`p-2 rounded-lg ${
                isDarkMode ? "hover:bg-gray-700" : "hover:bg-gray-100"
              }`}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <X
                className={`w-5 h-5 ${
                  isDarkMode ? "text-gray-400" : "text-gray-600"
                }`}
              />
            </motion.button>
          </div>

          {/* Content */}
          <div className="p-6 space-y-8">
            {/* What We Do */}
            <section>
              <h3
                className={`text-lg font-semibold mb-4 flex items-center space-x-2 ${
                  isDarkMode ? "text-white" : "text-gray-900"
                }`}
              >
                <Zap className="w-5 h-5 text-yellow-500" />
                <span>What We Do</span>
              </h3>
              <p
                className={`text-sm leading-relaxed ${
                  isDarkMode ? "text-gray-300" : "text-gray-700"
                }`}
              >
                Project Aegis is an AI-powered fact-checking system that helps
                you verify information across multiple formats. Whether it's
                text claims, images, videos, or audio recordings, our system
                analyzes content and provides evidence-based verification
                results to help you distinguish between truth and
                misinformation.
              </p>
            </section>

            {/* Verification Capabilities */}
            <section>
              <h3
                className={`text-lg font-semibold mb-4 flex items-center space-x-2 ${
                  isDarkMode ? "text-white" : "text-gray-900"
                }`}
              >
                <CheckCircle className="w-5 h-5 text-green-500" />
                <span>What You Can Verify</span>
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Text Verification */}
                <div
                  className={`p-4 rounded-lg border ${
                    isDarkMode
                      ? "bg-gray-700 border-gray-600"
                      : "bg-gray-50 border-gray-200"
                  }`}
                >
                  <div className="flex items-center space-x-3 mb-2">
                    <MessageSquare className="w-5 h-5 text-blue-500" />
                    <h4
                      className={`font-medium ${
                        isDarkMode ? "text-white" : "text-gray-900"
                      }`}
                    >
                      Text Claims
                    </h4>
                  </div>
                  <p
                    className={`text-sm ${
                      isDarkMode ? "text-gray-300" : "text-gray-600"
                    }`}
                  >
                    Verify written statements, news articles, social media
                    posts, and any text-based claims using our advanced
                    fact-checking database and AI analysis.
                  </p>
                </div>

                {/* Image Verification */}
                <div
                  className={`p-4 rounded-lg border ${
                    isDarkMode
                      ? "bg-gray-700 border-gray-600"
                      : "bg-gray-50 border-gray-200"
                  }`}
                >
                  <div className="flex items-center space-x-3 mb-2">
                    <Image className="w-5 h-5 text-green-500" />
                    <h4
                      className={`font-medium ${
                        isDarkMode ? "text-white" : "text-gray-900"
                      }`}
                    >
                      Images
                    </h4>
                  </div>
                  <p
                    className={`text-sm ${
                      isDarkMode ? "text-gray-300" : "text-gray-600"
                    }`}
                  >
                    Upload images to check if they're authentic, detect
                    deepfakes, verify context, and find reverse image matches
                    across the web.
                  </p>
                </div>

                {/* Video Verification */}
                <div
                  className={`p-4 rounded-lg border ${
                    isDarkMode
                      ? "bg-gray-700 border-gray-600"
                      : "bg-gray-50 border-gray-200"
                  }`}
                >
                  <div className="flex items-center space-x-3 mb-2">
                    <Video className="w-5 h-5 text-purple-500" />
                    <h4
                      className={`font-medium ${
                        isDarkMode ? "text-white" : "text-gray-900"
                      }`}
                    >
                      Videos & Links
                    </h4>
                  </div>
                  <p
                    className={`text-sm ${
                      isDarkMode ? "text-gray-300" : "text-gray-600"
                    }`}
                  >
                    Verify videos by uploading or pasting a link from YouTube,
                    Instagram, TikTok, Twitter, and other platforms. We analyze
                    video content frame-by-frame and check against our
                    verification database.
                  </p>
                </div>

                {/* Audio Verification */}
                <div
                  className={`p-4 rounded-lg border ${
                    isDarkMode
                      ? "bg-gray-700 border-gray-600"
                      : "bg-gray-50 border-gray-200"
                  }`}
                >
                  <div className="flex items-center space-x-3 mb-2">
                    <Mic className="w-5 h-5 text-orange-500" />
                    <h4
                      className={`font-medium ${
                        isDarkMode ? "text-white" : "text-gray-900"
                      }`}
                    >
                      Audio Recordings
                    </h4>
                  </div>
                  <p
                    className={`text-sm ${
                      isDarkMode ? "text-gray-300" : "text-gray-600"
                    }`}
                  >
                    Record or upload audio clips to verify speech authenticity,
                    detect AI-generated voices, and analyze audio content for
                    accuracy.
                  </p>
                </div>
              </div>
            </section>

            {/* How It Works */}
            <section>
              <h3
                className={`text-lg font-semibold mb-4 flex items-center space-x-2 ${
                  isDarkMode ? "text-white" : "text-gray-900"
                }`}
              >
                <Shield className="w-5 h-5 text-blue-500" />
                <span>How Our Verification Works</span>
              </h3>
              <div className="space-y-4">
                <div
                  className={`p-4 rounded-lg ${
                    isDarkMode ? "bg-blue-900 bg-opacity-30" : "bg-blue-50"
                  }`}
                >
                  <h4
                    className={`font-medium mb-2 ${
                      isDarkMode ? "text-blue-300" : "text-blue-800"
                    }`}
                  >
                    1. Multi-Source Analysis
                  </h4>
                  <p
                    className={`text-sm ${
                      isDarkMode ? "text-blue-200" : "text-blue-700"
                    }`}
                  >
                    We cross-reference your content against multiple
                    fact-checking databases, news sources, and verification
                    platforms to ensure comprehensive analysis.
                  </p>
                </div>
                <div
                  className={`p-4 rounded-lg ${
                    isDarkMode ? "bg-green-900 bg-opacity-30" : "bg-green-50"
                  }`}
                >
                  <h4
                    className={`font-medium mb-2 ${
                      isDarkMode ? "text-green-300" : "text-green-800"
                    }`}
                  >
                    2. AI-Powered Detection
                  </h4>
                  <p
                    className={`text-sm ${
                      isDarkMode ? "text-green-200" : "text-green-700"
                    }`}
                  >
                    Advanced AI algorithms analyze content for manipulation,
                    inconsistencies, and authenticity markers that human
                    reviewers might miss.
                  </p>
                </div>
                <div
                  className={`p-4 rounded-lg ${
                    isDarkMode ? "bg-purple-900 bg-opacity-30" : "bg-purple-50"
                  }`}
                >
                  <h4
                    className={`font-medium mb-2 ${
                      isDarkMode ? "text-purple-300" : "text-purple-800"
                    }`}
                  >
                    3. Evidence-Based Results
                  </h4>
                  <p
                    className={`text-sm ${
                      isDarkMode ? "text-purple-200" : "text-purple-700"
                    }`}
                  >
                    Every verification comes with detailed evidence, source
                    links, and confidence scores so you can understand how we
                    reached our conclusion.
                  </p>
                </div>
              </div>
            </section>

            {/* Current Rumours */}
            <section>
              <h3
                className={`text-lg font-semibold mb-4 flex items-center space-x-2 ${
                  isDarkMode ? "text-white" : "text-gray-900"
                }`}
              >
                <AlertTriangle className="w-5 h-5 text-red-500" />
                <span>Current Rumours Monitoring</span>
              </h3>
              <div
                className={`p-4 rounded-lg border ${
                  isDarkMode
                    ? "bg-gray-700 border-gray-600"
                    : "bg-gray-50 border-gray-200"
                }`}
              >
                <p
                  className={`text-sm leading-relaxed ${
                    isDarkMode ? "text-gray-300" : "text-gray-700"
                  }`}
                >
                  Our system continuously monitors social media platforms, news
                  sources, and online communities to identify trending
                  misinformation. When we detect suspicious content, we
                  automatically verify it and provide real-time fact-checking
                  results. You can view the latest debunked rumours in the
                  sidebar to stay informed about current misinformation
                  campaigns.
                </p>
              </div>
            </section>

            {/* Educational Modules */}
            <section>
              <h3
                className={`text-lg font-semibold mb-4 flex items-center space-x-2 ${
                  isDarkMode ? "text-white" : "text-gray-900"
                }`}
              >
                <BookOpen className="w-5 h-5 text-indigo-500" />
                <span>Educational Modules</span>
              </h3>
              <div
                className={`p-4 rounded-lg border ${
                  isDarkMode
                    ? "bg-gray-700 border-gray-600"
                    : "bg-gray-50 border-gray-200"
                }`}
              >
                <p
                  className={`text-sm leading-relaxed mb-3 ${
                    isDarkMode ? "text-gray-300" : "text-gray-700"
                  }`}
                >
                  Learn how to identify misinformation yourself with our
                  interactive educational modules. These modules cover essential
                  topics like:
                </p>
                <ul
                  className={`text-sm space-y-1 ml-4 ${
                    isDarkMode ? "text-gray-300" : "text-gray-600"
                  }`}
                >
                  <li>• How to spot fake news and misleading headlines</li>
                  <li>• Understanding deepfakes and manipulated media</li>
                  <li>• Critical thinking skills for the digital age</li>
                  <li>• Source verification and fact-checking techniques</li>
                  <li>• Recognizing bias and propaganda techniques</li>
                </ul>
                <p
                  className={`text-sm mt-3 ${
                    isDarkMode ? "text-gray-300" : "text-gray-700"
                  }`}
                >
                  Complete modules to earn badges and track your progress in
                  becoming a more informed digital citizen.
                </p>
              </div>
            </section>

            {/* Getting Started */}
            <section>
              <h3
                className={`text-lg font-semibold mb-4 ${
                  isDarkMode ? "text-white" : "text-gray-900"
                }`}
              >
                Getting Started
              </h3>
              <div
                className={`p-4 rounded-lg ${
                  isDarkMode ? "bg-gray-700" : "bg-gray-50"
                }`}
              >
                <p
                  className={`text-sm leading-relaxed ${
                    isDarkMode ? "text-gray-300" : "text-gray-700"
                  }`}
                >
                  Simply type your question or claim in the chat box, upload an
                  image or video file, or paste a link to social media content.
                  Our AI will analyze it and provide you with a detailed
                  verification report including sources and evidence. Start with
                  something you're curious about - there's no limit to what you
                  can verify!
                </p>
              </div>
            </section>
          </div>

          {/* Footer */}
          <div
            className={`p-6 border-t ${
              isDarkMode ? "border-gray-700" : "border-gray-200"
            }`}
          >
            <div className="flex items-center justify-between">
              <p
                className={`text-xs ${
                  isDarkMode ? "text-gray-400" : "text-gray-500"
                }`}
              >
                Project Aegis - Empowering digital literacy through AI-powered
                fact-checking
              </p>
              <motion.button
                onClick={onClose}
                className={`px-4 py-2 rounded-lg text-sm font-medium ${
                  isDarkMode
                    ? "bg-blue-600 hover:bg-blue-700 text-white"
                    : "bg-blue-500 hover:bg-blue-600 text-white"
                }`}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                Got it!
              </motion.button>
            </div>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
};

export default InfoModal;
