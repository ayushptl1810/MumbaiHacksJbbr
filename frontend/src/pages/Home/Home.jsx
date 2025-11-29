import { Suspense, useState } from "react";
import { Link } from "react-router-dom";
// eslint-disable-next-line no-unused-vars
import { motion, AnimatePresence } from "framer-motion";
import {
  Image,
  Video,
  Mic,
  FileText,
  ChevronDown,
  ChevronUp,
} from "lucide-react";
import Hero3D from "../../components/Hero3D";

const Home = () => {
  const [expandedType, setExpandedType] = useState(null);

  // Generate random values for particles once using lazy initializer
  const [particleConfigs] = useState(() => {
    return Array.from({ length: 20 }, () => ({
      duration: 3 + Math.random() * 2,
      delay: Math.random() * 2,
      left: Math.random() * 100,
      top: Math.random() * 100,
    }));
  });

  const verificationTypes = [
    {
      icon: FileText,
      title: "Text Verification",
      description: "Verify claims and statements with AI-powered fact-checking",
      expandedContent:
        "Our advanced AI analyzes text content using natural language processing, cross-references with trusted databases, and checks against known misinformation patterns. Get instant verification results with confidence scores and detailed source citations.",
    },
    {
      icon: Image,
      title: "Image Analysis",
      description: "Detect manipulated images and verify visual content",
      expandedContent:
        "Using cutting-edge computer vision and deep learning models, we detect image manipulation, deepfakes, and verify authenticity. Our system checks metadata, performs reverse image searches, and analyzes pixel-level inconsistencies to ensure image integrity.",
    },
    {
      icon: Video,
      title: "Video Verification",
      description: "Analyze videos for deepfakes and authenticity",
      expandedContent:
        "Comprehensive video analysis includes frame-by-frame deepfake detection, audio-video synchronization checks, and source verification. Our multi-model approach identifies synthetic media with high accuracy, protecting you from manipulated video content.",
    },
    {
      icon: Mic,
      title: "Audio Verification",
      description: "Verify audio recordings and detect voice manipulation",
      expandedContent:
        "Advanced audio forensics detect voice cloning, audio deepfakes, and verify recording authenticity. Our system analyzes spectral patterns, voice biometrics, and audio artifacts to ensure the audio content is genuine and unmanipulated.",
    },
  ];

  const processSteps = [
    {
      number: "1",
      title: "Submit Content",
      description: "Upload text, image, video, or audio for verification",
      details:
        "Simply drag and drop your content or paste text directly into our verification interface. Our system accepts multiple file formats and automatically detects the content type for optimal processing.",
    },
    {
      number: "2",
      title: "AI Analysis",
      description:
        "Our AI analyzes the content using multiple verification methods",
      details:
        "Advanced machine learning models process your content through multiple verification pipelines simultaneously. This includes deepfake detection, metadata analysis, reverse image/video searches, and pattern recognition algorithms.",
    },
    {
      number: "3",
      title: "Source Checking",
      description: "Cross-reference with reliable sources and databases",
      details:
        "Our system cross-references your content against trusted fact-checking databases, academic sources, and verified news outlets. We check for known misinformation patterns and verify claims against authoritative sources in real-time.",
    },
    {
      number: "4",
      title: "Get Results",
      description:
        "Receive detailed verification report with confidence scores",
      details:
        "Get comprehensive verification results including confidence scores, source citations, detailed analysis breakdown, and actionable insights. Our reports are transparent, showing exactly how we reached our conclusions.",
    },
  ];

  return (
    <div className="min-h-screen bg-black relative overflow-hidden">
      {/* Animated Background - Full Page */}
      <div
        className="fixed inset-0 overflow-hidden pointer-events-none"
        style={{ zIndex: 0 }}
      >
        {/* Animated gradient orbs */}
        <motion.div
          className="absolute w-96 h-96 bg-blue-600/20 rounded-full blur-3xl"
          animate={{
            x: [0, 100, 0],
            y: [0, 50, 0],
            scale: [1, 1.2, 1],
          }}
          transition={{
            duration: 20,
            repeat: Infinity,
            ease: "easeInOut",
          }}
          style={{ top: "10%", left: "10%" }}
        />
        <motion.div
          className="absolute w-96 h-96 bg-purple-600/20 rounded-full blur-3xl"
          animate={{
            x: [0, -100, 0],
            y: [0, -50, 0],
            scale: [1, 1.3, 1],
          }}
          transition={{
            duration: 25,
            repeat: Infinity,
            ease: "easeInOut",
          }}
          style={{ bottom: "10%", right: "10%" }}
        />
        <motion.div
          className="absolute w-96 h-96 bg-cyan-600/15 rounded-full blur-3xl"
          animate={{
            x: [0, 50, 0],
            y: [0, 100, 0],
            scale: [1, 1.1, 1],
          }}
          transition={{
            duration: 30,
            repeat: Infinity,
            ease: "easeInOut",
          }}
          style={{ top: "50%", left: "50%" }}
        />

        {/* Grid pattern overlay */}
        <div
          className="absolute inset-0 opacity-10"
          style={{
            backgroundImage: `
                linear-gradient(rgba(59, 130, 246, 0.1) 1px, transparent 1px),
                linear-gradient(90deg, rgba(59, 130, 246, 0.1) 1px, transparent 1px)
              `,
            backgroundSize: "50px 50px",
          }}
        />

        {/* Floating particles */}
        {particleConfigs.map((config, i) => (
          <motion.div
            key={i}
            className="absolute w-1 h-1 bg-blue-400 rounded-full"
            animate={{
              y: [0, -100, 0],
              opacity: [0.3, 1, 0.3],
              scale: [1, 1.5, 1],
            }}
            transition={{
              duration: config.duration,
              repeat: Infinity,
              delay: config.delay,
              ease: "easeInOut",
            }}
            style={{
              left: `${config.left}%`,
              top: `${config.top}%`,
            }}
          />
        ))}
      </div>

      {/* Hero Section */}
      <section
        className="relative min-h-screen flex items-center"
        style={{ zIndex: 1 }}
      >
        <div className="relative z-10 layout-container py-12 lg:py-16 w-full">
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-center">
            {/* Left Side - Content (65%) */}
            <motion.div
              initial={{ opacity: 0, x: -50 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.8 }}
              className="lg:col-span-8"
            >
              <h1 className="text-4xl sm:text-5xl lg:text-7xl font-bold text-white mb-6 leading-tight">
                Fight Misinformation
                <span className="block bg-gradient-to-r from-blue-400 via-purple-400 to-cyan-400 bg-clip-text text-transparent">
                  with AI Power
                </span>
              </h1>
              <p className="text-lg sm:text-xl lg:text-[1.3rem] text-gray-300 mb-8 leading-relaxed">
                Verify claims, detect deepfakes, and get accurate fact-checks in
                seconds. Trust the truth, not the rumors.
              </p>
              <div className="flex flex-row flex-wrap gap-4">
                <Link
                  to="/verify"
                  className="group relative flex-1 min-w-[140px] px-6 py-4 bg-blue-600 text-white rounded-lg font-semibold hover:bg-blue-700 transition-all overflow-hidden text-center"
                >
                  <span className="relative z-10">Start Verification</span>
                  <motion.div
                    className="absolute inset-0 bg-gradient-to-r from-blue-500 to-purple-500"
                    initial={{ x: "-100%" }}
                    whileHover={{ x: 0 }}
                    transition={{ duration: 0.3 }}
                  />
                </Link>
                <Link
                  to="/modules"
                  className="flex-1 min-w-[140px] px-6 py-4 bg-transparent border-2 border-gray-700 text-white rounded-lg font-semibold hover:border-blue-500 transition-colors text-center"
                >
                  Explore Modules
                </Link>
              </div>
            </motion.div>

            {/* Right Side - 3D Model (35%) */}
            <motion.div
              initial={{ opacity: 0, x: 50 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.8, delay: 0.2 }}
              className="hidden lg:block relative h-[700px] lg:col-span-4"
            >
              <Suspense
                fallback={
                  <div className="w-full h-full flex items-center justify-center">
                    <div className="w-32 h-32 border-4 border-blue-500 border-t-transparent rounded-full animate-spin" />
                  </div>
                }
              >
                {/* GLTF Model */}
                <Hero3D
                  type="gltf"
                  modelPath={
                    import.meta.env.VITE_GLTF_MODEL_PATH || "/scene.gltf"
                  }
                />
              </Suspense>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Four-Card Section */}
      <section className="py-20 relative" style={{ zIndex: 1 }}>
        <div className="layout-container">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
            className="text-center mb-12"
          >
            <h2 className="text-3xl sm:text-4xl font-bold text-white mb-4">
              Types of Verification We Support
            </h2>
            <p className="text-gray-400 text-lg">
              Comprehensive fact-checking across all media types
            </p>
          </motion.div>

          <div className="max-w-4xl mx-auto">
            {verificationTypes.map((type, index) => {
              const isExpanded = expandedType === index;
              return (
                <motion.div
                  key={type.title}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.6, delay: index * 0.1 }}
                  className="overflow-hidden"
                >
                  <button
                    onClick={() => setExpandedType(isExpanded ? null : index)}
                    className="w-full flex items-center justify-between py-6 text-left hover:opacity-80 transition-opacity"
                  >
                    <h3 className="text-xl font-bold text-white flex-1">
                      {type.title}
                    </h3>
                    <div className="ml-4 flex-shrink-0">
                      {isExpanded ? (
                        <ChevronUp className="w-5 h-5 text-white" />
                      ) : (
                        <ChevronDown className="w-5 h-5 text-white" />
                      )}
                    </div>
                  </button>

                  <AnimatePresence>
                    {isExpanded && (
                      <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: "auto", opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        transition={{ duration: 0.3 }}
                        className="overflow-hidden"
                      >
                        <div className="pb-6">
                          <p className="text-gray-400 text-base leading-relaxed">
                            {type.expandedContent}
                          </p>
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>

                  {/* Horizontal line below each verification type */}
                  <div className="border-b border-gray-800" />
                </motion.div>
              );
            })}
          </div>
        </div>
      </section>

      {/* Verification Process - Horizontal Timeline */}
      <section className="py-16 relative" style={{ zIndex: 1 }}>
        <div className="layout-container">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
            className="text-center mb-12"
          >
            <h2 className="text-3xl sm:text-4xl font-bold text-white mb-4">
              How It Works
            </h2>
            <p className="text-gray-400 text-lg">
              Simple, fast, and reliable verification process
            </p>
          </motion.div>

          {/* Horizontal Timeline Layout */}
          <div className="relative">
            {/* Straight connecting line - show on larger screens where steps are in a row */}
            <div className="hidden lg:block absolute top-36 left-0 right-0 h-1 bg-gradient-to-r from-blue-600 via-purple-600 to-cyan-600 z-0" />

            {/* Steps in a row */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 lg:gap-8 relative z-10">
              {processSteps.map((step, index) => (
                <motion.div
                  key={step.number}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.6, delay: index * 0.1 }}
                  className="flex flex-col items-center text-center"
                >
                  {/* Title above number */}
                  <div className="mb-6 text-center">
                    <h3 className="text-lg font-semibold text-white">
                      {step.title}
                    </h3>
                  </div>

                  {/* Step Number Circle */}
                  <div className="relative mb-6">
                    <motion.div
                      className="absolute inset-0 bg-blue-600/30 rounded-full blur-xl"
                      animate={{
                        scale: [1, 1.1, 1],
                        opacity: [0.5, 0.7, 0.5],
                      }}
                      transition={{
                        duration: 2,
                        repeat: Infinity,
                        ease: "easeInOut",
                      }}
                    />
                    <div className="relative w-16 h-16 bg-black border-2 border-blue-500 rounded-full flex items-center justify-center">
                      <span className="text-xl font-bold text-blue-400">
                        {step.number}
                      </span>
                    </div>
                  </div>

                  {/* Content below number */}
                  <div className="bg-black border border-gray-800 rounded-lg p-4 hover:border-blue-500 transition-colors w-full mt-6">
                    <p className="text-gray-400 text-sm mb-2 font-medium">
                      {step.description}
                    </p>
                    <p className="text-gray-500 text-xs leading-relaxed">
                      {step.details}
                    </p>
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        </div>
      </section>
    </div>
  );
};

export default Home;
