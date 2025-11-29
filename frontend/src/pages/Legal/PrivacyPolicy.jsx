import { motion } from "framer-motion";

const PrivacyPolicy = () => {
  return (
    <div className="min-h-screen bg-black py-12">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <h1 className="text-4xl font-bold text-white mb-4">Privacy Policy</h1>
          <p className="text-gray-400">
            Last updated: {new Date().toLocaleDateString("en-US", { year: "numeric", month: "long", day: "numeric" })}
          </p>
        </motion.div>

        <div className="space-y-8">
          <motion.section
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="rounded-2xl border border-white/10 bg-black/40 backdrop-blur-xl p-8"
          >
            <h2 className="text-2xl font-bold text-white mb-4">Introduction</h2>
            <p className="text-gray-300 leading-relaxed">
              Project Aegis ("we", "our", or "us") is committed to protecting your privacy. This Privacy Policy explains how we collect, use, disclose, and safeguard your information when you use our AI-powered misinformation detection and fact-checking platform.
            </p>
          </motion.section>

          <motion.section
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="rounded-2xl border border-white/10 bg-black/40 backdrop-blur-xl p-8"
          >
            <h2 className="text-2xl font-bold text-white mb-4">
              Information We Collect
            </h2>
            <div className="space-y-4 text-gray-300">
              <div>
                <h3 className="font-semibold text-white mb-2">Personal Information</h3>
                <p className="leading-relaxed">
                  We may collect personal information that you provide directly, including:
                </p>
                <ul className="list-disc list-inside mt-2 space-y-1 ml-4">
                  <li>Name and email address when you create an account</li>
                  <li>Phone number and age (optional)</li>
                  <li>Domain preferences for personalized content</li>
                  <li>Payment information for subscription services (processed securely through third-party providers)</li>
                </ul>
              </div>
              <div>
                <h3 className="font-semibold text-white mb-2">Usage Information</h3>
                <p className="leading-relaxed">
                  We automatically collect certain information about your device and how you interact with our platform:
                </p>
                <ul className="list-disc list-inside mt-2 space-y-1 ml-4">
                  <li>Verification requests and fact-checking queries</li>
                  <li>Chat session history and messages</li>
                  <li>IP address, browser type, and device information</li>
                  <li>Usage patterns and feature interactions</li>
                </ul>
              </div>
            </div>
          </motion.section>

          <motion.section
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="rounded-2xl border border-white/10 bg-black/40 backdrop-blur-xl p-8"
          >
            <h2 className="text-2xl font-bold text-white mb-4">
              How We Use Your Information
            </h2>
            <p className="text-gray-300 leading-relaxed mb-4">
              We use the collected information for various purposes:
            </p>
            <ul className="list-disc list-inside space-y-2 text-gray-300 ml-4">
              <li>To provide, maintain, and improve our fact-checking services</li>
              <li>To process your verification requests and provide results</li>
              <li>To manage your account and subscription</li>
              <li>To send you updates, notifications, and support communications</li>
              <li>To personalize your experience based on your preferences</li>
              <li>To analyze usage patterns and improve our platform</li>
              <li>To detect, prevent, and address technical issues and security threats</li>
            </ul>
          </motion.section>

          <motion.section
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
            className="rounded-2xl border border-white/10 bg-black/40 backdrop-blur-xl p-8"
          >
            <h2 className="text-2xl font-bold text-white mb-4">Data Security</h2>
            <p className="text-gray-300 leading-relaxed">
              We implement appropriate technical and organizational security measures to protect your personal information against unauthorized access, alteration, disclosure, or destruction. However, no method of transmission over the Internet or electronic storage is 100% secure, and we cannot guarantee absolute security.
            </p>
          </motion.section>

          <motion.section
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5 }}
            className="rounded-2xl border border-white/10 bg-black/40 backdrop-blur-xl p-8"
          >
            <h2 className="text-2xl font-bold text-white mb-4">Your Rights</h2>
            <p className="text-gray-300 leading-relaxed mb-4">
              Depending on your location, you may have certain rights regarding your personal information:
            </p>
            <ul className="list-disc list-inside space-y-2 text-gray-300 ml-4">
              <li>Access and receive a copy of your personal data</li>
              <li>Rectify inaccurate or incomplete information</li>
              <li>Request deletion of your personal information</li>
              <li>Object to processing of your personal data</li>
              <li>Request restriction of processing</li>
              <li>Data portability</li>
              <li>Withdraw consent at any time</li>
            </ul>
          </motion.section>

          <motion.section
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.6 }}
            className="rounded-2xl border border-white/10 bg-black/40 backdrop-blur-xl p-8"
          >
            <h2 className="text-2xl font-bold text-white mb-4">Contact Us</h2>
            <p className="text-gray-300 leading-relaxed">
              If you have any questions about this Privacy Policy or our data practices, please contact us at{" "}
              <a href="mailto:privacy@projectaegis.com" className="text-blue-400 hover:text-blue-300 underline">
                privacy@projectaegis.com
              </a>
            </p>
          </motion.section>
        </div>
      </div>
    </div>
  );
};

export default PrivacyPolicy;
