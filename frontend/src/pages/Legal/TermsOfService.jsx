import { motion } from "framer-motion";

const TermsOfService = () => {
  return (
    <div className="min-h-screen bg-black py-12">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <h1 className="text-4xl font-bold text-white mb-4">Terms of Service</h1>
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
            <h2 className="text-2xl font-bold text-white mb-4">
              Agreement to Terms
            </h2>
            <p className="text-gray-300 leading-relaxed">
              By accessing or using Project Aegis ("the Platform"), you agree to be bound by these Terms of Service ("Terms"). If you disagree with any part of these Terms, you may not access or use our services.
            </p>
          </motion.section>

          <motion.section
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="rounded-2xl border border-white/10 bg-black/40 backdrop-blur-xl p-8"
          >
            <h2 className="text-2xl font-bold text-white mb-4">
              Use of Service
            </h2>
            <div className="space-y-4 text-gray-300">
              <div>
                <h3 className="font-semibold text-white mb-2">Eligibility</h3>
                <p className="leading-relaxed">
                  You must be at least 18 years old to use our Platform. By using our services, you represent and warrant that you meet this age requirement.
                </p>
              </div>
              <div>
                <h3 className="font-semibold text-white mb-2">Account Responsibility</h3>
                <p className="leading-relaxed">
                  You are responsible for maintaining the confidentiality of your account credentials and for all activities that occur under your account. You agree to notify us immediately of any unauthorized use of your account.
                </p>
              </div>
              <div>
                <h3 className="font-semibold text-white mb-2">Acceptable Use</h3>
                <p className="leading-relaxed mb-2">You agree not to:</p>
                <ul className="list-disc list-inside space-y-1 ml-4">
                  <li>Use the Platform for any illegal or unauthorized purpose</li>
                  <li>Violate any laws in your jurisdiction</li>
                  <li>Transmit any malicious code, viruses, or harmful data</li>
                  <li>Attempt to gain unauthorized access to any part of the Platform</li>
                  <li>Interfere with or disrupt the Platform's operation</li>
                  <li>Use automated systems to access the Platform without authorization</li>
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
              Subscriptions and Payments
            </h2>
            <div className="space-y-4 text-gray-300">
              <p className="leading-relaxed">
                Project Aegis offers various subscription tiers (Free, Pro, Enterprise) with different features and limitations.
              </p>
              <ul className="list-disc list-inside space-y-2 ml-4">
                <li>Subscription fees are charged in advance on a recurring basis</li>
                <li>You may cancel your subscription at any time, with cancellation taking effect at the end of the current billing period</li>
                <li>Refunds are handled according to our refund policy</li>
                <li>We reserve the right to modify subscription prices with prior notice</li>
                <li>Failure to pay may result in suspension or termination of your account</li>
              </ul>
            </div>
          </motion.section>

          <motion.section
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
            className="rounded-2xl border border-white/10 bg-black/40 backdrop-blur-xl p-8"
          >
            <h2 className="text-2xl font-bold text-white mb-4">
              Intellectual Property
            </h2>
            <p className="text-gray-300 leading-relaxed">
              The Platform and its original content, features, and functionality are owned by Project Aegis and are protected by international copyright, trademark, patent, trade secret, and other intellectual property laws. You may not copy, modify, distribute, sell, or lease any part of our services without our express written permission.
            </p>
          </motion.section>

          <motion.section
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5 }}
            className="rounded-2xl border border-white/10 bg-black/40 backdrop-blur-xl p-8"
          >
            <h2 className="text-2xl font-bold text-white mb-4">
              Disclaimer of Warranties
            </h2>
            <p className="text-gray-300 leading-relaxed mb-4">
              Our fact-checking services are provided on an "as-is" and "as-available" basis. We do not warrant that:
            </p>
            <ul className="list-disc list-inside space-y-2 text-gray-300 ml-4">
              <li>The Platform will be uninterrupted, timely, secure, or error-free</li>
              <li>The results of any verification will be accurate or complete</li>
              <li>Defects will be corrected</li>
              <li>The Platform is free of viruses or other harmful components</li>
            </ul>
            <p className="text-gray-300 leading-relaxed mt-4">
              Our AI-powered verification is a tool to assist in fact-checking, but we cannot guarantee the absolute accuracy of all results. Users should exercise their own judgment and verify information through multiple sources.
            </p>
          </motion.section>

          <motion.section
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.6 }}
            className="rounded-2xl border border-white/10 bg-black/40 backdrop-blur-xl p-8"
          >
            <h2 className="text-2xl font-bold text-white mb-4">
              Limitation of Liability
            </h2>
            <p className="text-gray-300 leading-relaxed">
              To the maximum extent permitted by law, Project Aegis shall not be liable for any indirect, incidental, special, consequential, or punitive damages, or any loss of profits or revenues, whether incurred directly or indirectly, or any loss of data, use, goodwill, or other intangible losses resulting from your use of the Platform.
            </p>
          </motion.section>

          <motion.section
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.7 }}
            className="rounded-2xl border border-white/10 bg-black/40 backdrop-blur-xl p-8"
          >
            <h2 className="text-2xl font-bold text-white mb-4">Changes to Terms</h2>
            <p className="text-gray-300 leading-relaxed">
              We reserve the right to modify or replace these Terms at any time. If a revision is material, we will provide at least 30 days notice prior to any new terms taking effect. What constitutes a material change will be determined at our sole discretion.
            </p>
          </motion.section>

          <motion.section
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.8 }}
            className="rounded-2xl border border-white/10 bg-black/40 backdrop-blur-xl p-8"
          >
            <h2 className="text-2xl font-bold text-white mb-4">Contact Us</h2>
            <p className="text-gray-300 leading-relaxed">
              If you have any questions about these Terms of Service, please contact us at{" "}
              <a href="mailto:legal@projectaegis.com" className="text-blue-400 hover:text-blue-300 underline">
                legal@projectaegis.com
              </a>
            </p>
          </motion.section>
        </div>
      </div>
    </div>
  );
};

export default TermsOfService;
