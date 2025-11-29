import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
// eslint-disable-next-line no-unused-vars
import { motion, AnimatePresence } from "framer-motion";
import { Mail, Lock, UserPlus, Phone, Calendar, ArrowRight, ArrowLeft, Check, Eye, EyeOff } from "lucide-react";
import { getApiBaseUrl } from "../../services/api";
import { useAuth } from "../../contexts/AuthContext";

const DOMAIN_OPTIONS = [
  "Politics",
  "Technology",
  "Health",
  "Crime",
  "Military",
  "Sports",
  "Entertainment",
  "Social Media only",
];

const Signup = () => {
  const navigate = useNavigate();
  const { login } = useAuth();
  const [currentStep, setCurrentStep] = useState(1);
  const [formData, setFormData] = useState({
    name: "",
    email: "",
    password: "",
    confirmPassword: "",
    phoneNumber: "",
    age: "",
    domainPreferences: [],
  });
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);

  const validateStep1 = () => {
    if (!formData.name.trim()) {
      setError("Name is required");
      return false;
    }
    if (!formData.email.trim()) {
      setError("Email is required");
      return false;
    }
    if (formData.password.length < 6) {
      setError("Password must be at least 6 characters");
      return false;
    }
    if (formData.password !== formData.confirmPassword) {
      setError("Passwords do not match");
      return false;
    }
    return true;
  };

  const handleNextStep = () => {
    setError("");
    if (validateStep1()) {
      setCurrentStep(2);
    }
  };

  const handlePreviousStep = () => {
    setError("");
    setCurrentStep(1);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");

    if (currentStep === 1) {
      handleNextStep();
      return;
    }

    setLoading(true);

    try {
      const response = await fetch(`${getApiBaseUrl()}/auth/signup`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          name: formData.name,
          email: formData.email,
          password: formData.password,
          phone_number: formData.phoneNumber,
          age: formData.age ? parseInt(formData.age) : null,
          domain_preferences: formData.domainPreferences,
        }),
      });

      if (response.ok) {
        const data = await response.json();
        if (data.token && data.user) {
          login(data.user, data.token);
        }
        navigate("/");
      } else {
        const data = await response.json();
        setError(data.detail || "Signup failed. Please try again.");
      }
    } catch {
      setError("Network error. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-black relative overflow-hidden flex items-center justify-center py-12 px-4 sm:px-6 lg:px-8">
      {/* Animated Background */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none" style={{ zIndex: 0 }}>
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
      </div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="w-full max-w-md relative z-10"
      >
        <div className="bg-black/80 backdrop-blur-xl rounded-2xl p-6 border border-gray-800/50 shadow-2xl">
          {/* Header */}
          <div className="mb-4">
            <h2 className="text-3xl font-bold text-white mb-1 bg-gradient-to-r from-blue-400 via-purple-400 to-cyan-400 bg-clip-text text-transparent">
              Create Account
            </h2>
            <p className="text-gray-400 text-sm">Sign up to get started</p>
          </div>

          {/* Step Indicator */}
          <div className="mb-4">
            <div className="flex items-center justify-center space-x-4">
              {/* Step 1 */}
              <div className="flex flex-col items-center">
                <motion.div
                  className={`relative w-12 h-12 rounded-full flex items-center justify-center border-2 transition-all ${
                    currentStep >= 1
                      ? "bg-blue-600 border-blue-500"
                      : "bg-gray-800 border-gray-700"
                  }`}
                  initial={{ scale: 0.8 }}
                  animate={{ scale: 1 }}
                  transition={{ duration: 0.3 }}
                >
                  {currentStep > 1 ? (
                    <Check className="w-6 h-6 text-white" />
                  ) : (
                    <span className="text-white font-bold">1</span>
                  )}
                  {currentStep >= 1 && (
                    <motion.div
                      className="absolute inset-0 bg-blue-600 rounded-full"
                      initial={{ scale: 0 }}
                      animate={{ scale: 1.5, opacity: 0 }}
                      transition={{ duration: 0.6, repeat: Infinity, repeatDelay: 2 }}
                    />
                  )}
                </motion.div>
                <span className={`mt-1 text-xs font-medium ${currentStep >= 1 ? "text-blue-400" : "text-gray-500"}`}>
                  Basic Details
                </span>
              </div>

              {/* Connector Line */}
              <div className="flex-1 h-0.5 relative">
                <div className={`absolute inset-0 transition-all ${currentStep >= 2 ? "bg-gradient-to-r from-blue-600 to-purple-600" : "bg-gray-700"}`} />
                <motion.div
                  className={`absolute inset-0 bg-gradient-to-r from-blue-600 to-purple-600`}
                  initial={{ scaleX: 0 }}
                  animate={{ scaleX: currentStep >= 2 ? 1 : 0 }}
                  transition={{ duration: 0.5 }}
                  style={{ transformOrigin: "left" }}
                />
              </div>

              {/* Step 2 */}
              <div className="flex flex-col items-center">
                <motion.div
                  className={`relative w-12 h-12 rounded-full flex items-center justify-center border-2 transition-all ${
                    currentStep >= 2
                      ? "bg-purple-600 border-purple-500"
                      : "bg-gray-800 border-gray-700"
                  }`}
                  initial={{ scale: 0.8 }}
                  animate={{ scale: 1 }}
                  transition={{ duration: 0.3 }}
                >
                  <span className="text-white font-bold">2</span>
                  {currentStep >= 2 && (
                    <motion.div
                      className="absolute inset-0 bg-purple-600 rounded-full"
                      initial={{ scale: 0 }}
                      animate={{ scale: 1.5, opacity: 0 }}
                      transition={{ duration: 0.6, repeat: Infinity, repeatDelay: 2 }}
                    />
                  )}
                </motion.div>
                <span className={`mt-1 text-xs font-medium ${currentStep >= 2 ? "text-purple-400" : "text-gray-500"}`}>
                  Domains
                </span>
              </div>
            </div>
          </div>

          {error && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              className="mb-3 p-2 bg-red-900/30 border border-red-700/50 rounded-lg text-red-200 text-sm backdrop-blur-sm"
            >
              {error}
            </motion.div>
          )}

          <form onSubmit={handleSubmit} className="space-y-3">
            <AnimatePresence mode="wait">
              {currentStep === 1 ? (
                <motion.div
                  key="step1"
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: 20 }}
                  transition={{ duration: 0.3 }}
                  className="space-y-2"
                >
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-1">
                      Full Name
                    </label>
                    <div className="relative">
                      <UserPlus className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
                      <input
                        type="text"
                        required
                        value={formData.name}
                        onChange={(e) => {
                          setFormData({ ...formData, name: e.target.value });
                          setError("");
                        }}
                        className="w-full pl-9 pr-3 py-2 bg-gray-900/50 text-white rounded-lg border border-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all backdrop-blur-sm text-sm"
                        placeholder="John Doe"
                      />
                    </div>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-1">
                      Email
                    </label>
                    <div className="relative">
                      <Mail className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
                      <input
                        type="email"
                        required
                        value={formData.email}
                        onChange={(e) => {
                          setFormData({ ...formData, email: e.target.value });
                          setError("");
                        }}
                        className="w-full pl-9 pr-3 py-2 bg-gray-900/50 text-white rounded-lg border border-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all backdrop-blur-sm text-sm"
                        placeholder="you@example.com"
                      />
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-1">
                        Password
                      </label>
                      <div className="relative">
                        <Lock className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
                        <input
                          type={showPassword ? "text" : "password"}
                          required
                          value={formData.password}
                          onChange={(e) => {
                            setFormData({ ...formData, password: e.target.value });
                            setError("");
                          }}
                          className="w-full pl-9 pr-10 py-2 bg-gray-900/50 text-white rounded-lg border border-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all backdrop-blur-sm text-sm"
                          placeholder="••••••••"
                        />
                        <button
                          type="button"
                          onClick={() => setShowPassword(!showPassword)}
                          className="absolute right-2 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-gray-300 transition-colors"
                        >
                          {showPassword ? (
                            <EyeOff className="w-4 h-4" />
                          ) : (
                            <Eye className="w-4 h-4" />
                          )}
                        </button>
                      </div>
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-1">
                        Confirm Password
                      </label>
                      <div className="relative">
                        <Lock className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
                        <input
                          type={showConfirmPassword ? "text" : "password"}
                          required
                          value={formData.confirmPassword}
                          onChange={(e) => {
                            setFormData({ ...formData, confirmPassword: e.target.value });
                            setError("");
                          }}
                          className="w-full pl-9 pr-10 py-2 bg-gray-900/50 text-white rounded-lg border border-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all backdrop-blur-sm text-sm"
                          placeholder="••••••••"
                        />
                        <button
                          type="button"
                          onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                          className="absolute right-2 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-gray-300 transition-colors"
                        >
                          {showConfirmPassword ? (
                            <EyeOff className="w-4 h-4" />
                          ) : (
                            <Eye className="w-4 h-4" />
                          )}
                        </button>
                      </div>
                    </div>
                  </div>
                </motion.div>
              ) : (
                <motion.div
                  key="step2"
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: 20 }}
                  transition={{ duration: 0.3 }}
                  className="space-y-2"
                >
                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-1">
                        Phone Number
                      </label>
                      <div className="relative">
                        <Phone className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
                        <input
                          type="tel"
                          required
                          value={formData.phoneNumber}
                          onChange={(e) => {
                            setFormData({ ...formData, phoneNumber: e.target.value });
                            setError("");
                          }}
                          className="w-full pl-9 pr-3 py-2 bg-gray-900/50 text-white rounded-lg border border-gray-700 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all backdrop-blur-sm text-sm"
                          placeholder="+1234567890"
                        />
                      </div>
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-1">
                        Age
                      </label>
                      <div className="relative">
                        <Calendar className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
                        <input
                          type="number"
                          required
                          min="13"
                          max="120"
                          value={formData.age}
                          onChange={(e) => {
                            setFormData({ ...formData, age: e.target.value });
                            setError("");
                          }}
                          className="w-full pl-9 pr-3 py-2 bg-gray-900/50 text-white rounded-lg border border-gray-700 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all backdrop-blur-sm text-sm"
                          placeholder="25"
                        />
                      </div>
                    </div>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-1">
                      Select Domains for Updates (Optional)
                    </label>
                    <p className="text-xs text-gray-400 mb-2">
                      Choose domains you'd like to receive fact-check alerts for. You can skip this step.
                    </p>
                    <div className="grid grid-cols-2 gap-2">
                      {DOMAIN_OPTIONS.map((domain) => (
                        <motion.label
                          key={domain}
                          whileHover={{ scale: 1.02 }}
                          whileTap={{ scale: 0.98 }}
                          className={`flex items-center space-x-2 cursor-pointer p-2 rounded-lg border transition-all ${
                            formData.domainPreferences.includes(domain)
                              ? "border-purple-500 bg-purple-500/10"
                              : "border-gray-700 hover:border-purple-500/50 bg-gray-900/30"
                          } backdrop-blur-sm`}
                        >
                          <input
                            type="checkbox"
                            checked={formData.domainPreferences.includes(domain)}
                            onChange={(e) => {
                              if (e.target.checked) {
                                setFormData({
                                  ...formData,
                                  domainPreferences: [
                                    ...formData.domainPreferences,
                                    domain,
                                  ],
                                });
                              } else {
                                setFormData({
                                  ...formData,
                                  domainPreferences: formData.domainPreferences.filter(
                                    (d) => d !== domain
                                  ),
                                });
                              }
                            }}
                            className="w-3.5 h-3.5 text-purple-600 bg-gray-700 border-gray-600 rounded focus:ring-purple-500"
                          />
                          <span className={`text-xs ${formData.domainPreferences.includes(domain) ? "text-purple-300 font-medium" : "text-gray-300"}`}>
                            {domain}
                          </span>
                        </motion.label>
                      ))}
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Navigation Buttons */}
            <div className="flex gap-3 pt-2">
              {currentStep === 2 && (
                <button
                  type="button"
                  onClick={handlePreviousStep}
                  className="flex-1 flex items-center justify-center space-x-2 px-4 py-2 bg-gray-800 text-white rounded-lg font-semibold hover:bg-gray-700 transition-all border border-gray-700 text-sm"
                >
                  <ArrowLeft className="w-4 h-4" />
                  <span>Back</span>
                </button>
              )}
              <button
                type="submit"
                disabled={loading}
                className={`flex-1 flex items-center justify-center space-x-2 px-4 py-2 text-white rounded-lg font-semibold transition-all text-sm ${
                  currentStep === 1
                    ? "bg-blue-600 hover:bg-blue-700"
                    : "bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700"
                } disabled:opacity-50 disabled:cursor-not-allowed`}
              >
                {loading ? (
                  <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                ) : (
                  <>
                    {currentStep === 1 ? (
                      <>
                        <span>Next</span>
                        <ArrowRight className="w-4 h-4" />
                      </>
                    ) : (
                      <>
                        <UserPlus className="w-4 h-4" />
                        <span>Sign Up</span>
                      </>
                    )}
                  </>
                )}
              </button>
            </div>
          </form>

          <p className="mt-3 text-center text-gray-400 text-sm">
            Already have an account?{" "}
            <Link to="/login" className="text-blue-400 hover:text-blue-300 font-medium transition-colors">
              Sign in
            </Link>
          </p>
        </div>
      </motion.div>
    </div>
  );
};

export default Signup;

