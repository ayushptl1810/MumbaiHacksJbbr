import { useState, useEffect, useMemo } from "react";
import { useParams, useNavigate } from "react-router-dom";
// eslint-disable-next-line no-unused-vars
import { motion } from "framer-motion";
import {
  Search,
  ArrowLeft,
  ListChecks,
  AlertTriangle,
  CheckCircle,
  ExternalLink,
  TrendingUp,
  Tag,
  Lightbulb,
  Shield,
} from "lucide-react";
import ModuleCard from "../../components/ModuleCard";
import ContentSection from "../../components/ContentSection";
import PracticalTips from "../../components/PracticalTips";
import LoadingSpinner from "../../components/LoadingSpinner";
import { getApiBaseUrl } from "../../services/api";
import { useAuth } from "../../contexts/AuthContext";

const Modules = () => {
  const { id } = useParams();
  const navigate = useNavigate();
  const { user, isAuthenticated } = useAuth();
  const [modules, setModules] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedTags, setSelectedTags] = useState([]);
  const [userProgress] = useState({
    level: "beginner",
    completedModules: [],
    points: 0,
  });
  const [moduleContent, setModuleContent] = useState(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailError, setDetailError] = useState("");
  const [expandedSections, setExpandedSections] = useState({});

  const activeModuleMeta = useMemo(
    () => modules.find((module) => module.id === id),
    [modules, id]
  );

  const allTags = useMemo(() => {
    const tagSet = new Set();
    modules.forEach((module) => {
      (module.tags || []).forEach((tag) => {
        if (typeof tag === "string" && tag.trim()) {
          tagSet.add(tag.trim());
        }
      });
    });
    return Array.from(tagSet).sort((a, b) => a.localeCompare(b));
  }, [modules]);

  useEffect(() => {
    loadModules();
  }, []);

  useEffect(() => {
    if (id) {
      loadModuleContent(id);
    }
  }, [id]);

  useEffect(() => {
    if (!id) {
      setModuleContent(null);
      setDetailError("");
      setExpandedSections({});
    }
  }, [id]);

  const loadModules = async () => {
    try {
      setLoading(true);
      // Add cache-busting query parameter to ensure fresh data
      const cacheBuster = `_t=${Date.now()}`;
      const response = await fetch(
        `${getApiBaseUrl()}/educational/modules?${cacheBuster}`,
        {
          cache: "no-store",
          headers: {
            "Cache-Control": "no-cache",
          },
        }
      );
      if (response.ok) {
        const data = await response.json();
        setModules(data.modules || []);
      } else {
        // Fallback modules
        setModules([
          {
            id: "red_flags",
            title: "How to Spot Red Flags",
            description: "Learn to identify warning signs in misinformation",
            difficulty_levels: ["beginner", "intermediate", "advanced"],
            estimated_time: "10-15 minutes",
          },
          {
            id: "source_credibility",
            title: "Evaluating Source Credibility",
            description: "Understand how to assess source reliability",
            difficulty_levels: ["beginner", "intermediate", "advanced"],
            estimated_time: "15-20 minutes",
          },
          {
            id: "manipulation_techniques",
            title: "Common Manipulation Techniques",
            description: "Learn about various misinformation techniques",
            difficulty_levels: ["intermediate", "advanced"],
            estimated_time: "20-25 minutes",
          },
        ]);
      }
    } catch (error) {
      console.error("Failed to load modules:", error);
    } finally {
      setLoading(false);
    }
  };

  const loadModuleContent = async (moduleId) => {
    if (!moduleId) return;
    try {
      setDetailLoading(true);
      setDetailError("");
      // Add cache-busting query parameter to ensure fresh data
      const cacheBuster = `_t=${Date.now()}`;
      const response = await fetch(
        `${getApiBaseUrl()}/educational/modules/${moduleId}?${cacheBuster}`,
        {
          cache: "no-store",
          headers: {
            "Cache-Control": "no-cache",
          },
        }
      );
      if (!response.ok) {
        throw new Error("Failed to load module content");
      }
      const data = await response.json();
      setModuleContent(data);
      if (
        Array.isArray(data?.content_sections) &&
        data.content_sections.length
      ) {
        setExpandedSections({ [`section-0`]: true });
      } else {
        setExpandedSections({});
      }
    } catch (error) {
      setDetailError(error.message || "Unable to fetch module content");
      setModuleContent(null);
    } finally {
      setDetailLoading(false);
    }
  };

  const handleBackToModules = () => {
    navigate("/modules");
  };

  const toggleSection = (sectionId) => {
    setExpandedSections((prev) => ({
      ...prev,
      [sectionId]: !prev[sectionId],
    }));
  };

  const renderInteractiveSection = (title, items, formatter) => {
    if (!Array.isArray(items) || items.length === 0) return null;
    return (
      <div>
        <p className="text-sm uppercase tracking-wide text-blue-300">{title}</p>
        <div className="mt-3 space-y-3">{items.slice(0, 3).map(formatter)}</div>
      </div>
    );
  };

  const handleModuleClick = (moduleId) => {
    navigate(`/modules/${moduleId}`);
  };

  const normalizeTag = (tag) => (tag || "").toLowerCase().trim();

  const handleTagToggle = (tag) => {
    setSelectedTags((prev) =>
      prev.includes(tag) ? prev.filter((t) => t !== tag) : [...prev, tag]
    );
  };

  const handleClearTags = () => {
    setSelectedTags([]);
  };

  const handleForYouClick = () => {
    if (user?.domain_preferences && user.domain_preferences.length > 0) {
      setSelectedTags(user.domain_preferences);
    }
  };

  const isForYouActive =
    isAuthenticated &&
    user?.domain_preferences &&
    user.domain_preferences.length > 0 &&
    selectedTags.length === user.domain_preferences.length &&
    selectedTags.every((tag) => user.domain_preferences.includes(tag));

  const filteredModules = modules.filter((module) => {
    const matchesSearch =
      module.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      module.description.toLowerCase().includes(searchQuery.toLowerCase());

    const normalizedSelected = selectedTags.map(normalizeTag);
    const moduleTags = (module.tags || []).map(normalizeTag);

    const matchesTags =
      normalizedSelected.length === 0 ||
      moduleTags.some((tag) => normalizedSelected.includes(tag));

    return matchesSearch && matchesTags;
  });

  if (id) {
    const estimatedTime =
      moduleContent?.estimated_time ||
      activeModuleMeta?.estimated_time ||
      "15-20 minutes";
    const learningObjectives = moduleContent?.learning_objectives || [];
    const trendingScore = moduleContent?.trending_score || 0;
    const redFlagsCount = moduleContent?.red_flags?.length || 0;
    const verificationTipsCount = moduleContent?.verification_tips?.length || 0;
    const moduleTags = moduleContent?.tags || activeModuleMeta?.tags || [];

    return (
      <div className="min-h-screen bg-black py-10">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 space-y-8">
          <button
            onClick={handleBackToModules}
            className="inline-flex items-center gap-2 text-sm font-semibold text-gray-300 hover:text-white transition"
          >
            <ArrowLeft className="h-4 w-4" />
            Back to modules
          </button>

          <div className="rounded-[24px] border border-white/10 bg-gradient-to-br from-[#0b101f] via-[#070b17] to-[#05070c] p-1">
            <div className="rounded-[20px] bg-black/70 p-6 sm:p-10 space-y-8">
              <div className="flex flex-col gap-6 lg:flex-row lg:items-center lg:justify-between">
                <div>
                  <p className="text-xs uppercase tracking-[0.3em] text-blue-300/70">
                    Module overview
                  </p>
                  <h1 className="text-3xl font-bold text-white mt-2">
                    {moduleContent?.title ||
                      activeModuleMeta?.title ||
                      "Educational Module"}
                  </h1>
                  <p className="text-gray-400 mt-3 max-w-2xl">
                    {moduleContent?.overview ||
                      activeModuleMeta?.description ||
                      "Learn actionable strategies to identify misinformation."}
                  </p>

                  {moduleTags.length > 0 && (
                    <div className="mt-4 flex flex-wrap gap-2">
                      {moduleTags.map((tag) => (
                        <span
                          key={tag}
                          className="inline-flex items-center gap-1 rounded-full bg-gradient-to-r from-cyan-500/15 to-blue-500/15 px-3 py-1 text-[11px] font-medium text-cyan-200 border border-cyan-500/30"
                        >
                          <Tag className="w-3 h-3 text-cyan-300" />
                          {tag}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              </div>

              <div className="grid gap-4 md:grid-cols-4">
                {[
                  {
                    label: "Estimated time",
                    value: estimatedTime,
                    hint: "Average completion window",
                  },
                  {
                    label: "Trending Score",
                    value: trendingScore > 0 ? `${trendingScore}/10` : "N/A",
                    hint: "Current popularity",
                  },
                  {
                    label: "Red Flags",
                    value: `${redFlagsCount} warnings`,
                    hint: "Key indicators to watch",
                  },
                  {
                    label: "Verification Tips",
                    value: `${verificationTipsCount} strategies`,
                    hint: "Ways to verify information",
                  },
                ]
                  .filter((card) => card.value !== "N/A")
                  .map((card) => (
                    <div
                      key={card.label}
                      className="rounded-2xl border border-white/10 bg-white/5 px-5 py-4 text-left"
                    >
                      <p className="text-xs uppercase tracking-[0.3em] text-gray-400">
                        {card.label}
                      </p>
                      <p className="text-2xl font-semibold text-white mt-2">
                        {card.value}
                      </p>
                      <p className="text-xs text-gray-500 mt-1">{card.hint}</p>
                    </div>
                  ))}
              </div>

              {learningObjectives.length > 0 && (
                <div className="rounded-2xl border border-white/10 bg-white/5 p-6">
                  <div className="flex items-center gap-3 text-white font-semibold">
                    <ListChecks className="h-5 w-5 text-blue-300" />
                    Learning objectives
                  </div>
                  <ul className="mt-4 space-y-2 text-sm text-gray-300">
                    {learningObjectives.map((objective, index) => (
                      <li key={index} className="flex gap-2">
                        <span className="text-blue-400">•</span>
                        <span>{objective}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </div>

          {detailLoading ? (
            <LoadingSpinner />
          ) : detailError ? (
            <div className="rounded-2xl border border-red-500/40 bg-red-500/10 p-6 text-red-200">
              <p className="font-semibold">
                We couldn&apos;t load this module.
              </p>
              <p className="text-sm opacity-80 mt-2">{detailError}</p>
              <button
                onClick={() => loadModuleContent(id)}
                className="mt-4 inline-flex items-center gap-2 rounded-lg border border-red-500/40 px-4 py-2 text-sm hover:bg-red-500/20"
              >
                Try again
              </button>
            </div>
          ) : moduleContent ? (
            <>
              {/* Technique Explanation Section */}
              {moduleContent.technique_explanation && (
                <div className="rounded-2xl border border-white/10 bg-gradient-to-br from-blue-500/10 to-cyan-500/10 backdrop-blur-xl p-6 sm:p-8">
                  <div className="flex items-start gap-4 mb-4">
                    <div className="p-3 rounded-xl bg-blue-500/20">
                      <Shield className="w-6 h-6 text-blue-400" />
                    </div>
                    <div className="flex-1">
                      <h2 className="text-xl font-bold text-white mb-2">
                        {moduleContent.misinformation_type ||
                          moduleContent.title}
                      </h2>
                      <p className="text-gray-300 leading-relaxed">
                        {moduleContent.technique_explanation}
                      </p>
                    </div>
                  </div>
                  {moduleContent.trending_score > 0 && (
                    <div className="flex items-center gap-2 mt-4 text-sm">
                      <TrendingUp className="w-4 h-4 text-yellow-400" />
                      <span className="text-yellow-400 font-medium">
                        Trending Score: {moduleContent.trending_score}/10
                      </span>
                    </div>
                  )}
                </div>
              )}

              <div className="grid gap-6 lg:grid-cols-3">
                {/* Main Content - Left Side */}
                <div className="lg:col-span-2 space-y-6">
                  {/* Red Flags */}
                  {moduleContent.red_flags &&
                    moduleContent.red_flags.length > 0 && (
                      <div className="rounded-2xl border border-red-500/20 bg-gradient-to-br from-red-500/10 to-rose-500/10 backdrop-blur-xl p-6">
                        <div className="flex items-center gap-3 mb-4">
                          <AlertTriangle className="w-5 h-5 text-red-400" />
                          <h3 className="text-lg font-bold text-white">
                            Red Flags to Watch For
                          </h3>
                        </div>
                        <ul className="space-y-3">
                          {moduleContent.red_flags.map((flag, index) => (
                            <li
                              key={index}
                              className="flex gap-3 text-gray-300"
                            >
                              <span className="text-red-400 mt-1">•</span>
                              <span>{flag}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}

                  {/* Verification Tips */}
                  {moduleContent.verification_tips &&
                    moduleContent.verification_tips.length > 0 && (
                      <div className="rounded-2xl border border-green-500/20 bg-gradient-to-br from-green-500/10 to-emerald-500/10 backdrop-blur-xl p-6">
                        <div className="flex items-center gap-3 mb-4">
                          <CheckCircle className="w-5 h-5 text-green-400" />
                          <h3 className="text-lg font-bold text-white">
                            Verification Tips
                          </h3>
                        </div>
                        <ol className="space-y-3">
                          {moduleContent.verification_tips.map((tip, index) => (
                            <li
                              key={index}
                              className="flex gap-3 text-gray-300"
                            >
                              <span className="text-green-400 font-bold min-w-[24px]">
                                {index + 1}.
                              </span>
                              <span>{tip}</span>
                            </li>
                          ))}
                        </ol>
                      </div>
                    )}

                  {/* Real-World Example */}
                  {moduleContent.example && moduleContent.example.heading && (
                    <div className="rounded-2xl border border-white/10 bg-black/40 backdrop-blur-xl p-6">
                      <h3 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
                        <Lightbulb className="w-5 h-5 text-yellow-400" />
                        Real-World Example
                      </h3>
                      <div className="space-y-4">
                        {moduleContent.example.image_url && (
                          <img
                            src={moduleContent.example.image_url}
                            alt={moduleContent.example.heading}
                            className="rounded-xl w-full max-h-64 object-cover"
                            onError={(e) => {
                              e.target.style.display = "none";
                            }}
                          />
                        )}
                        <div>
                          <h4 className="text-white font-semibold mb-2">
                            {moduleContent.example.heading}
                          </h4>
                          {moduleContent.example.claim && (
                            <div className="mb-3 p-3 rounded-lg bg-red-500/10 border border-red-500/30">
                              <p className="text-sm text-red-200 font-medium mb-1">
                                The Claim:
                              </p>
                              <p className="text-sm text-gray-300">
                                {moduleContent.example.claim}
                              </p>
                            </div>
                          )}
                          {moduleContent.example.verdict && (
                            <div className="mb-3 p-3 rounded-lg bg-green-500/10 border border-green-500/30">
                              <p className="text-sm text-green-200 font-medium mb-1">
                                The Verdict:
                              </p>
                              <p className="text-sm text-gray-300">
                                {moduleContent.example.verdict}
                              </p>
                            </div>
                          )}
                          {moduleContent.example.body && (
                            <p className="text-gray-400 text-sm leading-relaxed mb-4">
                              {moduleContent.example.body}
                            </p>
                          )}
                          {moduleContent.example.tags &&
                            moduleContent.example.tags.length > 0 && (
                              <div className="flex flex-wrap gap-2 mb-4">
                                {moduleContent.example.tags.map((tag, idx) => (
                                  <span
                                    key={idx}
                                    className="px-3 py-1 rounded-full text-xs bg-blue-500/20 text-blue-300 border border-blue-500/30"
                                  >
                                    {tag}
                                  </span>
                                ))}
                              </div>
                            )}
                          {moduleContent.example.source_url && (
                            <a
                              href={moduleContent.example.source_url}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="inline-flex items-center gap-2 text-sm text-blue-400 hover:text-blue-300 transition"
                            >
                              Read full article{" "}
                              <ExternalLink className="w-4 h-4" />
                            </a>
                          )}
                        </div>
                      </div>
                    </div>
                  )}
                </div>

                {/* Sidebar - Right Side */}
                <div className="space-y-6">
                  {/* User Action Items */}
                  {moduleContent.user_action_items &&
                    moduleContent.user_action_items.length > 0 && (
                      <div className="rounded-2xl border border-purple-500/20 bg-gradient-to-br from-purple-500/10 to-indigo-500/10 backdrop-blur-xl p-6">
                        <h3 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
                          <Lightbulb className="w-5 h-5 text-purple-400" />
                          What You Can Do
                        </h3>
                        <ul className="space-y-2">
                          {moduleContent.user_action_items.map(
                            (action, index) => (
                              <li
                                key={index}
                                className="flex gap-2 text-sm text-gray-300"
                              >
                                <span className="text-purple-400 mt-1">→</span>
                                <span>{action}</span>
                              </li>
                            )
                          )}
                        </ul>
                      </div>
                    )}

                  {/* Related Patterns */}
                  {moduleContent.related_patterns &&
                    moduleContent.related_patterns.length > 0 && (
                      <div className="rounded-2xl border border-cyan-500/20 bg-gradient-to-br from-cyan-500/10 to-blue-500/10 backdrop-blur-xl p-6">
                        <h3 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
                          <Tag className="w-5 h-5 text-cyan-400" />
                          Related Patterns
                        </h3>
                        <div className="flex flex-wrap gap-2">
                          {moduleContent.related_patterns.map(
                            (pattern, index) => (
                              <span
                                key={index}
                                className="px-3 py-1.5 rounded-lg text-sm bg-cyan-500/20 text-cyan-300 border border-cyan-500/30"
                              >
                                {pattern}
                              </span>
                            )
                          )}
                        </div>
                      </div>
                    )}

                  {/* Sources of Technique */}
                  {moduleContent.sources_of_technique &&
                    moduleContent.sources_of_technique.length > 0 && (
                      <div className="rounded-2xl border border-white/10 bg-black/40 backdrop-blur-xl p-6">
                        <h3 className="text-lg font-bold text-white mb-4">
                          Common Sources
                        </h3>
                        <ul className="space-y-2">
                          {moduleContent.sources_of_technique.map(
                            (source, index) => (
                              <li
                                key={index}
                                className="text-sm text-gray-400 flex gap-2"
                              >
                                <span className="text-gray-500">•</span>
                                <span>{source}</span>
                              </li>
                            )
                          )}
                        </ul>
                      </div>
                    )}
                </div>
              </div>
            </>
          ) : (
            <div className="rounded-2xl border border-white/10 bg-white/5 p-6 text-gray-400">
              Module content will appear here as soon as it is generated.
            </div>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-black py-12">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <h1 className="text-4xl font-bold text-white mb-2">
            Educational Modules
          </h1>
          <p className="text-gray-400">
            Learn how to identify and combat misinformation
          </p>
        </motion.div>

        {/* Filters */}
        <div className="mb-8 space-y-4">
          <div className="flex flex-col md:flex-row gap-4">
            {/* Search */}
            <div className="flex-1 relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
              <input
                type="text"
                placeholder="Search modules..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full pl-10 pr-4 py-2 bg-black text-white rounded-lg border border-gray-900 focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
          </div>

          {/* Tag Filters & For You */}
          <div className="flex flex-wrap items-center gap-3">
            {isAuthenticated && user?.domain_preferences?.length > 0 && (
              <button
                type="button"
                onClick={handleForYouClick}
                className={`inline-flex items-center gap-2 rounded-full px-4 py-2 text-xs font-semibold uppercase tracking-wide border transition-all ${
                  isForYouActive
                    ? "bg-gradient-to-r from-blue-500 to-cyan-500 text-white border-transparent shadow-lg shadow-blue-500/30"
                    : "bg-black text-blue-300 border-blue-500/40 hover:bg-blue-500/10"
                }`}
              >
                <Tag className="w-3.5 h-3.5" />
                For you
              </button>
            )}

            {allTags.length > 0 && (
              <div className="flex flex-wrap gap-2">
                {allTags.map((tag) => {
                  const isSelected = selectedTags.includes(tag);
                  return (
                    <button
                      key={tag}
                      type="button"
                      onClick={() => handleTagToggle(tag)}
                      className={`inline-flex items-center gap-2 rounded-full px-3 py-1.5 text-xs font-medium border transition-all ${
                        isSelected
                          ? "bg-gradient-to-r from-cyan-500/30 to-blue-500/30 text-cyan-200 border-cyan-400/60 shadow shadow-cyan-500/30"
                          : "bg-black text-gray-300 border-gray-700 hover:border-cyan-500/60 hover:text-cyan-200"
                      }`}
                    >
                      <span className="inline-block w-1.5 h-1.5 rounded-full bg-cyan-400" />
                      {tag}
                    </button>
                  );
                })}

                {selectedTags.length > 0 && (
                  <button
                    type="button"
                    onClick={handleClearTags}
                    className="inline-flex items-center gap-1 rounded-full px-3 py-1.5 text-xs font-medium text-gray-400 hover:text-white hover:bg-white/5 transition"
                  >
                    Clear
                  </button>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Modules Grid */}
        {loading ? (
          <div className="text-center py-12">
            <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
            <p className="text-gray-400 mt-4">Loading modules...</p>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {filteredModules.map((module, index) => (
              <motion.div
                key={module.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
              >
                <ModuleCard
                  module={module}
                  isDarkMode={true}
                  isCompleted={userProgress.completedModules.includes(
                    module.id
                  )}
                  onClick={() => handleModuleClick(module.id)}
                />
              </motion.div>
            ))}
          </div>
        )}

        {!loading && filteredModules.length === 0 && (
          <div className="text-center py-12">
            <p className="text-gray-400">
              No modules found matching your criteria.
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default Modules;
