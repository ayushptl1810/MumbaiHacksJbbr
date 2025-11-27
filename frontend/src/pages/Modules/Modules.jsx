import { useState, useEffect, useMemo } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { Search, ArrowLeft, ListChecks } from "lucide-react";
import ModuleCard from "../../components/ModuleCard";
import ContentSection from "../../components/ContentSection";
import PracticalTips from "../../components/PracticalTips";
import LoadingSpinner from "../../components/LoadingSpinner";
import { getApiBaseUrl } from "../../services/api";

const Modules = () => {
  const { id } = useParams();
  const navigate = useNavigate();
  const [modules, setModules] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState("");
  const [difficultyFilter, setDifficultyFilter] = useState("all");
  const [userProgress, setUserProgress] = useState({
    level: "beginner",
    completedModules: [],
    points: 0,
  });
  const [moduleContent, setModuleContent] = useState(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailError, setDetailError] = useState("");
  const [detailDifficulty, setDetailDifficulty] = useState("beginner");
  const [expandedSections, setExpandedSections] = useState({});

  const activeModuleMeta = useMemo(
    () => modules.find((module) => module.id === id),
    [modules, id]
  );

  useEffect(() => {
    loadModules();
  }, []);

  useEffect(() => {
    if (id && activeModuleMeta?.difficulty_levels?.length) {
      if (!activeModuleMeta.difficulty_levels.includes(detailDifficulty)) {
        setDetailDifficulty(activeModuleMeta.difficulty_levels[0]);
      }
    }
  }, [id, activeModuleMeta, detailDifficulty]);

  useEffect(() => {
    if (id) {
      loadModuleContent(id, detailDifficulty);
    }
  }, [id, detailDifficulty]);

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
      const response = await fetch(`${getApiBaseUrl()}/educational/modules`);
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

  const loadModuleContent = async (moduleId, difficultyLevel) => {
    if (!moduleId) return;
    try {
      setDetailLoading(true);
      setDetailError("");
      const response = await fetch(
        `${getApiBaseUrl()}/educational/modules/${moduleId}?difficulty_level=${difficultyLevel}`
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

  const handleDifficultyChange = (value) => {
    setDetailDifficulty(value);
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
    const targetModule = modules.find((module) => module.id === moduleId);
    if (targetModule?.difficulty_levels?.length) {
      setDetailDifficulty(targetModule.difficulty_levels[0]);
    } else {
      setDetailDifficulty("beginner");
    }
    navigate(`/modules/${moduleId}`);
  };

  const filteredModules = modules.filter((module) => {
    const matchesSearch =
      module.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      module.description.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesDifficulty =
      difficultyFilter === "all" ||
      module.difficulty_levels.includes(difficultyFilter);
    return matchesSearch && matchesDifficulty;
  });

  if (id) {
    const difficultyOptions = activeModuleMeta?.difficulty_levels?.length
      ? activeModuleMeta.difficulty_levels
      : ["beginner", "intermediate", "advanced"];
    const estimatedTime =
      moduleContent?.estimated_time ||
      activeModuleMeta?.estimated_time ||
      "10-15 mins";
    const learningObjectives = moduleContent?.learning_objectives || [];
    const examples = moduleContent?.examples || [];
    const interactive = moduleContent?.interactive_elements || {};
    const hasInteractiveContent =
      (interactive.quiz_questions?.length || 0) > 0 ||
      (interactive.true_false?.length || 0) > 0 ||
      (interactive.scenarios?.length || 0) > 0;

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
                </div>

                {difficultyOptions.length > 1 && (
                  <div className="w-full max-w-md">
                    <p className="text-xs uppercase tracking-[0.3em] text-gray-400">
                      Difficulty presets
                    </p>
                    <div className="mt-3 flex flex-wrap gap-2">
                      {difficultyOptions.map((level) => {
                        const isActive = level === detailDifficulty;
                        return (
                          <button
                            key={level}
                            onClick={() => handleDifficultyChange(level)}
                            className={`rounded-full px-4 py-1.5 text-xs font-semibold uppercase tracking-wide transition ${
                              isActive
                                ? "bg-white text-black"
                                : "bg-white/5 text-gray-300 hover:text-white/80"
                            }`}
                          >
                            {level}
                          </button>
                        );
                      })}
                    </div>
                  </div>
                )}
              </div>

              <div className="grid gap-4 md:grid-cols-3">
                {[
                  {
                    label: "Estimated time",
                    value: estimatedTime,
                    hint: "Average completion window",
                  },
                  {
                    label: "Current preset",
                    value:
                      detailDifficulty.charAt(0).toUpperCase() +
                      detailDifficulty.slice(1),
                    hint: "Content is tailored to this level",
                  },
                  {
                    label: "Objectives covered",
                    value: `${learningObjectives.length || 0} goals`,
                    hint: "Based on this preset",
                  },
                ].map((card) => (
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
                onClick={() => loadModuleContent(id, detailDifficulty)}
                className="mt-4 inline-flex items-center gap-2 rounded-lg border border-red-500/40 px-4 py-2 text-sm hover:bg-red-500/20"
              >
                Try again
              </button>
            </div>
          ) : moduleContent ? (
            <>
              <div className="grid gap-6 lg:grid-cols-3">
                <div className="lg:col-span-2 space-y-4">
                  {(moduleContent.content_sections || []).map(
                    (section, index) => (
                      <ContentSection
                        key={`${section.title}-${index}`}
                        section={section}
                        index={index}
                        isExpanded={!!expandedSections[`section-${index}`]}
                        onToggle={() => toggleSection(`section-${index}`)}
                        isDarkMode
                      />
                    )
                  )}
                </div>
                <div className="space-y-6">
                  <PracticalTips
                    tips={moduleContent.practical_tips}
                    isDarkMode
                  />
                  {Array.isArray(moduleContent.common_mistakes) &&
                    moduleContent.common_mistakes.length > 0 && (
                      <div className="rounded-3xl border border-rose-500/20 bg-gradient-to-br from-[#2b1117] via-[#1c0c10] to-[#18090a] p-5">
                        <p className="text-rose-100 font-semibold mb-3">
                          Common mistakes
                        </p>
                        <ul className="space-y-2 text-sm text-rose-200">
                          {moduleContent.common_mistakes.map(
                            (mistake, index) => (
                              <li key={index} className="flex gap-2">
                                <span className="text-rose-400">•</span>
                                <span>{mistake}</span>
                              </li>
                            )
                          )}
                        </ul>
                      </div>
                    )}
                  {examples.length > 0 && (
                    <div className="rounded-xl border border-white/10 bg-white/5 p-5 space-y-4">
                      <p className="text-white font-semibold">
                        Real-world examples
                      </p>
                      {examples.slice(0, 3).map((example, index) => (
                        <div
                          key={index}
                          className="rounded-lg bg-black/40 p-4 space-y-2 text-sm text-gray-300"
                        >
                          <p className="text-white font-semibold">
                            {example.title || `Example ${index + 1}`}
                          </p>
                          <p>{example.scenario}</p>
                          {example.red_flags?.length ? (
                            <p className="text-xs text-gray-400">
                              Red flags: {example.red_flags.join(", ")}
                            </p>
                          ) : null}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>

              {hasInteractiveContent && (
                <div className="rounded-2xl border border-white/10 bg-white/5 p-6 space-y-6">
                  <p className="text-white text-lg font-semibold">
                    Interactive drills
                  </p>
                  {renderInteractiveSection(
                    "Quiz questions",
                    interactive.quiz_questions,
                    (question, index) => (
                      <div
                        key={`quiz-${index}`}
                        className="rounded-xl border border-white/10 bg-black/40 p-4 space-y-2 text-sm text-gray-300"
                      >
                        <p className="text-white font-semibold">
                          {question.question}
                        </p>
                        <ul className="space-y-1 text-xs">
                          {(question.options || []).map(
                            (option, optionIndex) => (
                              <li
                                key={optionIndex}
                                className={`flex gap-2 ${
                                  optionIndex === question.correct_answer
                                    ? "text-green-400"
                                    : ""
                                }`}
                              >
                                <span>
                                  {String.fromCharCode(65 + optionIndex)}.
                                </span>
                                <span>{option}</span>
                              </li>
                            )
                          )}
                        </ul>
                      </div>
                    )
                  )}
                  {renderInteractiveSection(
                    "True / False",
                    interactive.true_false,
                    (statement, index) => (
                      <div
                        key={`tf-${index}`}
                        className="rounded-xl border border-white/10 bg-black/40 p-4 space-y-2 text-sm text-gray-300"
                      >
                        <p className="text-white font-semibold">
                          {statement.statement}
                        </p>
                        <p className="text-xs text-gray-400">
                          Answer: {statement.answer ? "True" : "False"}
                        </p>
                        <p className="text-xs text-gray-500">
                          {statement.explanation}
                        </p>
                      </div>
                    )
                  )}
                  {renderInteractiveSection(
                    "Scenario responses",
                    interactive.scenarios,
                    (scenario, index) => (
                      <div
                        key={`scenario-${index}`}
                        className="rounded-xl border border-white/10 bg-black/40 p-4 space-y-2 text-sm text-gray-300"
                      >
                        <p className="text-white font-semibold">
                          {scenario.scenario}
                        </p>
                        <p className="text-xs text-gray-400">
                          Question: {scenario.question}
                        </p>
                        <p className="text-xs text-green-400">
                          Recommended action: {scenario.correct_action}
                        </p>
                        <p className="text-xs text-gray-500">
                          {scenario.explanation}
                        </p>
                      </div>
                    )
                  )}
                </div>
              )}
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

            {/* Difficulty Filter */}
            <select
              value={difficultyFilter}
              onChange={(e) => setDifficultyFilter(e.target.value)}
              className="px-4 py-2 bg-black text-white rounded-lg border border-gray-900 focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="all">All Difficulties</option>
              <option value="beginner">Beginner</option>
              <option value="intermediate">Intermediate</option>
              <option value="advanced">Advanced</option>
            </select>
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
