import React from "react";
import { Clock, CheckCircle, Tag } from "lucide-react";
import MotionText from "./MotionText";
import MotionCard from "./MotionCard";

const badgeThemes = {
  beginner: "text-green-300 border-green-500/40",
  intermediate: "text-amber-300 border-amber-500/40",
  advanced: "text-red-300 border-red-500/40",
};

const ModuleCard = ({ module, isDarkMode, isCompleted, onClick }) => {
  return (
    <MotionCard
      className="cursor-pointer rounded-2xl border border-white/10 bg-gradient-to-br from-slate-900/70 to-slate-900/30 p-6 shadow-[0_20px_45px_rgba(0,0,0,0.45)] transition hover:-translate-y-1 hover:border-blue-400/60"
      isDarkMode={isDarkMode}
      darkBackgroundColor="#101425"
      lightBackgroundColor="#101425"
      darkBorderColor="#1f2b4a"
      lightBorderColor="#1f2b4a"
      onClick={onClick}
      whileHover={{ scale: 1.01 }}
      whileTap={{ scale: 0.99 }}
    >
      <div className="flex items-start justify-between">
        <div>
          <p className="text-xs uppercase tracking-[0.2em] text-gray-400">
            Module
          </p>
          <MotionText
            className="text-xl font-semibold text-white"
            isDarkMode={isDarkMode}
            darkColor="#ffffff"
            lightColor="#ffffff"
          >
            {module.title}
          </MotionText>
        </div>
        {isCompleted && (
          <div className="flex items-center gap-2 text-emerald-400 text-xs font-semibold">
            <CheckCircle className="h-4 w-4" />
            Completed
          </div>
        )}
      </div>

      <p className="mt-3 text-sm text-gray-400">{module.description}</p>

      <div className="mt-5 flex flex-wrap items-center gap-3 text-sm text-gray-400">
        {module.estimated_time && (
          <div className="flex items-center gap-2">
            <Clock className="h-4 w-4 text-blue-300" />
            <span>{module.estimated_time}</span>
          </div>
        )}
        {module.difficulty_levels &&
          Array.isArray(module.difficulty_levels) &&
          module.difficulty_levels.length > 0 && (
            <div className="flex flex-wrap gap-2">
              {module.difficulty_levels.map((level) => (
                <span
                  key={level}
                  className={`rounded-full border px-3 py-1 text-xs font-semibold uppercase tracking-wide ${
                    badgeThemes[level] || badgeThemes.beginner
                  }`}
                >
                  {level}
                </span>
              ))}
            </div>
          )}
      </div>

      {module.tags && Array.isArray(module.tags) && module.tags.length > 0 && (
        <div className="mt-4 flex flex-wrap gap-2">
          {module.tags.slice(0, 4).map((tag) => (
            <span
              key={tag}
              className="inline-flex items-center gap-1 rounded-full bg-gradient-to-r from-cyan-500/15 to-blue-500/15 px-2.5 py-1 text-[11px] font-medium text-cyan-200 border border-cyan-500/30"
            >
              <Tag className="w-3 h-3 text-cyan-300" />
              {tag}
            </span>
          ))}
        </div>
      )}
    </MotionCard>
  );
};

export default ModuleCard;
