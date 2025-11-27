import { motion, AnimatePresence } from "framer-motion";
import { useMemo } from "react";
import { X, ExternalLink, Clock } from "lucide-react";

const verdictTokens = {
  true: {
    gradient: "from-emerald-400/30 via-emerald-500/20 to-transparent",
    chipClasses:
      "text-emerald-200 border-emerald-400/70 bg-emerald-900/40 shadow-[0_0_30px_rgba(16,185,129,0.35)]",
  },
  false: {
    gradient: "from-rose-500/30 via-rose-500/20 to-transparent",
    chipClasses:
      "text-rose-200 border-rose-400/70 bg-rose-900/40 shadow-[0_0_30px_rgba(244,63,94,0.35)]",
  },
  disputed: {
    gradient: "from-amber-400/30 via-amber-500/20 to-transparent",
    chipClasses:
      "text-amber-200 border-amber-400/70 bg-amber-900/40 shadow-[0_0_30px_rgba(251,191,36,0.25)]",
  },
  "mostly true": {
    gradient: "from-teal-400/30 via-teal-500/20 to-transparent",
    chipClasses:
      "text-teal-200 border-teal-400/70 bg-teal-900/40 shadow-[0_0_30px_rgba(45,212,191,0.25)]",
  },
  unverified: {
    gradient: "from-slate-400/30 via-slate-500/20 to-transparent",
    chipClasses:
      "text-slate-200 border-slate-400/70 bg-slate-900/40 shadow-[0_0_30px_rgba(148,163,184,0.25)]",
  },
};

const SectionCard = ({ label, children }) => (
  <div className="rounded-2xl border border-white/10 bg-white/5 p-4 backdrop-blur-sm">
    <p className="text-xs uppercase tracking-[0.3em] text-gray-500 mb-2">
      {label}
    </p>
    <div className="text-sm text-gray-100 leading-relaxed">{children}</div>
  </div>
);

const RumourModal = ({ post, isOpen, onClose }) => {
  const verdict = useMemo(
    () => (post?.verification?.verdict || "Unverified").toLowerCase(),
    [post]
  );
  const palette = verdictTokens[verdict] || verdictTokens.unverified;

  if (!post) return null;

  const renderSources = () => {
    const sources = post.verification?.sources;
    if (!sources || sources.count === 0) return null;

    return (
      <div className="space-y-2">
        {sources.links.map((link, index) => (
          <a
            key={link}
            href={link}
            target="_blank"
            rel="noopener noreferrer"
            className="group flex items-center gap-3 rounded-xl border border-white/10 bg-white/5 px-4 py-2 text-sm text-blue-200 transition hover:border-white/30"
          >
            <ExternalLink className="h-4 w-4 text-blue-300" />
            <span className="truncate">{sources.titles[index] || link}</span>
            <span className="text-xs text-blue-300 opacity-70 group-hover:opacity-100">
              ↗
            </span>
          </a>
        ))}
      </div>
    );
  };

  const sourcesContent = renderSources();

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          className="fixed inset-0 z-[70] flex items-center justify-center p-4"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
        >
          <motion.div
            className="absolute inset-0 bg-black/70 backdrop-blur-md"
            onClick={onClose}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          />

          <motion.div
            className="relative z-10 w-full max-w-3xl max-h-[90vh] overflow-y-auto rounded-[32px] border border-white/10 bg-gradient-to-br from-[#05070c]/95 via-[#020306]/95 to-black/95 p-6 shadow-[0_40px_120px_rgba(0,0,0,0.7)]"
            initial={{ y: 40, opacity: 0, scale: 0.98 }}
            animate={{ y: 0, opacity: 1, scale: 1 }}
            exit={{ y: 40, opacity: 0, scale: 0.98 }}
            transition={{ duration: 0.25, ease: "easeOut" }}
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-start justify-between gap-4">
              <div>
                <p className="text-xs uppercase tracking-[0.5em] text-gray-500">
                  Fact-check alert
                </p>
                <h2 className="mt-1 text-2xl font-semibold text-white">
                  {post.platform || "Unknown source"}
                </h2>
              </div>
              <button
                onClick={onClose}
                className="rounded-full border border-white/10 p-2 text-gray-400 transition hover:text-white hover:border-white/40"
              >
                <X className="h-5 w-5" />
              </button>
            </div>

            <div className="mt-6 rounded-2xl border border-white/10 bg-black/30 p-5">
              <p className="text-xs uppercase tracking-[0.4em] text-gray-500 mb-2">
                Claim
              </p>
              <p className="text-base text-gray-100 leading-relaxed">
                {post.claim}
              </p>
            </div>

            <div className="mt-6 flex flex-wrap gap-4">
              <div
                className={`flex-1 min-w-[200px] rounded-2xl border px-5 py-4 ${palette.chipClasses}`}
              >
                <p className="text-[11px] uppercase tracking-[0.4em] text-white/70">
                  Verdict
                </p>
                <p className="mt-2 text-lg font-semibold text-white">
                  {post.verification?.verdict || "Unverified"}
                </p>
              </div>
              <div className="flex-1 min-w-[200px] rounded-2xl border border-white/10 bg-white/5 px-5 py-4">
                <p className="text-[11px] uppercase tracking-[0.4em] text-gray-500">
                  Verified on
                </p>
                <p className="mt-2 text-base text-gray-100 flex items-center gap-2">
                  <Clock className="h-4 w-4 text-gray-400" />
                  {new Date(
                    post.verification?.verification_date || Date.now()
                  ).toLocaleString()}
                </p>
              </div>
            </div>

            <div className="mt-6 grid gap-4 md:grid-cols-2">
              <SectionCard label="Summary">{post.summary}</SectionCard>
              <SectionCard label="Verification Message">
                {post.verification?.message}
              </SectionCard>
              <SectionCard label="Reasoning">
                {post.verification?.reasoning}
              </SectionCard>
              {post.Post_link && (
                <SectionCard label="Original Post">
                  <a
                    href={post.Post_link}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-blue-200 underline-offset-4 hover:underline"
                  >
                    View on {post.platform}
                  </a>
                </SectionCard>
              )}
            </div>

            {sourcesContent && (
              <div className="mt-6">
                <p className="text-xs uppercase tracking-[0.4em] text-gray-500 mb-3">
                  Sources ({post.verification.sources.count})
                </p>
                {sourcesContent}
              </div>
            )}
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};

export default RumourModal;
