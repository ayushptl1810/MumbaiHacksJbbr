// eslint-disable-next-line no-unused-vars
import { motion, AnimatePresence } from "framer-motion";
import { useMemo, useState } from "react";
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
  uncertain: {
    gradient: "from-amber-400/30 via-amber-500/20 to-transparent",
    chipClasses:
      "text-amber-200 border-amber-400/70 bg-amber-900/40 shadow-[0_0_30px_rgba(251,191,36,0.25)]",
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
  const [summaryExpanded, setSummaryExpanded] = useState(false);
  const [bodyExpanded, setBodyExpanded] = useState(false);
  
  const verdict = useMemo(
    () => (post?.verification?.verdict || "Unverified").toLowerCase(),
    [post]
  );
  const palette = verdictTokens[verdict] || verdictTokens.unverified;

  if (!post) return null;

  // Get confidence percentage - use confidence_percentage directly if available
  const confidencePercentage = post.verification?.confidence_percentage !== undefined
    ? post.verification.confidence_percentage
    : post.verification?.confidence !== undefined
    ? Math.round(post.verification.confidence * 100)
    : null;

  // Check if summary exceeds 4 sentences
  const summaryText = post.summary || post.postContent?.summary || "No summary available";
  
  // Split text into sentences (ending with . ! or ?)
  const getSentences = (text) => {
    if (!text || text.trim() === "") return [];
    // Clean the text first
    const cleanText = text.trim();
    // Match sentences ending with . ! or ? followed by space or end of string
    // Also handle cases where punctuation might be followed by quotes or other characters
    const sentenceRegex = /[^.!?]*[.!?]+(?:\s+|$|["'")])/g;
    let matches = cleanText.match(sentenceRegex);
    if (matches && matches.length > 0) {
      // Clean up the matches
      matches = matches.map(s => s.trim()).filter(s => s.length > 0);
      return matches;
    }
    // If no sentence endings found, try splitting by newlines or treat as one sentence
    if (cleanText.includes('\n')) {
      return cleanText.split('\n').filter(s => s.trim().length > 0);
    }
    // If still no matches, treat the whole text as one sentence
    return [cleanText];
  };
  
  const summarySentences = getSentences(summaryText);
  const isSummaryLong = summarySentences.length > 4;
  
  // Get first 4 sentences if summary is long
  const getFirstFourSentences = (sentencesArray) => {
    if (sentencesArray.length === 0) return "";
    return sentencesArray.slice(0, 4).join(' ').trim();
  };
  
  const displaySummary = summaryExpanded || !isSummaryLong 
    ? summaryText 
    : getFirstFourSentences(summarySentences) + '...';

  // Body content handling
  const bodyText = post.body || post.postContent?.body || "No body content available";
  const bodySentences = getSentences(bodyText);
  const isBodyLong = bodySentences.length > 4;
  
  const displayBody = bodyExpanded || !isBodyLong
    ? bodyText
    : getFirstFourSentences(bodySentences) + '...';

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
            className="relative z-10 w-full max-w-4xl max-h-[90vh] overflow-y-auto scrollbar-hide rounded-[32px] border border-white/10 bg-gradient-to-br from-[#05070c]/95 via-[#020306]/95 to-black/95 p-8 shadow-[0_40px_120px_rgba(0,0,0,0.7)]"
            initial={{ y: 40, opacity: 0, scale: 0.98 }}
            animate={{ y: 0, opacity: 1, scale: 1 }}
            exit={{ y: 40, opacity: 0, scale: 0.98 }}
            transition={{ duration: 0.3, ease: "easeOut" }}
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-start justify-between gap-4">
              <div>
                <p className="text-xs uppercase tracking-[0.5em] text-gray-400 font-semibold">
                  Fact-check alert
                </p>
                <h2 className="mt-2 text-2xl font-bold text-white">
                  {post.platform || "Unknown source"}
                </h2>
              </div>
              <button
                onClick={onClose}
                className="rounded-full border border-white/10 p-2 text-gray-400 transition-all duration-200 hover:text-white hover:border-white/40 hover:bg-white/5"
              >
                <X className="h-5 w-5" />
              </button>
            </div>

            {/* Claim Section */}
            <div className="mt-6 rounded-2xl border border-white/10 bg-black/30 p-6 hover:border-white/20 transition-all duration-200">
              <p className="text-xs uppercase tracking-[0.4em] text-gray-400 mb-3 font-semibold">
                Claim
              </p>
              <p className="text-base text-gray-100 leading-relaxed">
                {post.claim || "No claim available"}
              </p>
            </div>

            {/* Verdict and Verified on - Split Layout (50/50) */}
            <div className="mt-6 grid grid-cols-2 gap-6">
              {/* Verdict section on the left */}
              <div className={`rounded-2xl border px-6 py-5 ${palette.chipClasses} hover:shadow-lg transition-all duration-200`}>
                <div className="flex items-start justify-between h-full">
                  <div className="flex-1 min-w-0">
                    <p className="text-[11px] uppercase tracking-[0.4em] text-white/70 mb-3 font-semibold">
                      Verdict
                    </p>
                    <p className="text-xl font-bold text-white">
                      {post.verification?.verdict || "Unverified"}
                    </p>
                  </div>
                  {/* Confidence percentage on the right */}
                  {confidencePercentage !== null && (
                    <div className="text-right ml-6 border-l border-white/20 pl-6 flex-shrink-0">
                      <p className="text-[11px] uppercase tracking-[0.4em] text-white/70 mb-3 font-semibold">
                        Confidence
                      </p>
                      <p className="text-xl font-bold text-white">
                        {confidencePercentage}%
                      </p>
                    </div>
                  )}
                </div>
              </div>
              
              {/* Verified on section on the right */}
              <div className="rounded-2xl border border-white/10 bg-white/5 px-6 py-5 hover:border-white/20 transition-all duration-200">
                <p className="text-[11px] uppercase tracking-[0.4em] text-gray-400 mb-3 font-semibold">
                  Verified on
                </p>
                <p className="text-base text-gray-100 flex items-center gap-2">
                  <Clock className="h-4 w-4 text-gray-400 flex-shrink-0" />
                  <span className="break-words">
                    {post.verification?.verification_date
                      ? new Date(post.verification.verification_date).toLocaleString()
                      : "Not verified"}
                  </span>
                </p>
              </div>
            </div>

            {/* Body and Summary - Split Layout (50/50) */}
            <div className="mt-6 grid grid-cols-2 gap-6">
              {/* Main body content on the left */}
              <div className="rounded-2xl border border-white/10 bg-white/5 p-6 backdrop-blur-sm flex flex-col hover:border-white/20 transition-all duration-200">
                <p className="text-xs uppercase tracking-[0.3em] text-gray-400 mb-4 font-semibold">
                  Body
                </p>
                <div className={`text-sm text-gray-100 leading-relaxed whitespace-pre-wrap overflow-y-auto scrollbar-hide pr-2 transition-all duration-300 ${bodyExpanded ? 'max-h-none' : 'max-h-[200px]'}`}>
                  {displayBody}
                </div>
                {isBodyLong && (
                  <button
                    onClick={() => setBodyExpanded(!bodyExpanded)}
                    className="mt-4 text-blue-400 hover:text-blue-300 text-sm font-semibold transition-all duration-200 self-start flex items-center gap-1.5 hover:gap-2 group"
                  >
                    {bodyExpanded ? (
                      <>
                        <span>Read less</span>
                        <span className="text-xs transition-transform group-hover:-translate-y-0.5">↑</span>
                      </>
                    ) : (
                      <>
                        <span>Read more</span>
                        <span className="text-xs transition-transform group-hover:translate-x-0.5">→</span>
                      </>
                    )}
                  </button>
                )}
              </div>

              {/* Summary on the right */}
              <div className="rounded-2xl border border-white/10 bg-white/5 p-6 backdrop-blur-sm flex flex-col hover:border-white/20 transition-all duration-200">
                <p className="text-xs uppercase tracking-[0.3em] text-gray-400 mb-4 font-semibold">
                  Summary
                </p>
                <div className={`text-sm text-gray-100 leading-relaxed overflow-y-auto scrollbar-hide pr-2 transition-all duration-300 ${summaryExpanded ? 'max-h-none' : 'max-h-[200px]'}`}>
                  {displaySummary}
                </div>
                {isSummaryLong && (
                  <button
                    onClick={() => setSummaryExpanded(!summaryExpanded)}
                    className="mt-4 text-blue-400 hover:text-blue-300 text-sm font-semibold transition-all duration-200 self-start flex items-center gap-1.5 hover:gap-2 group"
                  >
                    {summaryExpanded ? (
                      <>
                        <span>Read less</span>
                        <span className="text-xs transition-transform group-hover:-translate-y-0.5">↑</span>
                      </>
                    ) : (
                      <>
                        <span>Read more</span>
                        <span className="text-xs transition-transform group-hover:translate-x-0.5">→</span>
                      </>
                    )}
                  </button>
                )}

                {/* Original Post section */}
                {post.Post_link && post.Post_link !== "#" && (
                  <div className="mt-6 pt-6 border-t border-white/10">
                    <p className="text-xs uppercase tracking-[0.3em] text-gray-400 mb-3 font-semibold">
                      Original Post
                    </p>
                    <a
                      href={post.Post_link}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-blue-400 hover:text-blue-300 underline-offset-4 hover:underline text-sm font-medium break-words inline-flex items-center gap-2 group transition-all duration-200"
                    >
                      <span className="group-hover:underline">{post.heading || post.postContent?.heading || "View Original Post"}</span>
                      <ExternalLink className="h-3.5 w-3.5 opacity-70 group-hover:opacity-100 transition-all duration-200 group-hover:translate-x-0.5 group-hover:-translate-y-0.5" />
                    </a>
                  </div>
                )}
              </div>
            </div>

            {sourcesContent && (
              <div className="mt-8 pt-6 border-t border-white/10">
                <p className="text-xs uppercase tracking-[0.4em] text-gray-400 mb-4 font-semibold">
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
