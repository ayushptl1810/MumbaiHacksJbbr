import { motion } from "framer-motion";
import { Clock } from "lucide-react";

const palettes = {
  true: {
    accent: "from-emerald-400/60 via-emerald-500/20 to-transparent",
    badge: "text-emerald-200 border-emerald-400 bg-emerald-900/30",
  },
  false: {
    accent: "from-rose-500/60 via-rose-500/20 to-transparent",
    badge: "text-rose-200 border-rose-400 bg-rose-900/30",
  },
  disputed: {
    accent: "from-amber-400/60 via-amber-500/20 to-transparent",
    badge: "text-amber-200 border-amber-400 bg-amber-900/30",
  },
  "mostly true": {
    accent: "from-teal-400/60 via-teal-500/20 to-transparent",
    badge: "text-teal-200 border-teal-400 bg-teal-900/30",
  },
  unverified: {
    accent: "from-slate-300/60 via-slate-500/20 to-transparent",
    badge: "text-slate-200 border-slate-400 bg-slate-900/30",
  },
};

const RumourCard = ({ post, onClick, now = new Date() }) => {
  const verdict = (post.verification?.verdict || "Unverified").toLowerCase();
  const palette = palettes[verdict] || palettes.unverified;
  const truncated =
    post.claim.length > 140 ? `${post.claim.slice(0, 140)}…` : post.claim;

  const formatTimestamp = (timestamp) => {
    if (!timestamp) return "Unknown";
    const reference = now instanceof Date ? now : new Date();
    const then = new Date(timestamp);
    const diff = reference - then;
    const minutes = Math.floor(diff / 60000);
    if (minutes < 1) return "Just now";
    if (minutes < 60) return `${minutes}m ago`;
    const hours = Math.floor(minutes / 60);
    if (hours < 24) return `${hours}h ago`;
    const days = Math.floor(hours / 24);
    if (days < 7) return `${days}d ago`;
    return then.toLocaleDateString();
  };

  return (
    <motion.button
      whileHover={{ scale: 1.015 }}
      whileTap={{ scale: 0.985 }}
      onClick={onClick}
      className="group relative w-full overflow-hidden rounded-2xl border border-white/10 bg-gradient-to-br from-white/5 via-white/0 to-transparent p-4 text-left transition-colors hover:border-white/30"
    >
      <div
        className={`pointer-events-none absolute inset-0 opacity-0 blur-2xl transition-opacity duration-300 group-hover:opacity-100 bg-gradient-to-br ${palette.accent}`}
      />

      <div className="relative z-10 flex flex-col gap-3">
        <div className="flex items-center justify-between text-[11px] uppercase tracking-[0.3em] text-gray-500">
          <span>Live rumour</span>
          <span>{formatTimestamp(post.verification?.verification_date)}</span>
        </div>

        <div className="h-[1px] w-full bg-gradient-to-r from-transparent via-white/30 to-transparent opacity-60" />

        <p className="text-sm font-medium text-gray-100 leading-relaxed">
          {truncated}
        </p>

        <div className="flex items-center justify-between text-xs text-gray-400">
          <span
            className={`rounded-full border px-3 py-1 font-semibold ${palette.badge}`}
          >
            {post.verification?.verdict || "Unverified"}
          </span>

          <span className="inline-flex items-center gap-1 text-gray-400">
            <Clock className="h-3.5 w-3.5" />
            {formatTimestamp(post.verification?.verification_date)}
          </span>
        </div>
      </div>
    </motion.button>
  );
};

export default RumourCard;
