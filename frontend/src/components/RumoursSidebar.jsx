import { motion } from "framer-motion";
import { AlertTriangle, X } from "lucide-react";
import RumourCard from "./RumourCard";

const RumoursSidebar = ({ rumours, onRumourClick, onClose }) => {
  return (
    <motion.aside
      initial={{ x: "100%" }}
      animate={{ x: 0 }}
      exit={{ x: "100%" }}
      transition={{ type: "spring", stiffness: 260, damping: 30 }}
      className="fixed top-0 right-0 z-[60] h-full w-[90vw] sm:w-[60vw] lg:w-[25vw] xl:w-[20vw] min-w-[280px] max-w-md border-l border-white/10 bg-gradient-to-b from-[#05070c] via-[#03050a] to-black shadow-[0_25px_80px_rgba(0,0,0,0.65)] flex flex-col"
    >
      <div className="flex items-center justify-between border-b border-white/10 px-5 py-4">
        <div className="flex items-center gap-2">
          <div className="rounded-full bg-orange-500/10 p-2">
            <AlertTriangle className="h-5 w-5 text-orange-400" />
          </div>
          <div>
            <p className="text-sm font-semibold text-white">Live Alerts</p>
            <p className="text-xs text-gray-400">
              {rumours.length} rumour{rumours.length === 1 ? "" : "s"} surfaced
            </p>
          </div>
        </div>
        <button
          onClick={onClose}
          className="rounded-full p-2 text-gray-400 transition hover:bg-white/10 hover:text-white"
        >
          <X className="h-4 w-4" />
        </button>
      </div>

      <div className="flex-1 overflow-y-auto px-4 py-5 space-y-3">
        {rumours.length === 0 ? (
          <div className="rounded-2xl border border-dashed border-white/10 p-6 text-center text-sm text-gray-400">
            No recent rumours. You're all clear.
          </div>
        ) : (
          rumours.map((rumour, idx) => (
            <RumourCard
              key={rumour.post_id || idx}
              post={rumour}
              now={new Date()}
              onClick={() => onRumourClick(rumour)}
            />
          ))
        )}
      </div>
    </motion.aside>
  );
};

export default RumoursSidebar;

