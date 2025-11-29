// eslint-disable-next-line no-unused-vars
import { motion } from "framer-motion";
import { AlertTriangle, X } from "lucide-react";
import RumourCard from "./RumourCard";

const RumoursDropdown = ({ rumours, onRumourClick, onClose }) => {
  return (
    <motion.div
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
      className="absolute right-0 mt-2 w-80 bg-black border border-gray-900 rounded-lg shadow-xl z-50 max-h-96 overflow-hidden flex flex-col"
    >
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-gray-700">
        <div className="flex items-center space-x-2">
          <AlertTriangle className="w-5 h-5 text-orange-500" />
          <h3 className="text-lg font-semibold text-white">Recent Rumours</h3>
        </div>
        <button
          onClick={onClose}
          className="p-1 rounded-lg text-gray-400 hover:text-white hover:bg-gray-700 transition-colors"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      {/* Rumours List */}
      <div className="overflow-y-auto max-h-80">
        {rumours.length === 0 ? (
          <div className="p-4 text-center text-gray-400 text-sm">
            No recent rumours
          </div>
        ) : (
          <div className="p-2 space-y-2">
            {rumours.slice(0, 5).map((rumour, index) => (
              <div
                key={rumour.post_id || index}
                onClick={() => onRumourClick(rumour)}
                className="cursor-pointer"
              >
                <RumourCard
                  post={rumour}
                  isDarkMode={true}
                  now={new Date()}
                  onClick={() => onRumourClick(rumour)}
                />
              </div>
            ))}
          </div>
        )}
      </div>
    </motion.div>
  );
};

export default RumoursDropdown;

