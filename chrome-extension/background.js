// Background service worker for Aegis Rumour Alert extension

// Default API base URL (can be configured in extension settings)
const DEFAULT_API_BASE = 'http://localhost:8000';

// Default frontend base URL (can be configured via environment or storage)
const DEFAULT_FRONTEND_BASE = 'http://localhost:5173';

// Debounce delay for API calls (ms)
const DEBOUNCE_DELAY = 500;

// Minimum query length to trigger search
const MIN_QUERY_LENGTH = 3;

// Similarity threshold
const DEFAULT_SIMILARITY_THRESHOLD = 0.6;

// Cache for recent searches
const searchCache = new Map();
const CACHE_TTL = 5 * 60 * 1000; // 5 minutes

/**
 * Get API base URL from storage or use default
 */
async function getApiBaseUrl() {
  const result = await chrome.storage.sync.get(['apiBaseUrl']);
  return result.apiBaseUrl || DEFAULT_API_BASE;
}

/**
 * Get similarity threshold from storage or use default
 */
async function getSimilarityThreshold() {
  const result = await chrome.storage.sync.get(['similarityThreshold']);
  return result.similarityThreshold || DEFAULT_SIMILARITY_THRESHOLD;
}

/**
 * Get frontend base URL from storage or use default
 * Checks for environment variable via storage, falls back to localhost
 */
async function getFrontendBaseUrl() {
  const result = await chrome.storage.sync.get(['frontendBaseUrl']);
  return result.frontendBaseUrl || DEFAULT_FRONTEND_BASE;
}

/**
 * Check if query is cached and still valid
 */
function getCachedResult(query) {
  const cached = searchCache.get(query);
  if (cached && Date.now() - cached.timestamp < CACHE_TTL) {
    return cached.data;
  }
  return null;
}

/**
 * Cache search result
 */
function cacheResult(query, data) {
  searchCache.set(query, {
    data,
    timestamp: Date.now()
  });
}

/**
 * Search for similar rumours
 */
async function searchSimilarRumours(query) {
  if (!query || query.trim().length < MIN_QUERY_LENGTH) {
    return null;
  }

  // Check cache first
  const cached = getCachedResult(query);
  if (cached) {
    return cached;
  }

  try {
    const apiBaseUrl = await getApiBaseUrl();
    const threshold = await getSimilarityThreshold();
    
    const url = `${apiBaseUrl}/mongodb/search-similar?query=${encodeURIComponent(query)}&similarity_threshold=${threshold}&limit=5`;
    
    const response = await fetch(url, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json'
      }
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    
    if (data.success && data.results && data.results.length > 0) {
      // Cache the result
      cacheResult(query, data);
      return data;
    }
    
    return null;
  } catch (error) {
    // Silently handle errors
    return null;
  }
}

/**
 * Handle messages from content script
 */
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'searchSimilar') {
    searchSimilarRumours(request.query)
      .then(result => {
        sendResponse({ success: true, data: result });
      })
      .catch(error => {
        sendResponse({ success: false, error: error.message });
      });
    return true; // Indicates we will send a response asynchronously
  }
  
  if (request.action === 'getSettings') {
    Promise.all([
      getApiBaseUrl(),
      getSimilarityThreshold()
    ]).then(([apiUrl, threshold]) => {
      sendResponse({ success: true, apiUrl, threshold });
    });
    return true;
  }
  
  if (request.action === 'injectContentScript') {
    // Programmatically inject content script if declarative injection fails
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      if (tabs[0]) {
        chrome.scripting.executeScript({
          target: { tabId: tabs[0].id },
          files: ['content.js']
        }).then(() => {
          sendResponse({ success: true });
        }).catch(error => {
          sendResponse({ success: false, error: error.message });
        });
      }
    });
    return true;
  }
});

/**
 * Create context menu item
 */
function createContextMenu() {
  chrome.contextMenus.create({
    id: 'verifyWithAegis',
    title: 'Verify with Aegis',
    contexts: ['selection']
  });
}

/**
 * Initialize context menu on extension install or startup
 */
chrome.runtime.onInstalled.addListener(() => {
  createContextMenu();
});

// Also create on startup (in case it was removed)
chrome.runtime.onStartup.addListener(() => {
  createContextMenu();
});

// Create immediately if already running
createContextMenu();

/**
 * Handle context menu clicks
 */
chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId === 'verifyWithAegis' && info.selectionText) {
    const frontendBaseUrl = await getFrontendBaseUrl();
    const selectedText = info.selectionText.trim();
    
    if (selectedText) {
      const verifyUrl = `${frontendBaseUrl}/verify?text=${encodeURIComponent(selectedText)}`;
      chrome.tabs.create({ url: verifyUrl });
    }
  }
});

/**
 * Clean up old cache entries periodically
 */
setInterval(() => {
  const now = Date.now();
  for (const [query, cached] of searchCache.entries()) {
    if (now - cached.timestamp >= CACHE_TTL) {
      searchCache.delete(query);
    }
  }
}, 60000); // Run every minute

