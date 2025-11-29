// Content script to monitor search bar input

// Debounce timer
let debounceTimer = null;
const DEBOUNCE_DELAY = 500; // ms

// Minimum query length
const MIN_QUERY_LENGTH = 3;

// Alert container
let alertContainer = null;

/**
 * Create alert container if it doesn't exist
 */
function createAlertContainer() {
  if (alertContainer && document.body.contains(alertContainer)) {
    return alertContainer;
  }

  const container = document.createElement("div");
  container.id = "aegis-rumour-alert-container";
  container.style.cssText = `
    position: fixed;
    top: 20px;
    right: 20px;
    z-index: 10000;
    max-width: 400px;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
  `;

  document.body.appendChild(container);
  alertContainer = container;
  return container;
}

/**
 * Remove alert container
 */
function removeAlertContainer() {
  if (alertContainer && document.body.contains(alertContainer)) {
    alertContainer.remove();
    alertContainer = null;
  }
}

/**
 * Show alert with rumour information
 */
function showAlert(rumourData) {
  const container = createAlertContainer();

  // Clear existing alerts
  container.innerHTML = "";

  rumourData.results.forEach((rumour, index) => {
    const alert = document.createElement("div");
    alert.className = "aegis-rumour-alert";
    alert.style.cssText = `
      background: rgba(15, 23, 42, 0.95);
      backdrop-filter: blur(12px);
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-left: 4px solid ${
        rumour.verification?.verdict === "false"
          ? "#ef4444"
          : rumour.verification?.verdict === "true"
          ? "#3b82f6"
          : "#f59e0b"
      };
      border-radius: 12px;
      padding: 16px;
      margin-bottom: 12px;
      box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);
      animation: slideIn 0.3s ease-out;
    `;

    const claim =
      typeof rumour.claim === "string"
        ? rumour.claim
        : rumour.claim?.text ||
          rumour.claim?.claim_text ||
          "No claim available";
    const verdict = rumour.verification?.verdict || "Unknown";
    const message =
      rumour.verification?.message ||
      rumour.verification?.reasoning ||
      "No verification message";
    const similarity = (rumour.similarity_score * 100).toFixed(1);

    alert.innerHTML = `
      <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 8px;">
        <h3 style="margin: 0; font-size: 14px; font-weight: 600; color: #f9fafb; flex: 1;">
          Similar Rumour Found (${similarity}% match)
        </h3>
        <button class="aegis-close-btn" style="
          background: none;
          border: none;
          font-size: 18px;
          cursor: pointer;
          color: #9ca3af;
          padding: 0;
          margin-left: 8px;
          transition: color 0.2s;
        " onmouseover="this.style.color='#f9fafb'" onmouseout="this.style.color='#9ca3af'">×</button>
      </div>
      <p style="margin: 0 0 8px 0; font-size: 13px; color: #d1d5db; line-height: 1.4;">
        <strong style="color: #f9fafb;">Claim:</strong> ${claim.substring(0, 150)}${
      claim.length > 150 ? "..." : ""
    }
      </p>
      <div style="margin-bottom: 8px;">
        <span style="
          display: inline-block;
          padding: 6px 12px;
          border-radius: 6px;
          font-size: 12px;
          font-weight: 500;
          background: ${
            verdict === "false"
              ? "rgba(239, 68, 68, 0.2)"
              : verdict === "true"
              ? "rgba(59, 130, 246, 0.2)"
              : "rgba(245, 158, 11, 0.2)"
          };
          border: 1px solid ${
            verdict === "false"
              ? "rgba(239, 68, 68, 0.4)"
              : verdict === "true"
              ? "rgba(59, 130, 246, 0.4)"
              : "rgba(245, 158, 11, 0.4)"
          };
          color: ${
            verdict === "false"
              ? "#fca5a5"
              : verdict === "true"
              ? "#93c5fd"
              : "#fbbf24"
          };
        ">
          ${verdict.charAt(0).toUpperCase() + verdict.slice(1)}
        </span>
      </div>
      <p style="margin: 0; font-size: 12px; color: #9ca3af; line-height: 1.4;">
        ${message.substring(0, 100)}${message.length > 100 ? "..." : ""}
      </p>
      ${
        rumour.Post_link
          ? `
        <a href="${rumour.Post_link}" target="_blank" style="
          display: inline-block;
          margin-top: 8px;
          font-size: 12px;
          color: #60a5fa;
          text-decoration: none;
          transition: color 0.2s;
        " onmouseover="this.style.color='#93c5fd'" onmouseout="this.style.color='#60a5fa'">View Source →</a>
      `
          : ""
      }
    `;

    // Add close button handler
    const closeBtn = alert.querySelector(".aegis-close-btn");
    closeBtn.addEventListener("click", () => {
      alert.remove();
      if (container.children.length === 0) {
        removeAlertContainer();
      }
    });

    container.appendChild(alert);
  });

  // Add CSS animation
  if (!document.getElementById("aegis-alert-styles")) {
    const style = document.createElement("style");
    style.id = "aegis-alert-styles";
    style.textContent = `
      @keyframes slideIn {
        from {
          transform: translateX(100%);
          opacity: 0;
        }
        to {
          transform: translateX(0);
          opacity: 1;
        }
      }
    `;
    document.head.appendChild(style);
  }
}

/**
 * Monitor search input fields
 */
function monitorSearchInputs() {
  // Common search input selectors (including Google-specific)
  // IMPORTANT: Google homepage uses TEXTAREA, not input!
  const searchSelectors = [
    // Google homepage - MUST be first (uses textarea!)
    'textarea[name="q"]', // Google homepage main search (textarea!)
    "textarea.gLFyf", // Google homepage class
    'form[action="/search"] textarea', // Google form textarea
    'form[role="search"] textarea', // Google form textarea
    'textarea[aria-label*="Search" i]', // Google aria
    // Google search results page (uses input)
    'input[name="q"]', // Google search results
    "input.gLFyf", // Google search page class
    'input[aria-label*="Search" i]', // Google
    // Generic search selectors
    'input[type="search"]',
    'input[name*="search"]',
    'input[name*="q"]',
    'input[placeholder*="search" i]',
    'input[aria-label*="search" i]',
    "#search",
    "#q",
    ".search-input",
    'textarea[name*="search"]',
    'textarea[name*="q"]',
    // Fallbacks
    "textarea", // Any textarea (Google homepage uses textarea)
    'input[type="text"]', // Generic text input (fallback)
  ];

  // Find all potential search inputs
  const inputs = document.querySelectorAll(searchSelectors.join(", "));

  // Fallback: Monitor the first text input if no search inputs found
  const allInputs = document.querySelectorAll("input, textarea");
  if (allInputs.length > 0 && inputs.length === 0) {
    const firstTextInput = Array.from(allInputs).find(
      (inp) =>
        (inp.tagName === "INPUT" &&
          (inp.type === "text" || inp.type === "search" || !inp.type)) ||
        inp.tagName === "TEXTAREA"
    );

    if (firstTextInput && !firstTextInput.dataset.aegisMonitored) {
      firstTextInput.dataset.aegisMonitored = "true";
      setupInputMonitoring(firstTextInput);
    }
  }

  inputs.forEach((input) => {
    // Skip if already monitored
    if (input.dataset.aegisMonitored === "true") {
      return;
    }

    input.dataset.aegisMonitored = "true";
    setupInputMonitoring(input);
  });
}

/**
 * Process a query value - check for similar rumours
 */
function processQuery(query, source = "event") {
  const trimmedQuery = query.trim();

  // Clear existing timer
  if (debounceTimer) {
    clearTimeout(debounceTimer);
  }

  // Remove alert if query is too short
  if (trimmedQuery.length < MIN_QUERY_LENGTH) {
    removeAlertContainer();
    return;
  }

  // Debounce the search
  debounceTimer = setTimeout(() => {
    // Send message to background script
    try {
      chrome.runtime.sendMessage(
        { action: "searchSimilar", query: trimmedQuery },
        (response) => {
          if (chrome.runtime.lastError) {
            // Show error to user
            const errorDiv = document.createElement("div");
            errorDiv.style.cssText =
              "position:fixed;top:50px;left:0;background:red;color:white;padding:10px;z-index:999999;";
            errorDiv.textContent = `AEGIS Error: ${chrome.runtime.lastError.message}`;
            document.body.appendChild(errorDiv);
            setTimeout(() => errorDiv.remove(), 5000);
            return;
          }

          if (response && response.success && response.data) {
            const results = response.data.results || [];
            if (results.length > 0) {
              showAlert(response.data);
            }
          }
        }
      );
    } catch (error) {
      // Silently handle errors
    }
  }, DEBOUNCE_DELAY);
}

/**
 * Setup monitoring for a specific input element
 */
function setupInputMonitoring(input) {
  let lastValue = input.value || "";
  let pollInterval = null;

  // Function to handle input changes
  const handleInput = (e) => {
    const query = e ? e.target.value : input.value;
    if (query !== lastValue) {
      lastValue = query;
      processQuery(query, "event");
    }
  };

  // Check current value immediately when setting up monitoring
  if (input.value && input.value.trim().length >= MIN_QUERY_LENGTH) {
    lastValue = input.value;
    processQuery(input.value, "initial");
  }

  // Add multiple event listeners to catch all input types
  input.addEventListener("input", handleInput);
  input.addEventListener("keyup", handleInput);
  input.addEventListener("keydown", handleInput);
  input.addEventListener("paste", (e) => {
    // Wait for paste to complete
    setTimeout(() => handleInput(e), 10);
  });
  input.addEventListener("change", handleInput);

  // Polling fallback: Check value periodically (in case events don't fire)
  // This is especially useful for Google which might set values programmatically
  pollInterval = setInterval(() => {
    const currentValue = input.value || "";
    if (currentValue !== lastValue) {
      lastValue = currentValue;
      processQuery(currentValue, "polling");
    }
  }, 1000); // Check every second

  // Store interval ID on the input so we can clear it later if needed
  input.dataset.aegisPollInterval = pollInterval;

  // Clear alert on blur if input is empty
  input.addEventListener("blur", (e) => {
    if (!e.target.value.trim()) {
      setTimeout(() => {
        removeAlertContainer();
      }, 200);
    }
  });

  // Clean up polling when input is removed
  const observer = new MutationObserver((mutations) => {
    if (!document.body.contains(input)) {
      if (pollInterval) {
        clearInterval(pollInterval);
      }
      observer.disconnect();
    }
  });
  observer.observe(document.body, { childList: true, subtree: true });
}

/**
 * Initialize monitoring
 */
function init() {
  // For Google specifically, wait longer and check multiple times
  const isGoogle = window.location.hostname.includes("google.com");
  const isGoogleHomepage =
    isGoogle &&
    (window.location.pathname === "/" || window.location.pathname === "");
  const checkInterval = isGoogle ? 200 : 1000; // Check every 200ms for Google (more frequent)
  const maxChecks = isGoogle ? 50 : 5; // Check up to 50 times for Google (10 seconds total)

  let checkCount = 0;

  // Aggressive checking for dynamic content (especially Google)
  const checkForInputs = () => {
    checkCount++;

    // Try multiple Google-specific selectors (homepage and search page)
    // Google homepage uses different structure - try all possible selectors
    const googleSelectors = [
      // Google homepage main search (MUST be first - homepage uses textarea!)
      'textarea[name="q"]', // PRIMARY: Google homepage
      "textarea.gLFyf", // Google homepage class
      'form[action="/search"] textarea[name="q"]', // Form with textarea
      'form[action*="/search"] textarea', // Any form with /search
      'textarea[aria-label*="Search" i]', // Aria label
      'textarea[aria-label*="search" i]', // Aria label lowercase
      // Google search results page (uses input)
      'input[name="q"]', // Google search results
      "input.gLFyf", // Google search page class
      'input[aria-label*="Search" i]', // Google
      'input[aria-label*="search" i]', // Google
      // Form-based selectors
      'form[action="/search"] input',
      'form[role="search"] textarea',
      'form[role="search"] input',
      // Generic search inputs
      'textarea[type="search"]',
      'input[type="search"]',
    ];

    let foundInput = null;

    // Try each selector
    for (const selector of googleSelectors) {
      try {
        const elements = document.querySelectorAll(selector);
        if (elements.length > 0) {
          // Find the first visible, interactive input
          for (const el of elements) {
            const rect = el.getBoundingClientRect();
            const isVisible =
              rect.width > 0 &&
              rect.height > 0 &&
              window.getComputedStyle(el).display !== "none";
            const isInteractive = !el.disabled && !el.readOnly;

            if (isVisible && isInteractive) {
              // For Google homepage, be less strict - accept any textarea with name="q"
              // For other pages, check size/position
              const isGoogleHomepage =
                window.location.hostname.includes("google.com") &&
                (window.location.pathname === "/" ||
                  window.location.pathname === "");

              if (
                isGoogleHomepage &&
                el.tagName === "TEXTAREA" &&
                el.name === "q"
              ) {
                // This is definitely the Google homepage search - accept it immediately
                foundInput = el;
                break;
              }

              // Additional check: make sure it's actually the main search box
              // Google homepage search is usually large and centered
              const isLarge = rect.width > 200; // Main search is usually wide
              const isCentered =
                Math.abs(rect.left + rect.width / 2 - window.innerWidth / 2) <
                window.innerWidth / 3;

              // Prioritize inputs that look like main search boxes
              if (
                isLarge ||
                isCentered ||
                selector.includes('name="q"') ||
                selector.includes("gLFyf") ||
                (el.tagName === "TEXTAREA" && isGoogleHomepage) // Any textarea on Google homepage
              ) {
                foundInput = el;
                break;
              }
            }
          }
          if (foundInput) break;
        }
      } catch (e) {
        // Invalid selector, continue
      }
    }

    if (foundInput) {
      // Set up monitoring for this specific input right away
      if (!foundInput.dataset.aegisMonitored) {
        foundInput.dataset.aegisMonitored = "true";
        setupInputMonitoring(foundInput);
      }
      // Also run full monitoring to catch any other inputs
      monitorSearchInputs();
      return; // Stop checking, we found it
    }

    // If no specific input found, try to find the main one by size/position
    // For Google homepage, be more aggressive
    if (checkCount >= 2) {
      const allTextareas = document.querySelectorAll("textarea");
      const allInputs = document.querySelectorAll(
        'input[type="text"], input[type="search"], input:not([type="hidden"]):not([type="button"]):not([type="submit"]):not([type="checkbox"]):not([type="radio"])'
      );
      const allSearchElements = [
        ...Array.from(allTextareas),
        ...Array.from(allInputs),
      ];

      // Find the largest, most visible one (likely the main search)
      const visibleElements = allSearchElements
        .filter((el) => {
          const rect = el.getBoundingClientRect();
          const style = window.getComputedStyle(el);
          const isVisible =
            rect.width > 0 &&
            rect.height > 0 &&
            style.display !== "none" &&
            style.visibility !== "hidden" &&
            style.opacity !== "0" &&
            !el.disabled &&
            !el.readOnly;

          // For Google homepage, prioritize large, centered elements
          const isGoogleHomepage =
            window.location.hostname.includes("google.com") &&
            (window.location.pathname === "/" ||
              window.location.pathname === "");

          // On Google homepage, accept any visible textarea (homepage uses textarea)
          if (isGoogleHomepage && el.tagName === "TEXTAREA") {
            return isVisible; // Accept any visible textarea on homepage
          }

          const isLarge = rect.width > 200; // Main search is usually wide
          const isCentered =
            Math.abs(rect.left + rect.width / 2 - window.innerWidth / 2) <
            window.innerWidth / 4;
          const isInViewport = rect.top >= 0 && rect.top < window.innerHeight;

          return isVisible && (isLarge || isCentered) && isInViewport;
        })
        .map((el) => {
          const rect = el.getBoundingClientRect();
          return {
            el,
            area: rect.width * rect.height,
            width: rect.width,
            height: rect.height,
            centerX: rect.left + rect.width / 2,
            centerY: rect.top + rect.height / 2,
            distanceFromCenter: Math.abs(
              rect.left + rect.width / 2 - window.innerWidth / 2
            ),
            rect: rect,
          };
        })
        .sort((a, b) => {
          // Prioritize: 1) Large size, 2) Centered position
          const aScore = a.area + (window.innerWidth - a.distanceFromCenter);
          const bScore = b.area + (window.innerWidth - b.distanceFromCenter);
          return bScore - aScore;
        });

      if (visibleElements.length > 0) {
        const mainSearch = visibleElements[0].el;
        if (!mainSearch.dataset.aegisMonitored) {
          mainSearch.dataset.aegisMonitored = "true";
          setupInputMonitoring(mainSearch);
          monitorSearchInputs();
          return; // Stop checking
        }
      }
    }

    // Continue checking if we haven't found search inputs yet
    if (checkCount < maxChecks) {
      setTimeout(checkForInputs, checkInterval);
    } else {
      monitorSearchInputs(); // Try anyway with fallback
    }
  };

  // Start aggressive checking immediately
  checkForInputs();

  // Also start checking after multiple delays to catch different load stages
  setTimeout(checkForInputs, 50);
  setTimeout(checkForInputs, 200);
  setTimeout(checkForInputs, 500);
  setTimeout(checkForInputs, 1000);

  // Also monitor existing inputs immediately
  monitorSearchInputs();

  // Monitor for dynamically added inputs (e.g., SPA navigation)
  const observer = new MutationObserver(() => {
    monitorSearchInputs();
  });

  if (document.body) {
    observer.observe(document.body, {
      childList: true,
      subtree: true,
      attributes: false,
      characterData: false,
    });
  } else {
    // Wait for body to appear
    const bodyObserver = new MutationObserver((mutations, obs) => {
      if (document.body) {
        obs.disconnect();
        init(); // Re-initialize once body appears
      }
    });
    bodyObserver.observe(document.documentElement, {
      childList: true,
      subtree: true,
    });
  }
}

// Start monitoring when DOM is ready
// For Google and other SPAs, we need to be more aggressive
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", () => {
    setTimeout(init, 100); // Small delay even after DOMContentLoaded
  });
} else {
  // Small delay to ensure dynamic content has a chance to load
  setTimeout(init, 100);
}

// Also try to initialize when window loads (for very slow pages)
window.addEventListener("load", () => {
  setTimeout(() => {
    monitorSearchInputs();
  }, 500);
});
