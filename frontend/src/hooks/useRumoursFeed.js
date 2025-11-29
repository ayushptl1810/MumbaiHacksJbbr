import { useCallback, useEffect, useMemo, useState } from "react";
import mockRumoursData from "../assets/mockRumours.json";
import { getApiBaseUrl, getWsUrl } from "../services/api";
import { useWebSocket } from "./useWebSocket";

const transformMongoData = (posts) => {
  if (!Array.isArray(posts)) return [];

  return posts.map((post) => {
    // Use claim.text for claim text
    const claimText = post.claim?.text || "No claim text available";
    
    // Use claim.verdict for verdict
    const claimVerdict = post.claim?.verdict || "uncertain";
    
    // Use post_content.body for body
    const bodyContent = post.post_content?.body || "No body content available";
    
    // Use post_content.summary for summary
    const summaryContent = post.post_content?.summary || "No summary available";
    
    // Use post_content.heading for original post heading
    const postHeading = post.post_content?.heading || "Original Post";
    
    // Get final source - check multiple possible locations
    const finalSource = 
      post.sources?.confirmation_sources?.[0]?.url ||
      post.sources?.misinformation_sources?.[0]?.url ||
      post.Post_link ||
      post.metadata?.original_verification?.source ||
      "#";
    
    // Use confidence_percentage directly (should be 0-100)
    const confidencePercentage = post.confidence_percentage !== undefined 
      ? post.confidence_percentage / 100  // Convert to 0-1 if it's 0-100
      : undefined;

    const mapVerdict = (verdict) => {
      switch (verdict?.toLowerCase()) {
        case "true":
          return "True";
        case "false":
          return "False";
        case "uncertain":
          return "Uncertain";
        case "disputed":
          return "Uncertain";
        default:
          return "Unverified";
      }
    };

    return {
      post_id: post.post_id || post._id,
      claim: claimText,
      claimObject: post.claim,
      postContent: post.post_content,
      summary: summaryContent,
      body: bodyContent,
      heading: postHeading,
      platform: post.platform || "Social Media",
      Post_link: finalSource,
      verification: {
        verified: post.claim?.verified || false,
        verdict: mapVerdict(claimVerdict),
        confidence: confidencePercentage,
        confidence_percentage: post.confidence_percentage,
        sources: {
          links: post.sources?.confirmation_sources?.map(s => s.url) || [],
          titles: post.sources?.confirmation_sources?.map(s => s.title) || [],
          count: post.sources?.total_sources || 0,
        },
        verification_date:
          post.verification_date || post.stored_at || new Date().toISOString(),
      },
    };
  });
};

export const useRumoursFeed = () => {
  const [rumours, setRumours] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [now, setNow] = useState(() => new Date());

  const fetchRecentPosts = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await fetch(
        `${getApiBaseUrl()}/mongodb/recent-posts?limit=20`
      );
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const data = await response.json();
      if (data.success && Array.isArray(data.posts)) {
        const transformed = transformMongoData(data.posts);
        // Filter to only show False and Uncertain verdicts
        const filtered = transformed.filter((rumour) => {
          const verdict = (rumour.verification?.verdict || "").toLowerCase();
          return verdict === "false" || verdict === "uncertain" || verdict === "disputed";
        });
        setRumours(filtered.slice(0, 6)); // Limit to 6 for display
      } else {
        throw new Error("Unexpected API shape");
      }
    } catch (err) {
      setError(err.message);
      const transformed = transformMongoData(mockRumoursData.posts || []);
      // Filter mock data too
      const filtered = transformed.filter((rumour) => {
        const verdict = (rumour.verification?.verdict || "").toLowerCase();
        return verdict === "false" || verdict === "uncertain" || verdict === "disputed";
      });
      setRumours(filtered.slice(0, 6));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchRecentPosts();
  }, [fetchRecentPosts]);

  useEffect(() => {
    const interval = setInterval(() => setNow(new Date()), 60000);
    return () => clearInterval(interval);
  }, []);

  const handleWebSocketMessage = useCallback((data) => {
      if (data.type === "new_post" && data.data?.post) {
        const [nextPost] = transformMongoData([data.data.post]);
        if (nextPost) {
          // Only add if verdict is False or Uncertain
          const verdict = (nextPost.verification?.verdict || "").toLowerCase();
          if (verdict === "false" || verdict === "uncertain" || verdict === "disputed") {
            setRumours((prev) => {
              const exists = prev.some((item) => item.post_id === nextPost.post_id);
              if (exists) {
                return prev;
              }
              return [nextPost, ...prev].slice(0, 6);
            });
          }
        }
      }
  }, []);

  const wsUrl = useMemo(() => getWsUrl(), []);

  useWebSocket(wsUrl, {
    onMessage: handleWebSocketMessage,
  });

  return {
    rumours,
    loading,
    error,
    now,
    refresh: fetchRecentPosts,
  };
};

