import { useCallback, useEffect, useMemo, useState } from "react";
import mockRumoursData from "../assets/mockRumours.json";
import { getApiBaseUrl, getWsUrl } from "../services/api";
import { useWebSocket } from "./useWebSocket";

const transformMongoData = (posts) => {
  if (!Array.isArray(posts)) return [];

  return posts.map((post) => {
    if (post.claim && post.verification) {
      return {
        ...post,
        verification: {
          ...post.verification,
          verification_date:
            post.verification.verification_date || post.stored_at,
        },
      };
    }

    const claimText =
      post.claim?.text ||
      post.metadata?.original_verification?.claim_text ||
      post.claim?.claim_text ||
      "No claim text available";
    const verification = post.claim || {};
    const originalVerification = post.metadata?.original_verification || {};
    const details = post.metadata?.original_verification || {};

    const mapVerdict = (verdict) => {
      switch (verdict?.toLowerCase()) {
        case "true":
          return "True";
        case "false":
          return "False";
        case "uncertain":
          return "Disputed";
        default:
          return "Unverified";
      }
    };

    return {
      post_id: post.post_id || post._id,
      claim: claimText,
      summary:
        details.content_summary ||
        post.post_content?.summary ||
        "No summary available",
      platform: post.platform || "Social Media",
      Post_link: details.source || post.Post_link || "#",
      verification: {
        verified: verification.verified || false,
        verdict: mapVerdict(verification.verdict),
        message:
          originalVerification.verification?.message ||
          details.error ||
          "No verification message available",
        reasoning:
          originalVerification.verification?.message ||
          details.error ||
          "No reasoning available",
        sources: {
          links: details.source ? [details.source] : [],
          titles: ["Source"],
          count: details.source ? 1 : 0,
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
        `${getApiBaseUrl()}/mongodb/recent-posts?limit=6`
      );
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const data = await response.json();
      if (data.success && Array.isArray(data.posts)) {
        setRumours(transformMongoData(data.posts));
      } else {
        throw new Error("Unexpected API shape");
      }
    } catch (err) {
      setError(err.message);
      setRumours(transformMongoData(mockRumoursData.posts || []));
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
        setRumours((prev) => {
          const exists = prev.some((item) => item.post_id === nextPost.post_id);
          if (exists) {
            return prev;
          }
          return [nextPost, ...prev].slice(0, 10);
        });
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

