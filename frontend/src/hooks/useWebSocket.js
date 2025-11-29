import { useEffect, useRef, useState, useCallback } from "react";

/**
 * Simplified WebSocket hook that's more similar to the working HTML test
 */
export const useWebSocket = (url, options = {}) => {
  const { onOpen, onClose, onMessage, onError } = options;

  const [isConnected, setIsConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState(null);
  const [error, setError] = useState(null);

  const ws = useRef(null);
  const reconnectTimeoutRef = useRef(null);
  const reconnectAttemptsRef = useRef(0);
  const maxReconnectAttempts = 5;
  const reconnectInterval = 3000;
  const connectRef = useRef(null);

  const connect = useCallback(() => {
    if (ws.current?.readyState === WebSocket.OPEN) {
      console.log("🔌 WebSocket already connected");
      return;
    }

    console.log("🔌 Connecting to WebSocket:", url);

    try {
      ws.current = new WebSocket(url);

      ws.current.onopen = (event) => {
        console.log("✅ WebSocket connected");
        setIsConnected(true);
        setError(null);
        reconnectAttemptsRef.current = 0;

        if (onOpen) {
          onOpen(event);
        }
      };

      ws.current.onclose = (event) => {
        const reason = event.reason || "";
        console.log("🔌 WebSocket disconnected:", event.code, reason);
        setIsConnected(false);

        if (onClose) {
          onClose(event);
        }

        // Attempt to reconnect if not manually closed
        if (reconnectAttemptsRef.current < maxReconnectAttempts) {
          reconnectAttemptsRef.current += 1;
          console.log(
            `🔄 Attempting to reconnect (${reconnectAttemptsRef.current}/${maxReconnectAttempts})...`
          );

          reconnectTimeoutRef.current = setTimeout(() => {
            if (connectRef.current) {
              connectRef.current();
            }
          }, reconnectInterval);
        }
      };

      ws.current.onmessage = (event) => {
        console.log("📨 WebSocket message received:", event.data);
        setLastMessage(event.data);

        if (onMessage) {
          try {
            const data = JSON.parse(event.data);
            onMessage(data, event);
          } catch {
            onMessage(event.data, event);
          }
        }
      };

      ws.current.onerror = (event) => {
        console.error("❌ WebSocket error:", event);
        setError(`WebSocket error (readyState ${ws.current?.readyState})`);

        if (onError) {
          onError(event);
        }
      };
    } catch (error) {
      console.error("❌ Error creating WebSocket:", error);
      setError(error.message);
    }
  }, [url, onOpen, onClose, onMessage, onError, reconnectInterval, maxReconnectAttempts]);

  // Store connect function in ref for use in onclose handler
  useEffect(() => {
    connectRef.current = connect;
  }, [connect]);

  const disconnect = useCallback(() => {
    console.log("🔌 Manually disconnecting WebSocket");

    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    if (ws.current) {
      ws.current.close();
      ws.current = null;
    }

    setIsConnected(false);
  }, []);

  const sendMessage = useCallback((message) => {
    if (ws.current?.readyState === WebSocket.OPEN) {
      try {
        const messageStr =
          typeof message === "string" ? message : JSON.stringify(message);
        ws.current.send(messageStr);
        console.log("📤 WebSocket message sent:", message);
        return true;
      } catch (error) {
        console.error("❌ Error sending WebSocket message:", error);
        return false;
      }
    } else {
      console.warn("⚠️ WebSocket is not connected. Cannot send message.");
      return false;
    }
  }, []);

  const reconnect = useCallback(() => {
    console.log("🔄 Manually reconnecting WebSocket");
    reconnectAttemptsRef.current = 0;
    disconnect();
    setTimeout(connect, 100);
  }, [connect, disconnect]);

  // Initialize connection on mount
  useEffect(() => {
    const timer = setTimeout(connect, 1000); // 1 second delay

    return () => {
      clearTimeout(timer);
      disconnect();
    };
  }, [connect, disconnect]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
    };
  }, []);

  return {
    isConnected,
    lastMessage,
    error,
    connect,
    disconnect,
    reconnect,
    sendMessage,
  };
};
