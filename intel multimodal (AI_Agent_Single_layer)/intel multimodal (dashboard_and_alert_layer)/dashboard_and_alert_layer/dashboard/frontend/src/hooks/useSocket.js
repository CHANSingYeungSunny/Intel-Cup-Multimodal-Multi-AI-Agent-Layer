import { useEffect, useState, useRef } from 'react';
import { io } from 'socket.io-client';

// Same-origin: proxied through React dev server (port 3000) in dev,
// or served directly by Flask (port 5000) in production.
const SOCKET_URL = '';

export default function useSocket() {
  const socketRef = useRef(null);
  const [connected, setConnected] = useState(false);
  const [lastHealthUpdate, setLastHealthUpdate] = useState(null);
  const [lastAlert, setLastAlert] = useState(null);
  const [systemStatus, setSystemStatus] = useState(null);
  const [experimentChanged, setExperimentChanged] = useState(null);
  const [lastAgentAdvice, setLastAgentAdvice] = useState(null);
  const [lastAgentError, setLastAgentError] = useState(null);

  useEffect(() => {
    const socket = io(SOCKET_URL, {
      transports: ['websocket', 'polling'],
      reconnection: true,
      reconnectionAttempts: 10,
      reconnectionDelay: 1000,
    });
    socketRef.current = socket;

    socket.on('connect', () => setConnected(true));
    socket.on('disconnect', () => setConnected(false));

    socket.on('health_update', (data) => setLastHealthUpdate(data));
    socket.on('alert_triggered', (data) => setLastAlert(data));
    socket.on('alert_cleared', (data) => setLastAlert(data));
    socket.on('system_status', (data) => setSystemStatus(data));
    socket.on('experiment_changed', (data) => setExperimentChanged(data));
    socket.on('agent_advice', (data) => setLastAgentAdvice(data));
    socket.on('agent_error', (data) => {
      console.error('[Agent Error]', data);
      setLastAgentError(data);
    });

    return () => {
      socket.disconnect();
    };
  }, []);

  const emit = (event, data) => {
    if (socketRef.current) {
      socketRef.current.emit(event, data);
    }
  };

  return {
    socket: socketRef.current,
    connected,
    lastHealthUpdate,
    lastAlert,
    systemStatus,
    experimentChanged,
    lastAgentAdvice,
    lastAgentError,
    emit,
  };
}
