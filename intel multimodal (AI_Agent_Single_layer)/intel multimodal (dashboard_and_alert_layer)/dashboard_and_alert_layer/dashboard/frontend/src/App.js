import React, { useState, useEffect, useCallback, useRef } from 'react';
import useSocket from './hooks/useSocket';
import useApi from './hooks/useApi';
import { ENDPOINTS, postJSON } from './utils/api';
import SystemStatusBar from './components/SystemStatusBar';
import ExperimentSelector from './components/ExperimentSelector';
import Dashboard from './components/Dashboard';

export default function App() {
  const socket = useSocket();
  const { data: experimentsData, refetch: refetchExperiments } = useApi(ENDPOINTS.experiments);
  const { data: healthData, refetch: refetchHealth } = useApi(ENDPOINTS.healthState);
  const { data: diseaseData, refetch: refetchDisease } = useApi(ENDPOINTS.diseaseClassification);
  const { data: agentAdviceData, loading: agentLoading, error: agentError, refetch: refetchAgent } = useApi(ENDPOINTS.agentAdvice);

  const [activeExperimentId, setActiveExperimentId] = useState(1);
  const [alertLog, setAlertLog] = useState([]);
  const [agentAdviceLog, setAgentAdviceLog] = useState([]);
  const [simSpeed, setSimSpeed] = useState(1);
  const [simPaused, setSimPaused] = useState(false);
  // force child components to remount on experiment change
  const [refreshKey, setRefreshKey] = useState(0);

  // Only sync experiment ID from API on initial load, not on every refetch
  const initialSyncDone = useRef(false);
  useEffect(() => {
    if (!initialSyncDone.current && experimentsData?.active_experiment_id) {
      setActiveExperimentId(experimentsData.active_experiment_id);
      initialSyncDone.current = true;
    }
  }, [experimentsData]);

  // Track alerts from WebSocket
  useEffect(() => {
    if (socket.lastAlert) {
      setAlertLog((prev) => [socket.lastAlert, ...prev].slice(0, 50));
    }
  }, [socket.lastAlert]);

  // Track agent advice from WebSocket
  useEffect(() => {
    if (socket.lastAgentAdvice) {
      setAgentAdviceLog((prev) => [socket.lastAgentAdvice, ...prev].slice(0, 50));
    }
  }, [socket.lastAgentAdvice]);

  const handleExperimentChange = useCallback(async (expId) => {
    setActiveExperimentId(expId);
    // Switch via REST first (reliable), then notify via socket
    try {
      await postJSON(ENDPOINTS.switchExperiment(expId));
    } catch (e) {
      // Fallback: try socket if REST fails
      socket.emit('set_experiment', { experiment_id: expId });
      await new Promise((r) => setTimeout(r, 300));
    }
    // Refetch ALL data now that backend has switched
    refetchHealth();
    refetchDisease();
    refetchExperiments();
    refetchAgent();
    setRefreshKey((k) => k + 1);
  }, [socket, refetchHealth, refetchDisease, refetchExperiments, refetchAgent]);

  const handleTestAlert = () => {
    socket.emit('request_alert_test');
  };

  const handleSpeedChange = (speed) => {
    setSimSpeed(speed);
    socket.emit('set_simulation_speed', { speed });
  };

  const handlePauseToggle = () => {
    setSimPaused(!simPaused);
    socket.emit('pause_simulation');
  };

  return (
    <div className="app">
      <SystemStatusBar
        connected={socket.connected}
        systemStatus={socket.systemStatus}
        lastAgentError={socket.lastAgentError}
        onTestAlert={handleTestAlert}
      />
      <div className="app-header">
        <h1 className="app-title">Multimodal Health Monitoring</h1>
        <div className="app-controls">
          <ExperimentSelector
            experiments={experimentsData?.experiments || []}
            activeId={activeExperimentId}
            onChange={handleExperimentChange}
          />
          <div className="sim-controls">
            <label>Speed:
              <select value={simSpeed} onChange={(e) => handleSpeedChange(Number(e.target.value))}>
                <option value={0.5}>0.5x</option>
                <option value={1}>1x</option>
                <option value={2}>2x</option>
                <option value={5}>5x</option>
              </select>
            </label>
            <button className="btn" onClick={handlePauseToggle}>
              {simPaused ? '▶ Resume' : '⏸ Pause'}
            </button>
          </div>
        </div>
      </div>
      <Dashboard
        key={refreshKey}
        healthData={healthData}
        diseaseData={diseaseData}
        lastHealthUpdate={socket.lastHealthUpdate}
        alertLog={alertLog}
        lastAgentAdvice={socket.lastAgentAdvice}
        agentAdviceData={agentAdviceData}
        agentAdviceLog={agentAdviceLog}
        agentLoading={agentLoading}
        agentError={agentError}
        socket={socket}
        onTestAlert={handleTestAlert}
      />
    </div>
  );
}
