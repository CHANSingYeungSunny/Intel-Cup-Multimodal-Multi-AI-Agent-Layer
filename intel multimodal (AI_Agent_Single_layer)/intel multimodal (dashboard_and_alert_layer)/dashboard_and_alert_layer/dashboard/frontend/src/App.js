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
  const {
    data: liveSensorsData,
    loading: liveSensorsLoading,
    error: liveSensorsError,
    refetch: refetchLiveSensors,
  } = useApi(ENDPOINTS.liveSensors);
  const {
    data: liveSummaryData,
    loading: liveSummaryLoading,
    error: liveSummaryError,
    refetch: refetchLiveSummary,
  } = useApi(ENDPOINTS.liveSummary);
  const {
    data: microphoneLevelData,
    loading: microphoneLevelLoading,
    error: microphoneLevelError,
    refetch: refetchMicrophoneLevel,
  } = useApi(ENDPOINTS.microphoneLevel);
  const { data: diseaseData, refetch: refetchDisease } = useApi(ENDPOINTS.diseaseClassification);
  const { data: agentAdviceData, loading: agentLoading, error: agentError, refetch: refetchAgent } = useApi(ENDPOINTS.agentAdvice);

  const [activeExperimentId, setActiveExperimentId] = useState(1);
  const [alertLog, setAlertLog] = useState([]);
  const [agentAdviceLog, setAgentAdviceLog] = useState([]);
  const [simSpeed, setSimSpeed] = useState(1);
  const [simPaused, setSimPaused] = useState(false);
  const [refreshKey, setRefreshKey] = useState(0);

  const initialSyncDone = useRef(false);
  useEffect(() => {
    if (!initialSyncDone.current && experimentsData?.active_experiment_id) {
      setActiveExperimentId(experimentsData.active_experiment_id);
      initialSyncDone.current = true;
    }
  }, [experimentsData]);

  useEffect(() => {
    if (socket.lastAlert) {
      setAlertLog((prev) => [socket.lastAlert, ...prev].slice(0, 50));
    }
  }, [socket.lastAlert]);

  useEffect(() => {
    if (socket.lastAgentAdvice) {
      setAgentAdviceLog((prev) => [socket.lastAgentAdvice, ...prev].slice(0, 50));
    }
  }, [socket.lastAgentAdvice]);

  useEffect(() => {
    const intervalId = window.setInterval(() => {
      refetchLiveSensors();
      refetchLiveSummary();
    }, 20000);

    return () => window.clearInterval(intervalId);
  }, [refetchLiveSensors, refetchLiveSummary]);

  useEffect(() => {
    const intervalId = window.setInterval(() => {
      refetchMicrophoneLevel();
    }, 3000);

    return () => window.clearInterval(intervalId);
  }, [refetchMicrophoneLevel]);

  const handleExperimentChange = useCallback(async (expId) => {
    setActiveExperimentId(expId);
    try {
      await postJSON(ENDPOINTS.switchExperiment(expId));
    } catch (e) {
      socket.emit('set_experiment', { experiment_id: expId });
      await new Promise((resolve) => setTimeout(resolve, 300));
    }

    refetchHealth();
    refetchLiveSensors();
    refetchLiveSummary();
    refetchMicrophoneLevel();
    refetchDisease();
    refetchExperiments();
    refetchAgent();
    setRefreshKey((value) => value + 1);
  }, [
    socket,
    refetchHealth,
    refetchLiveSensors,
    refetchLiveSummary,
    refetchMicrophoneLevel,
    refetchDisease,
    refetchExperiments,
    refetchAgent,
  ]);

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
              {simPaused ? 'Resume' : 'Pause'}
            </button>
          </div>
        </div>
      </div>
      <Dashboard
        key={refreshKey}
        healthData={healthData}
        liveSummaryData={liveSummaryData}
        liveSummaryLoading={liveSummaryLoading}
        liveSummaryError={liveSummaryError}
        liveSensorsData={liveSensorsData}
        liveSensorsLoading={liveSensorsLoading}
        liveSensorsError={liveSensorsError}
        microphoneLevelData={microphoneLevelData}
        microphoneLevelLoading={microphoneLevelLoading}
        microphoneLevelError={microphoneLevelError}
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

