/**
 * API endpoint constants and fetch helpers.
 */
const API_BASE = '';

export const ENDPOINTS = {
  healthState: `${API_BASE}/api/health_state`,
  healthHistory: `${API_BASE}/api/health_history`,
  liveSensors: `${API_BASE}/api/live_sensors`,
  liveSummary: `${API_BASE}/api/live_summary`,
  cameraSnapshot: `${API_BASE}/api/camera_snapshot`,
  cameraStream: `${API_BASE}/api/camera_stream`,
  microphoneLevel: `${API_BASE}/api/microphone_level`,
  coughCurve: `${API_BASE}/api/cough_curve`,
  physioTrend: `${API_BASE}/api/physio_trend`,
  diseaseClassification: `${API_BASE}/api/disease_classification`,
  featureViz: `${API_BASE}/api/feature_viz`,
  experiments: `${API_BASE}/api/experiments`,
  experiment: (id) => `${API_BASE}/api/experiments/${id}`,
  switchExperiment: (id) => `${API_BASE}/api/experiments/switch/${id}`,
  agentAdvice: `${API_BASE}/api/agent_advice`,
  liveInference: `${API_BASE}/api/live_inference`,
  demoOverride: `${API_BASE}/api/demo/override`,
  demoStatus: `${API_BASE}/api/demo/status`,
};

export async function fetchJSON(url) {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`HTTP ${res.status}: ${res.statusText}`);
  }
  return res.json();
}

export async function postJSON(url) {
  const res = await fetch(url, { method: 'POST' });
  if (!res.ok) {
    throw new Error(`HTTP ${res.status}: ${res.statusText}`);
  }
  return res.json();
}



