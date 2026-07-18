import React from 'react';
import { ENDPOINTS } from '../utils/api';
import HealthStateCard from './HealthStateCard';
import CoughCurveChart from './CoughCurveChart';
import PhysioTrendChart from './PhysioTrendChart';
import DiseaseClassification from './DiseaseClassification';
import FeatureVizPanel from './FeatureVizPanel';
import AlertStatusPanel from './AlertStatusPanel';
import AgentSuggestionsPanel from './AgentSuggestionsPanel';

export default function Dashboard({
  healthData,
  liveSummaryData,
  liveSummaryLoading,
  liveSummaryError,
  liveSensorsData,
  liveSensorsLoading,
  liveSensorsError,
  microphoneLevelData,
  microphoneLevelLoading,
  microphoneLevelError,
  diseaseData,
  lastHealthUpdate,
  alertLog,
  lastAgentAdvice,
  agentAdviceData,
  agentAdviceLog,
  agentLoading,
  agentError,
  liveInferenceData,
  liveInferenceLoading,
  liveInferenceError,
  demoOverride,
  onDemoOverride,
  socket,
  onTestAlert,
}) {
  const scd40 = liveSensorsData?.scd40 || {};
  const mlx90614 = liveSensorsData?.mlx90614 || {};
  const max30102 = liveSensorsData?.max30102 || {};
  const camera = liveSensorsData?.camera || {};
  const microphone = liveSensorsData?.microphone || {};
  const max30102Detected = Boolean(max30102.detected);
  const fingerPresent = Boolean(max30102.finger_present);
  const irAvailable = Boolean(mlx90614.detected) && mlx90614.object_temperature_c != null;
  const cameraDetected = Boolean(camera.detected);
  const microphoneDetected = Boolean(microphone.detected);

  return (
    <div className="dashboard-grid">
      <div className="full-width">
        <HealthStateCard
          healthData={healthData}
          lastUpdate={lastHealthUpdate}
        />
      </div>

      <div className="full-width">
        <LiveSummaryCard
          liveSummaryData={liveSummaryData}
          liveSummaryLoading={liveSummaryLoading}
          liveSummaryError={liveSummaryError}
        />
      </div>

      <div className="full-width">
        <CameraPreviewCard cameraDevices={camera.devices} />
      </div>

      <div className="full-width">
        <div className="card">
          <div className="card-header">Live Sensor Snapshot</div>
          {liveSensorsError ? (
            <div className="error">{liveSensorsError}</div>
          ) : liveSensorsLoading && !liveSensorsData ? (
            <div className="loading">Loading live sensors...</div>
          ) : (
            <>
              <div className="sensor-grid">
                <SensorTile
                  label="CO2 ppm"
                  value={formatMetric(scd40.co2_ppm, 'ppm', 0)}
                  detail="SCD40 /dev/i2c-1 @ 0x62"
                  tone={scd40.co2_ppm != null ? 'ok' : 'warn'}
                  status={scd40.co2_ppm != null ? 'Live' : humanizeStatus(scd40.status || 'unavailable')}
                />
                <SensorTile
                  label="Temperature"
                  value={formatMetric(scd40.temperature_c, 'C', 1)}
                  detail={scd40.error || 'Ambient from SCD40'}
                  tone={scd40.temperature_c != null ? 'ok' : 'warn'}
                  status={scd40.temperature_c != null ? 'Live' : humanizeStatus(scd40.status || 'unavailable')}
                />
                <SensorTile
                  label="Humidity"
                  value={formatMetric(scd40.humidity_percent, '%', 2)}
                  detail="Ambient relative humidity"
                  tone={scd40.humidity_percent != null ? 'ok' : 'warn'}
                  status={scd40.humidity_percent != null ? 'Live' : humanizeStatus(scd40.status || 'unavailable')}
                />
                <SensorTile
                  label="IR object temp"
                  value={irAvailable ? formatMetric(mlx90614.object_temperature_c, 'C', 1) : 'Unavailable'}
                  detail={mlx90614.status || 'Sensor not detected'}
                  tone={irAvailable ? 'ok' : 'warn'}
                  status={irAvailable ? 'Live' : 'Unavailable'}
                />
                <SensorTile
                  label="MAX30102"
                  value={!max30102Detected ? 'Unavailable' : fingerPresent ? 'Finger detected' : 'No finger'}
                  detail={!max30102Detected ? 'Sensor not detected' : humanizeStatus(max30102.ppg_status)}
                  tone={!max30102Detected ? 'error' : fingerPresent ? 'ok' : 'warn'}
                  status={!max30102Detected ? 'Offline' : fingerPresent ? 'Ready' : 'Waiting'}
                />
                <SensorTile
                  label="Camera"
                  value={cameraDetected ? 'Detected' : 'Unavailable'}
                  detail={cameraDetected ? summarizeCamera(camera.devices) : (camera.status || 'No camera found')}
                  tone={cameraDetected ? 'ok' : 'error'}
                  status={cameraDetected ? 'Online' : 'Offline'}
                />
                <SensorTile
                  label="Microphone"
                  value={microphoneDetected ? 'Detected' : 'Unavailable'}
                  detail={microphoneDetected ? summarizeMicrophone(microphone) : (microphone.status || 'No microphone found')}
                  tone={microphoneDetected ? 'ok' : 'error'}
                  status={microphoneDetected ? 'Online' : 'Offline'}
                />
              </div>
              {liveSensorsData?.timestamp && (
                <div className="sensor-footer">
                  Sensor timestamp: {formatSensorTimestamp(liveSensorsData.timestamp)}
                </div>
              )}
            </>
          )}
        </div>
      </div>

      <MicrophoneLevelCard
        microphoneLevelData={microphoneLevelData}
        microphoneLevelLoading={microphoneLevelLoading}
        microphoneLevelError={microphoneLevelError}
      />
      <CoughCurveChart />
      <PhysioTrendChart />
      <DiseaseClassification diseaseData={diseaseData} />
      <FeatureVizPanel />

      <div className="full-width">
        <DemoControlCard
          activeOverride={demoOverride}
          onOverride={onDemoOverride}
        />
      </div>

      <div className="full-width">
        <LiveInferenceCard
          data={liveInferenceData}
          loading={liveInferenceLoading}
          error={liveInferenceError}
        />
      </div>

      <div className="full-width">
        <AgentSuggestionsPanel
          lastAgentAdvice={lastAgentAdvice}
          agentAdviceData={agentAdviceData}
          adviceHistory={agentAdviceLog}
          loading={agentLoading}
          error={agentError}
        />
      </div>

      <div className="full-width">
        <AlertStatusPanel
          alertLog={alertLog}
          socket={socket}
          onTestAlert={onTestAlert}
        />
      </div>
    </div>
  );
}

function LiveSummaryCard({ liveSummaryData, liveSummaryLoading, liveSummaryError }) {
  if (liveSummaryError) {
    return (
      <div className="card summary-card">
        <div className="summary-header-row">
          <div className="card-header">Current Multimodal Status</div>
        </div>
        <div className="error">{liveSummaryError}</div>
      </div>
    );
  }

  if (liveSummaryLoading && !liveSummaryData) {
    return (
      <div className="card summary-card">
        <div className="summary-header-row">
          <div className="card-header">Current Multimodal Status</div>
        </div>
        <div className="loading">Loading live summary...</div>
      </div>
    );
  }

  const items = liveSummaryData?.items || [];
  const overallStatus = liveSummaryData?.overall_status || 'unavailable';
  const overallTone = toneForSummaryStatus(overallStatus);

  return (
    <div className="card summary-card">
      <div className="summary-header-row">
        <div className="card-header">Current Multimodal Status</div>
        <div className={`sensor-state ${overallTone}`}>{humanizeStatus(overallStatus)}</div>
      </div>
      <p className="summary-text">
        {liveSummaryData?.summary || 'Live summary is not currently available.'}
      </p>
      <div className="summary-item-list">
        {items.map((item) => (
          <div key={item.label} className="summary-item">
            <div className="summary-item-top">
              <div className="summary-item-label">{item.label}</div>
              <div className={`sensor-state ${toneForSummaryStatus(item.status)}`}>
                {humanizeStatus(item.status)}
              </div>
            </div>
            <div className="summary-item-message">{item.message}</div>
          </div>
        ))}
      </div>
      {liveSummaryData?.timestamp && (
        <div className="sensor-footer">
          Summary timestamp: {formatSensorTimestamp(liveSummaryData.timestamp)}
        </div>
      )}
    </div>
  );
}

function SensorTile({ label, value, detail, status, tone }) {
  return (
    <div className="sensor-tile">
      <div className="sensor-label">{label}</div>
      <div className="sensor-value">{value}</div>
      <div className="sensor-detail">{detail}</div>
      <div className={`sensor-state ${tone}`}>{status}</div>
    </div>
  );
}

function CameraPreviewCard({ cameraDevices }) {
  const [snapshotKey, setSnapshotKey] = React.useState(() => Date.now());
  const [streamFailed, setStreamFailed] = React.useState(false);
  const [fallbackFailed, setFallbackFailed] = React.useState(false);

  React.useEffect(() => {
    if (!streamFailed || fallbackFailed) {
      return undefined;
    }

    const intervalId = window.setInterval(() => {
      setSnapshotKey(Date.now());
    }, 7000);

    return () => window.clearInterval(intervalId);
  }, [streamFailed, fallbackFailed]);

  const deviceSummary = summarizeCamera(cameraDevices);
  const imageSrc = streamFailed ? `${ENDPOINTS.cameraSnapshot}?ts=${snapshotKey}` : ENDPOINTS.cameraStream;

  const handleImageError = () => {
    if (!streamFailed) {
      setStreamFailed(true);
      setFallbackFailed(false);
      setSnapshotKey(Date.now());
      return;
    }

    setFallbackFailed(true);
  };

  return (
    <div className="card media-card camera-card">
      <div className="card-header">Camera Preview</div>
      <div className="camera-preview-shell">
        {fallbackFailed ? (
          <div className="media-placeholder camera-preview-placeholder error">
            Camera stream unavailable, and snapshot fallback could not be loaded.
          </div>
        ) : (
          <img
            className="camera-preview camera-preview-large"
            src={imageSrc}
            alt="Live camera preview"
            onError={handleImageError}
          />
        )}
      </div>
      <div className="media-meta">
        Source: {streamFailed ? '/api/camera_snapshot fallback' : '/api/camera_stream'}{deviceSummary ? ` - ${deviceSummary}` : ''}
      </div>
      <div className="sensor-footer">
        {streamFailed ? 'Snapshot fallback refreshes every 7 seconds' : 'Live stream target: 6 FPS'}
      </div>
    </div>
  );
}

function MicrophoneLevelCard({ microphoneLevelData, microphoneLevelLoading, microphoneLevelError }) {
  if (microphoneLevelLoading && !microphoneLevelData) {
    return (
      <div className="card media-card">
        <div className="card-header">Microphone Level</div>
        <div className="loading">Sampling microphone...</div>
      </div>
    );
  }

  const hasLiveReading = Boolean(microphoneLevelData?.detected);
  const levelPercent = clampPercent(microphoneLevelData?.level_percent ?? 0);
  const levelRms = microphoneLevelData?.level_rms;
  const status = microphoneLevelError ? 'microphone_unavailable' : (microphoneLevelData?.status || 'microphone_unavailable');
  const tone = toneForMicrophoneStatus(status);
  const detailMessage = microphoneLevelError || microphoneLevelData?.error || 'USB capture not available';

  return (
    <div className="card media-card">
      <div className="card-header">Microphone Level</div>
      <div className="media-level-row">
        <div className="media-level-value">
          {hasLiveReading ? `${levelPercent.toFixed(1)}%` : 'Unavailable'}
        </div>
        <div className={`sensor-state ${tone}`}>{humanizeStatus(status)}</div>
      </div>
      <div className="mic-meter">
        <div className={`mic-meter-fill ${tone}`} style={{ width: `${hasLiveReading ? levelPercent : 0}%` }} />
      </div>
      <div className="media-meta">
        {hasLiveReading ? `RMS: ${levelRms} - Device: ${microphoneLevelData?.capture_device || 'USB capture not available'}` : detailMessage}
      </div>
      <div className="sensor-footer">Refreshes every 3 seconds</div>
    </div>
  );
}

function formatMetric(value, unit, digits = 1) {
  if (value == null) return 'Unavailable';
  return `${Number(value).toFixed(digits)} ${unit}`;
}

function humanizeStatus(value) {
  if (!value) return 'Status unavailable';
  return value
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function summarizeCamera(devices) {
  if (!Array.isArray(devices) || devices.length === 0) {
    return '';
  }
  return devices.join(', ');
}

function summarizeMicrophone(microphone) {
  const pcm = microphone?.pcm
    ?.split('\n')
    .find((line) => line.toLowerCase().includes('usb audio'));

  if (pcm) {
    return pcm.trim();
  }

  return microphone?.status || 'USB microphone status unavailable';
}

function formatSensorTimestamp(timestamp) {
  if (!timestamp) return '';

  if (typeof timestamp === 'number') {
    return new Date(timestamp * 1000).toLocaleString();
  }

  const numericTimestamp = Number(timestamp);
  if (!Number.isNaN(numericTimestamp)) {
    return new Date(numericTimestamp * 1000).toLocaleString();
  }

  const parsed = new Date(timestamp);
  if (!Number.isNaN(parsed.getTime())) {
    return parsed.toLocaleString();
  }

  return String(timestamp);
}

function clampPercent(value) {
  const numeric = Number(value);
  if (Number.isNaN(numeric)) return 0;
  return Math.max(0, Math.min(100, numeric));
}

function toneForMicrophoneStatus(status) {
  if (status === 'normal') return 'ok';
  if (status === 'quiet') return 'warn';
  return 'error';
}

function toneForSummaryStatus(status) {
  if (status === 'normal' || status === 'online' || status === 'active') return 'ok';
  if (status === 'attention' || status === 'waiting' || status === 'quiet') return 'warn';
  return 'error';
}


/** Live AI Inference card — shows real-time AI predictions from hardware. */
function LiveInferenceCard({ data, loading, error }) {
  if (error) {
    return (
      <div className="card">
        <div className="card-header">🧠 Live AI Inference</div>
        <div className="error" style={{ fontSize: 12 }}>
          AI inference unavailable: {error}
        </div>
      </div>
    );
  }

  if (loading && !data) {
    return (
      <div className="card">
        <div className="card-header">🧠 Live AI Inference</div>
        <div className="loading" style={{ fontSize: 12 }}>
          Connecting to AI inference pipeline…
        </div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="card">
        <div className="card-header">🧠 Live AI Inference</div>
        <div className="empty" style={{ fontSize: 12 }}>
          Waiting for first inference cycle…
        </div>
      </div>
    );
  }

  const healthState = data.health_state || 'Unknown';
  const prediction = data.prediction;
  const confidence = data.confidence;
  const status = data.status || 'unknown';
  const mode = data.mode || 'mock';
  const advice = data.advice;
  const anomalies = data.anomalies || [];
  const timestamp = data.timestamp;

  const severityColors = {
    Healthy: '#22c55e',
    'Sub-healthy': '#eab308',
    Unhealthy: '#ef4444',
    'Initializing...': '#6b7280',
    Unknown: '#6b7280',
  };
  const color = severityColors[healthState] || '#6b7280';

  const statusBadge = {
    ok: { label: 'Live', className: 'ok' },
    degraded: { label: 'Degraded', className: 'warn' },
    initializing: { label: 'Initializing', className: '' },
    error: { label: 'Error', className: 'error' },
  }[status] || { label: status, className: '' };

  return (
    <div className="card">
      <div className="card-header">
        🧠 Live AI Inference
        <span style={{ float: 'right', fontSize: 11, fontWeight: 400 }}>
          <span className={`status-badge ${statusBadge.className}`}>
            {statusBadge.label}
          </span>
          {' '}
          <span style={{ color: 'var(--text-secondary)' }}>
            ({mode === 'live' ? '🟢 HW' : '🟡 Mock'})
          </span>
        </span>
      </div>

      <div style={{ display: 'flex', gap: 16, alignItems: 'center', flexWrap: 'wrap' }}>
        <div style={{
          background: color,
          color: '#fff',
          borderRadius: 8,
          padding: '12px 18px',
          fontSize: 20,
          fontWeight: 700,
          minWidth: 140,
          textAlign: 'center',
        }}>
          {healthState}
        </div>

        <div style={{ flex: 1, minWidth: 200 }}>
          {prediction != null && (
            <div style={{ fontSize: 13, marginBottom: 4 }}>
              Prediction class: <strong>{prediction}</strong>
              {confidence != null && (
                <span> (confidence: {(confidence * 100).toFixed(0)}%)</span>
              )}
            </div>
          )}
          {advice && (
            <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 4 }}>
              <strong>Severity:</strong> {advice.severity || 'N/A'}
              {advice.possible_condition && (
                <> — {advice.possible_condition}</>
              )}
            </div>
          )}
          {advice && advice.advice && (
            <div style={{ fontSize: 12, lineHeight: 1.5, marginBottom: 4 }}>
              {advice.advice.slice(0, 200)}{advice.advice.length > 200 ? '…' : ''}
            </div>
          )}
          {anomalies.length > 0 && (
            <div style={{ fontSize: 11, color: '#ef4444' }}>
              ⚠️ {anomalies.length} anomaly(s) detected
            </div>
          )}
          {timestamp && (
            <div style={{ fontSize: 10, color: 'var(--text-secondary)', marginTop: 4 }}>
              {new Date(timestamp).toLocaleTimeString()}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}


/** Demo Mode Control — health state override buttons for competition demo. */
function DemoControlCard({ activeOverride, onOverride }) {
  const states = [
    { key: 'healthy', label: '🟢 Healthy', color: '#22c55e', desc: 'Normal vitals, clear audio' },
    { key: 'semi_healthy', label: '🟡 Sub-healthy', color: '#eab308', desc: 'Elevated HR, mild cough' },
    { key: 'unhealthy', label: '🔴 Unhealthy', color: '#ef4444', desc: 'High HR, low SpO₂, strong cough' },
  ];

  const isActive = (key) => activeOverride === key;

  return (
    <div className="card" style={{ borderLeft: '4px solid #8b5cf6' }}>
      <div className="card-header">
        🎮 Demo Mode
        <span style={{ float: 'right', fontSize: 11, fontWeight: 400, color: 'var(--text-secondary)' }}>
          {activeOverride ? `Active: ${activeOverride}` : 'Live Data'}
        </span>
      </div>

      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 8 }}>
        {states.map((s) => (
          <button
            key={s.key}
            onClick={() => onOverride(isActive(s.key) ? null : s.key)}
            style={{
              flex: 1,
              minWidth: 120,
              padding: '8px 12px',
              fontSize: 13,
              fontWeight: isActive(s.key) ? 700 : 400,
              border: isActive(s.key) ? `2px solid ${s.color}` : '1px solid var(--border)',
              borderRadius: 8,
              background: isActive(s.key) ? `${s.color}18` : 'transparent',
              color: isActive(s.key) ? s.color : 'var(--text-primary)',
              cursor: 'pointer',
              transition: 'all 0.15s',
            }}
          >
            {s.label}
          </button>
        ))}
      </div>

      <div style={{ fontSize: 11, color: 'var(--text-secondary)' }}>
        {activeOverride
          ? `Override active — simulator sends fixed ${activeOverride} data each tick. Click again to clear.`
          : 'Click a state to override the simulator with fixed health data for demo purposes.'}
      </div>
    </div>
  );
}

