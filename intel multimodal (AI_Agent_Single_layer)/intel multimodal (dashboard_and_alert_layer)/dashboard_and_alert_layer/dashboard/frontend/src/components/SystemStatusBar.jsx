import React from 'react';

export default function SystemStatusBar({ connected, systemStatus, lastAgentError, onTestAlert }) {
  const alertActive = systemStatus?.alert_active || false;
  const uptime = systemStatus?.uptime || 0;
  const uptimeStr = formatUptime(uptime);
  const processed = systemStatus?.predictions_processed ?? 0;
  const alerts = systemStatus?.alerts_triggered ?? 0;

  return (
    <div className="status-bar">
      <div className="status-bar-left">
        <span>
          <span className={`status-dot ${connected ? 'online' : 'offline'}`} />
          {' '}{connected ? 'Connected' : 'Disconnected'}
        </span>
        <span>⏱ Uptime: {uptimeStr}</span>
        <span>📊 Processed: {processed}</span>
        <span>🚨 Alerts: {alerts}</span>
        <span>
          Alert:{' '}
          <span className={`alert-dot ${alertActive ? 'active' : 'inactive'}`} />
          {' '}{alertActive ? 'ACTIVE' : 'Normal'}
        </span>
        {lastAgentError && (
          <span style={{ color: 'var(--unhealthy)', fontSize: 12, fontWeight: 600 }}>
            ⚠ Agent: {lastAgentError.error || 'Error'}
          </span>
        )}
      </div>
      <div className="status-bar-right">
        <button className="btn btn-danger" onClick={onTestAlert}>
          🔔 Test Alert
        </button>
      </div>
    </div>
  );
}

function formatUptime(seconds) {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}
