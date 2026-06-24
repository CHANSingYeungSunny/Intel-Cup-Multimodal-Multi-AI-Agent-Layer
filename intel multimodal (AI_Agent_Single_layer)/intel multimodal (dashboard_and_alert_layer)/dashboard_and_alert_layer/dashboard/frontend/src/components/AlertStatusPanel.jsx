import React from 'react';

export default function AlertStatusPanel({ alertLog, socket, onTestAlert }) {
  const lastAlert = alertLog[0];
  const alertActive = lastAlert?.type === 'triggered';

  return (
    <div className="card">
      <div className="card-header">Alert System Status</div>

      {/* Alert indicators */}
      <div className="alert-indicators">
        <div className="led-indicator">
          <span>LED:</span>
          <span className={`led-bulb ${alertActive ? 'on-red' : ''}`} />
          <span style={{ fontSize: 12, color: '#94a3b8' }}>
            {alertActive ? 'BLINKING' : 'OFF'}
          </span>
        </div>
        <div className="buzzer-indicator">
          <span>Buzzer:</span>
          <span className={`buzzer-icon ${alertActive ? 'active' : ''}`}>
            {alertActive ? '🔊' : '🔇'}
          </span>
          <span style={{ fontSize: 12, color: '#94a3b8' }}>
            {alertActive ? 'BEEPING' : 'SILENT'}
          </span>
        </div>
        <div style={{ fontSize: 13, color: '#94a3b8' }}>
          Telegram: {socket?.lastAlert ? '📤 Sent' : '✅ Ready'}
        </div>
        <button className="btn btn-danger" onClick={onTestAlert} style={{ marginLeft: 'auto' }}>
          🔔 Test Alert
        </button>
      </div>

      {/* Alert log */}
      <div className="alert-panel">
        {alertLog.length === 0 && (
          <div className="empty">No alerts yet. Alerts trigger when Unhealthy state is detected.</div>
        )}
        {alertLog.map((alert, i) => (
          <div
            key={i}
            className={`alert-entry ${alert.type === 'triggered' ? 'critical' : 'cleared'}`}
          >
            <div className="alert-icon">
              {alert.type === 'triggered' ? '🚨' : '✅'}
            </div>
            <div className="alert-content">
              <div style={{ fontWeight: 600 }}>
                {alert.type === 'triggered' ? 'ALERT: Unhealthy Detected' : 'CLEARED: System Normal'}
              </div>
              {alert.subject && (
                <div style={{ fontSize: 12, marginTop: 2 }}>
                  Subject: {alert.subject} · Level: {alert.level}
                </div>
              )}
              {alert.message && (
                <div style={{ fontSize: 12, color: '#94a3b8', marginTop: 2 }}>{alert.message}</div>
              )}
              <div className="alert-time">{formatAlertTime(alert.timestamp)}</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function formatAlertTime(ts) {
  if (!ts) return '';
  try {
    const d = new Date(ts);
    return d.toLocaleString();
  } catch {
    return ts;
  }
}
