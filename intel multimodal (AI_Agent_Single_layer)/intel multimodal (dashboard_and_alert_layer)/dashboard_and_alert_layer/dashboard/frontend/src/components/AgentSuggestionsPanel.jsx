import React, { useState } from 'react';

const VALID_SEVERITIES = ['high', 'medium', 'low'];

/**
 * AgentSuggestionsPanel — displays AI Agent health advice.
 *
 * Receives real-time advice via SocketIO (``lastAgentAdvice``) and
 * falls back to REST data (``agentAdviceData``) on initial load.
 */
export default function AgentSuggestionsPanel({
  lastAgentAdvice,
  agentAdviceData,
  adviceHistory = [],
  loading = false,
  error = null,
}) {
  const [showContext, setShowContext] = useState(false);
  const [showHistory, setShowHistory] = useState(false);

  // Resolve advice: prefer real-time SocketIO, fall back to REST
  const advice = lastAgentAdvice || agentAdviceData?.latest_advice || null;
  const trendSummary = agentAdviceData?.trend_summary || null;

  // ---- Error state ----
  if (error) {
    return (
      <div className="card">
        <div className="card-header">🤖 AI Agent Suggestions</div>
        <div className="error">Failed to load agent advice: {error}</div>
        {adviceHistory.length > 0 && (
          <div style={{ marginTop: 12 }}>
            <span
              className="agent-context-toggle"
              onClick={() => setShowHistory(!showHistory)}
            >
              {showHistory
                ? `▾ Hide history (${adviceHistory.length} entries)`
                : `▸ Show history (${adviceHistory.length} entries)`}
            </span>
            {showHistory && <HistoryList adviceHistory={adviceHistory} />}
          </div>
        )}
      </div>
    );
  }

  // ---- Loading state ----
  if (loading && !advice) {
    return (
      <div className="card">
        <div className="card-header">🤖 AI Agent Suggestions</div>
        <div className="loading">Loading agent advice...</div>
      </div>
    );
  }

  // ---- Empty state (no advice, no history) ----
  if (!advice && adviceHistory.length === 0) {
    return (
      <div className="card">
        <div className="card-header">🤖 AI Agent Suggestions</div>
        <div className="empty">
          Waiting for agent analysis… Collecting health data from the simulation stream.
        </div>
      </div>
    );
  }

  // ---- Pending state (no current advice, but history exists) ----
  if (!advice && adviceHistory.length > 0) {
    return (
      <div className="card">
        <div className="card-header">🤖 AI Agent Suggestions</div>
        <div className="empty" style={{ padding: '12px 0' }}>
          Agent is analyzing the latest health data…
        </div>
        <div style={{ marginTop: 8 }}>
          <span
            className="agent-context-toggle"
            onClick={() => setShowHistory(!showHistory)}
          >
            {showHistory
              ? `▾ Hide history (${adviceHistory.length} entries)`
              : `▸ Show history (${adviceHistory.length} entries)`}
          </span>
          {showHistory && <HistoryList adviceHistory={adviceHistory} />}
        </div>
      </div>
    );
  }

  // ---- Normal state: render advice card ----
  const severity = advice.severity || 'low';
  const severityKey = VALID_SEVERITIES.includes(severity) ? severity : 'low';
  const severityClass = `agent-severity-badge ${severityKey}`;

  return (
    <div className="card">
      <div className="card-header">🤖 AI Agent Suggestions</div>

      <div className="agent-advice-card">
        {/* Left: severity badge + condition */}
        <div style={{ flex: 1 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
            <span className={severityClass}>{severity.toUpperCase()}</span>
            {advice.matched_rule_id && (
              <span style={{ fontSize: 11, color: 'var(--text-secondary)' }}>
                Rule: {advice.matched_rule_name || advice.matched_rule_id}
              </span>
            )}
          </div>

          <div className="agent-condition">
            {advice.possible_condition || 'No specific condition identified'}
          </div>

          <div className="agent-advice-text">
            {advice.advice || 'Continue monitoring as usual.'}
          </div>

          {/* Recommended actions */}
          {advice.actions && advice.actions.length > 0 && (
            <div style={{ marginTop: 10, display: 'flex', gap: 6, flexWrap: 'wrap' }}>
              {advice.actions.map((action) => (
                <span key={action} className="agent-action-chip">
                  {formatAction(action)}
                </span>
              ))}
            </div>
          )}

          {/* Timestamp */}
          {advice.timestamp && (
            <div style={{ fontSize: 11, color: 'var(--text-secondary)', marginTop: 8 }}>
              {formatTimestamp(advice.timestamp)}
            </div>
          )}

          {/* Context toggle */}
          {advice.context && (
            <div>
              <span
                className="agent-context-toggle"
                onClick={() => setShowContext(!showContext)}
              >
                {showContext ? '▾ Hide context' : '▸ Show context'}
              </span>
              {showContext && (
                <div className="agent-context-details">
                  <ContextRow label="Current prediction" value={predictionName(advice.context.current_prediction)} />
                  <ContextRow label="Trend" value={advice.context.trend} />
                  <ContextRow label="Unhealthy ratio" value={safePercent(advice.context.unhealthy_ratio)} />
                  <ContextRow label="Healthy ratio" value={safePercent(advice.context.healthy_ratio)} />
                  <ContextRow label="HR slope" value={safeSlope(advice.context.hr_slope, 'bpm/tick')} />
                  <ContextRow label="SpO₂ slope" value={safeSlope(advice.context.spo2_slope, '%/tick')} />
                  <ContextRow label="RR slope" value={safeSlope(advice.context.rr_slope, 's/tick')} />
                </div>
              )}
            </div>
          )}
        </div>

        {/* Right: trend indicator */}
        {trendSummary && (
          <div style={{ minWidth: 120, textAlign: 'center' }}>
            <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 4 }}>
              Trend
            </div>
            <div className={`agent-trend-badge ${trendSummary.trend || 'stable'}`}>
              {trendSummary.trend || 'stable'}
            </div>
            <div style={{ fontSize: 11, color: 'var(--text-secondary)', marginTop: 4 }}>
              {trendSummary.trend_window_size || 0} obs
            </div>
          </div>
        )}
      </div>

      {/* History toggle */}
      {adviceHistory.length > 0 && (
        <div style={{ marginTop: 16 }}>
          <span
            className="agent-context-toggle"
            onClick={() => setShowHistory(!showHistory)}
          >
            {showHistory
              ? `▾ Hide history (${adviceHistory.length} entries)`
              : `▸ Show history (${adviceHistory.length} entries)`}
          </span>
          {showHistory && <HistoryList adviceHistory={adviceHistory} />}
        </div>
      )}
    </div>
  );
}

// -----------------------------------------------------------------------
// Sub-components
// -----------------------------------------------------------------------

function HistoryList({ adviceHistory }) {
  return (
    <div className="agent-history">
      {adviceHistory.map((entry) => (
        <div
          key={(entry.timestamp || '') + '_' + (entry.matched_rule_id || 'default')}
          className={`agent-history-entry ${entry.severity || 'low'}`}
        >
          <div style={{ fontWeight: 600 }}>
            <span
              className={`agent-severity-badge ${entry.severity || 'low'}`}
              style={{ fontSize: 10, padding: '1px 8px', marginRight: 8 }}
            >
              {(entry.severity || 'low').toUpperCase()}
            </span>
            {entry.possible_condition}
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-secondary)', marginTop: 2 }}>
            {entry.advice?.slice(0, 100)}{(entry.advice?.length > 100) ? '…' : ''}
          </div>
          {entry.timestamp && (
            <div style={{ fontSize: 10, color: 'var(--text-secondary)', marginTop: 2 }}>
              {formatTimestamp(entry.timestamp)}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

function ContextRow({ label, value }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
      <span style={{ color: 'var(--text-secondary)' }}>{label}</span>
      <span style={{ fontWeight: 600 }}>{value ?? '—'}</span>
    </div>
  );
}

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

function predictionName(p) {
  const names = { 0: 'Healthy', 1: 'Sub-healthy', 2: 'Unhealthy' };
  return names[p] ?? String(p);
}

function formatAction(action) {
  return action
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

function formatTimestamp(ts) {
  if (!ts) return '';
  try {
    return new Date(ts).toLocaleString();
  } catch {
    return ts;
  }
}

/** Safely format a ratio (0–1) as a percentage string. */
function safePercent(ratio) {
  if (ratio == null) return 'N/A';
  return (ratio * 100).toFixed(0) + '%';
}

/** Safely format a slope value with units. */
function safeSlope(slope, unit) {
  if (slope == null) return 'N/A';
  return slope.toFixed(2) + ' ' + unit;
}
