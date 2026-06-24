import React from 'react';
import { Doughnut } from 'react-chartjs-2';
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from 'chart.js';
import { CHART_DEFAULTS, LABEL_COLORS, LABEL_NAMES } from '../utils/chartConfig';
ChartJS.register(ArcElement, Tooltip, Legend);

export default function HealthStateCard({ healthData, lastUpdate }) {
  if (!healthData) {
    return (
      <div className="card">
        <div className="card-header">Health State Overview</div>
        <div className="loading">Loading health data...</div>
      </div>
    );
  }

  const counts = healthData.prediction_counts || {};
  const currentState = lastUpdate?.prediction_name || healthData.current_state || 'Unknown';
  const accuracy = healthData.accuracy ?? 0;
  const total = healthData.total_samples || 0;
  const alertActive = lastUpdate?.alert_active || false;

  const stateClass =
    currentState === 'Healthy' ? 'healthy' :
    currentState === 'Sub-healthy' ? 'sub-healthy' : 'unhealthy';

  const chartData = {
    labels: ['Healthy', 'Sub-healthy', 'Unhealthy'],
    datasets: [{
      data: [
        counts['Healthy'] || 0,
        counts['Sub-healthy'] || 0,
        counts['Unhealthy'] || 0,
      ],
      backgroundColor: [LABEL_COLORS[0], LABEL_COLORS[1], LABEL_COLORS[2]],
      borderColor: '#1e293b',
      borderWidth: 3,
      hoverBorderColor: '#475569',
    }],
  };

  const options = {
    ...CHART_DEFAULTS,
    cutout: '65%',
    plugins: {
      legend: { display: false },
      tooltip: {
        callbacks: {
          label: (ctx) => `${ctx.label}: ${ctx.parsed} (${((ctx.parsed / total) * 100).toFixed(1)}%)`,
        },
      },
    },
  };

  return (
    <div className={`card ${alertActive ? 'alert-active' : ''}`}
         style={alertActive ? { borderColor: '#ef4444', boxShadow: '0 0 24px rgba(239,68,68,0.3)' } : {}}>
      <div className="card-header">Health State Overview</div>
      <div style={{ display: 'flex', alignItems: 'center', gap: '32px', flexWrap: 'wrap' }}>
        <div className="health-gauge">
          <Doughnut data={chartData} options={options} />
        </div>
        <div style={{ flex: 1, minWidth: 200 }}>
          <div className={`health-state-label ${stateClass}`}>
            {alertActive && '🚨 '}{currentState}
          </div>
          <div style={{ fontSize: 13, color: '#94a3b8', marginBottom: 16 }}>
            Model Accuracy: {accuracy}% · {total} Samples · Mode: {healthData.label_mode}
          </div>
          <div className="health-counts">
            {['Healthy', 'Sub-healthy', 'Unhealthy'].map((label) => (
              <span key={label}>
                <span className={`count-dot ${label[0].toLowerCase()}`} />
                {label}: {counts[label] || 0}
              </span>
            ))}
          </div>
        </div>
        {lastUpdate && (
          <div style={{ fontSize: 12, color: '#64748b', minWidth: 160, textAlign: 'right' }}>
            <div>Last Update:</div>
            <div>Subject: {lastUpdate.subject}</div>
            <div>Prediction: {lastUpdate.prediction_name}</div>
            <div>Ground Truth: {lastUpdate.label_name}</div>
          </div>
        )}
      </div>
    </div>
  );
}
