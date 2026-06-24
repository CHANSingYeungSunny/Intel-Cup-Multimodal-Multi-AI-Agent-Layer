/**
 * Shared Chart.js configuration defaults and color mappings.
 */

export const LABEL_COLORS = {
  0: '#22c55e',  // Healthy - green
  1: '#eab308',  // Sub-healthy - yellow
  2: '#ef4444',  // Unhealthy - red
};

export const LABEL_NAMES = {
  0: 'Healthy',
  1: 'Sub-healthy',
  2: 'Unhealthy',
};

export const CHART_DEFAULTS = {
  responsive: true,
  maintainAspectRatio: false,
  animation: { duration: 400 },
  plugins: {
    legend: {
      labels: { color: '#cbd5e1', font: { size: 12 } },
    },
  },
};

export const SCALES_DARK = {
  x: {
    ticks: { color: '#94a3b8', font: { size: 11 } },
    grid: { color: 'rgba(148,163,184,0.1)' },
  },
  y: {
    ticks: { color: '#94a3b8', font: { size: 11 } },
    grid: { color: 'rgba(148,163,184,0.1)' },
  },
};

export function classColor(prediction) {
  return LABEL_COLORS[prediction] || '#6b7280';
}

export function className(prediction) {
  return LABEL_NAMES[prediction] || 'Unknown';
}
