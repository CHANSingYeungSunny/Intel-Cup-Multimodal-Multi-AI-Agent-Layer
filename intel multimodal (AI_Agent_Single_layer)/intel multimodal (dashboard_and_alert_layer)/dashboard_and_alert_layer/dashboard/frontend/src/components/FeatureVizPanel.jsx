import React, { useState, useEffect } from 'react';
import { Scatter } from 'react-chartjs-2';
import {
  Chart as ChartJS, LinearScale, PointElement, Tooltip, Legend
} from 'chart.js';
import { ENDPOINTS, fetchJSON } from '../utils/api';
import { CHART_DEFAULTS, SCALES_DARK, LABEL_COLORS, LABEL_NAMES } from '../utils/chartConfig';
ChartJS.register(LinearScale, PointElement, Tooltip, Legend);

export default function FeatureVizPanel() {
  const [method, setMethod] = useState('pca');
  const [vizData, setVizData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [hiddenClasses, setHiddenClasses] = useState(new Set());

  useEffect(() => {
    setLoading(true);
    fetchJSON(`${ENDPOINTS.featureViz}?method=${method}&components=2`)
      .then((data) => {
        setVizData(data);
        setLoading(false);
      })
      .catch((e) => {
        setError(e.message);
        setLoading(false);
      });
  }, [method]);

  const toggleClass = (cls) => {
    setHiddenClasses((prev) => {
      const next = new Set(prev);
      if (next.has(cls)) next.delete(cls);
      else next.add(cls);
      return next;
    });
  };

  // Group points by class for Scatter chart
  const datasets = vizData
    ? [0, 1, 2].map((cls) => {
        const pts = vizData.points.filter((p) => p.label === cls);
        return {
          label: LABEL_NAMES[cls],
          data: pts.map((p) => ({ x: p.x, y: p.y, subject: p.subject })),
          backgroundColor: LABEL_COLORS[cls] + (hiddenClasses.has(cls) ? '10' : 'CC'),
          pointRadius: 5,
          pointHoverRadius: 8,
        };
      })
    : [];

  const options = {
    ...CHART_DEFAULTS,
    scales: SCALES_DARK,
    plugins: {
      ...CHART_DEFAULTS.plugins,
      tooltip: {
        callbacks: {
          label: (ctx) => {
            const pt = vizData?.points?.find(
              (p) => p.x === ctx.parsed.x && p.y === ctx.parsed.y
            );
            return pt
              ? `${pt.subject} · Pred: ${LABEL_NAMES[pt.prediction]} · True: ${LABEL_NAMES[pt.label]}`
              : '';
          },
        },
      },
      title: {
        display: true,
        text: vizData?.explained_variance
          ? `PCA Explained Variance: ${vizData.explained_variance.map((v) => (v * 100).toFixed(1) + '%').join(', ')}`
          : '',
        color: '#94a3b8',
        font: { size: 12 },
        position: 'bottom',
      },
    },
  };

  return (
    <div className="card">
      <div className="card-header">Feature Space Visualization</div>
      <div style={{ display: 'flex', gap: 12, marginBottom: 12, alignItems: 'center' }}>
        <button
          className="btn"
          style={{ background: method === 'pca' ? '#3b82f6' : undefined }}
          onClick={() => setMethod('pca')}
        >
          PCA
        </button>
        <button
          className="btn"
          style={{ background: method === 'tsne' ? '#3b82f6' : undefined }}
          onClick={() => setMethod('tsne')}
        >
          t-SNE
        </button>
        <span style={{ fontSize: 11, color: '#64748b', marginLeft: 8 }}>
          {vizData?.n_points ?? 0} points
        </span>
      </div>
      <div className="chart-container tall">
        {loading && <div className="loading">{method === 'tsne' ? 'Computing t-SNE...' : 'Loading...'}</div>}
        {error && <div className="error">{error}</div>}
        {!loading && !error && <Scatter data={{ datasets }} options={options} />}
      </div>
      <div className="scatter-legend">
        {[0, 1, 2].map((cls) => (
          <div
            key={cls}
            className={`legend-item ${hiddenClasses.has(cls) ? 'muted' : ''}`}
            onClick={() => toggleClass(cls)}
          >
            <span className="legend-dot" style={{ background: LABEL_COLORS[cls] }} />
            {LABEL_NAMES[cls]}
          </div>
        ))}
      </div>
    </div>
  );
}
