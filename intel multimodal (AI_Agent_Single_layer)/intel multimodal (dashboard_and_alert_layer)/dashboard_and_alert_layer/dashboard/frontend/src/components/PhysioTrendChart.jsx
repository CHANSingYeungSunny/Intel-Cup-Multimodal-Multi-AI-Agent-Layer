import React, { useState, useEffect } from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS, CategoryScale, LinearScale, PointElement,
  LineElement, Title, Tooltip, Legend
} from 'chart.js';
import { ENDPOINTS, fetchJSON } from '../utils/api';
import { CHART_DEFAULTS, SCALES_DARK, classColor, className } from '../utils/chartConfig';
ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

export default function PhysioTrendChart() {
  const [subjects, setSubjects] = useState([]);
  const [selectedSubject, setSelectedSubject] = useState('');
  const [trendData, setTrendData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchJSON('/api/health_history')
      .then((data) => {
        const subs = [...new Set(data.predictions.map((p) => {
          const m = p.filename.match(/subject(\d+)/);
          return m ? `subject${m[1]}` : null;
        }).filter(Boolean))].sort();
        setSubjects(subs);
        if (subs.length > 0) setSelectedSubject(subs[0]);
      })
      .catch((e) => setError(e.message));
  }, []);

  useEffect(() => {
    if (!selectedSubject) return;
    setLoading(true);
    fetchJSON(`${ENDPOINTS.physioTrend}?subject=${selectedSubject}`)
      .then((data) => {
        setTrendData(data);
        setLoading(false);
      })
      .catch((e) => {
        setError(e.message);
        setLoading(false);
      });
  }, [selectedSubject]);

  const chartData = trendData ? {
    labels: trendData.window_indices.map((w) => `Win ${w}`),
    datasets: [
      {
        label: 'Heart Rate (bpm)',
        data: trendData.heart_rate_sim,
        borderColor: '#ef4444',
        backgroundColor: '#ef4444',
        tension: 0.3,
        pointRadius: 5,
        pointBackgroundColor: trendData.predictions.map((p) => classColor(p)),
        pointBorderColor: trendData.predictions.map((p) => classColor(p)),
        pointBorderWidth: 2,
        yAxisID: 'y',
      },
      {
        label: 'SpO2 (%)',
        data: trendData.spo2_sim,
        borderColor: '#3b82f6',
        backgroundColor: '#3b82f6',
        tension: 0.3,
        pointRadius: 5,
        pointBackgroundColor: trendData.predictions.map((p) => classColor(p)),
        pointBorderColor: trendData.predictions.map((p) => classColor(p)),
        pointBorderWidth: 2,
        yAxisID: 'y1',
      },
      {
        label: 'RR Interval (s)',
        data: trendData.rr_interval_sim,
        borderColor: '#22c55e',
        backgroundColor: '#22c55e',
        tension: 0.3,
        pointRadius: 5,
        pointBackgroundColor: trendData.predictions.map((p) => classColor(p)),
        pointBorderColor: trendData.predictions.map((p) => classColor(p)),
        pointBorderWidth: 2,
        yAxisID: 'y2',
      },
    ],
  } : null;

  const options = {
    ...CHART_DEFAULTS,
    scales: {
      ...SCALES_DARK,
      x: { ...SCALES_DARK.x, title: { display: true, text: 'Window', color: '#94a3b8' } },
      y: {
        ...SCALES_DARK.y,
        type: 'linear',
        position: 'left',
        title: { display: true, text: 'Heart Rate (bpm)', color: '#ef4444' },
        min: 50, max: 120,
      },
      y1: {
        type: 'linear',
        position: 'right',
        title: { display: true, text: 'SpO2 (%)', color: '#3b82f6' },
        min: 80, max: 100,
        ticks: { color: '#94a3b8' },
        grid: { drawOnChartArea: false },
      },
      y2: {
        type: 'linear',
        position: 'right',
        title: { display: true, text: 'RR Interval (s)', color: '#22c55e' },
        min: 0.4, max: 1.4,
        ticks: { color: '#94a3b8' },
        grid: { drawOnChartArea: false },
      },
    },
  };

  return (
    <div className="card">
      <div className="card-header">Physiological Signal Trends</div>
      <div className="subject-selector">
        Subject:
        <select value={selectedSubject} onChange={(e) => setSelectedSubject(e.target.value)}>
          {subjects.map((s) => <option key={s} value={s}>{s}</option>)}
        </select>
        <span style={{ fontSize: 11, color: '#64748b', flex: 1, textAlign: 'right' }}>
          Point colors: {className(0)} | {className(1)} | {className(2)}
        </span>
      </div>
      <div className="chart-container tall">
        {loading && <div className="loading">Loading...</div>}
        {error && <div className="error">{error}</div>}
        {!loading && !error && chartData && <Line data={chartData} options={options} />}
        {!loading && !error && !chartData && <div className="empty">No data available</div>}
      </div>
    </div>
  );
}
