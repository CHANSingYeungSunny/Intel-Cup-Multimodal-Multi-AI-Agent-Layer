import React, { useState, useEffect } from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS, CategoryScale, LinearScale, PointElement,
  LineElement, Title, Tooltip, Legend, Filler
} from 'chart.js';
import { ENDPOINTS, fetchJSON } from '../utils/api';
import { CHART_DEFAULTS, SCALES_DARK, classColor } from '../utils/chartConfig';
ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, Filler);

export default function CoughCurveChart() {
  const [subjects, setSubjects] = useState([]);
  const [selectedSubject, setSelectedSubject] = useState('');
  const [curveData, setCurveData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Fetch subject list
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

  // Fetch curve data when subject changes
  useEffect(() => {
    if (!selectedSubject) return;
    setLoading(true);
    fetchJSON(`${ENDPOINTS.coughCurve}?subject=${selectedSubject}`)
      .then((data) => {
        setCurveData(data);
        setLoading(false);
      })
      .catch((e) => {
        setError(e.message);
        setLoading(false);
      });
  }, [selectedSubject]);

  const color = classColor(curveData?.prediction ?? 0);

  const chartData = curveData ? {
    labels: curveData.timestamps.map((t) => t.toFixed(1)),
    datasets: [
      {
        label: 'Respiratory Waveform',
        data: curveData.amplitude,
        borderColor: color,
        backgroundColor: color + '30',
        fill: true,
        tension: 0.3,
        pointRadius: 0,
        borderWidth: 2,
      },
      {
        label: 'Upper Envelope',
        data: curveData.envelope_upper,
        borderColor: color + '60',
        borderDash: [4, 4],
        borderWidth: 1,
        pointRadius: 0,
        fill: false,
      },
      {
        label: 'Lower Envelope',
        data: curveData.envelope_lower,
        borderColor: color + '60',
        borderDash: [4, 4],
        borderWidth: 1,
        pointRadius: 0,
        fill: false,
      },
    ],
  } : null;

  const options = {
    ...CHART_DEFAULTS,
    scales: {
      ...SCALES_DARK,
      x: { ...SCALES_DARK.x, title: { display: true, text: 'Time (s)', color: '#94a3b8' } },
      y: { ...SCALES_DARK.y, title: { display: true, text: 'Amplitude', color: '#94a3b8' } },
    },
    plugins: {
      ...CHART_DEFAULTS.plugins,
      title: {
        display: true,
        text: curveData
          ? `Peaks: ${curveData.peak_count} · Resp Rate: ${curveData.respiratory_rate_bpm} bpm`
          : '',
        color: '#94a3b8',
        font: { size: 12 },
        position: 'bottom',
      },
    },
  };

  return (
    <div className="card">
      <div className="card-header">Cough / Respiration Curve</div>
      <div className="subject-selector">
        Subject:
        <select value={selectedSubject} onChange={(e) => setSelectedSubject(e.target.value)}>
          {subjects.map((s) => <option key={s} value={s}>{s}</option>)}
        </select>
      </div>
      <div className="chart-container">
        {loading && <div className="loading">Loading...</div>}
        {error && <div className="error">{error}</div>}
        {!loading && !error && chartData && <Line data={chartData} options={options} />}
        {!loading && !error && !chartData && <div className="empty">No data available</div>}
      </div>
    </div>
  );
}
