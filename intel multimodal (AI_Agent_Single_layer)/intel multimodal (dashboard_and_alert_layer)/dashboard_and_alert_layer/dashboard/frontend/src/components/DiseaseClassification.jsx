import React from 'react';
import { Pie } from 'react-chartjs-2';
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from 'chart.js';
import { CHART_DEFAULTS, LABEL_COLORS, LABEL_NAMES } from '../utils/chartConfig';
ChartJS.register(ArcElement, Tooltip, Legend);

export default function DiseaseClassification({ diseaseData }) {
  if (!diseaseData) {
    return (
      <div className="card">
        <div className="card-header">Disease Classification</div>
        <div className="loading">Loading classification data...</div>
      </div>
    );
  }

  const { metrics, per_class, class_names, confusion_matrix, experiment_label } = diseaseData;
  const accuracy = metrics?.accuracy ?? 0;

  // Pie chart: class distribution
  const pieData = {
    labels: class_names,
    datasets: [{
      data: per_class.map((c) => c.support),
      backgroundColor: [LABEL_COLORS[0], LABEL_COLORS[1], LABEL_COLORS[2]],
      borderColor: '#1e293b',
      borderWidth: 2,
    }],
  };

  // Confusion matrix cell color intensity
  const maxCM = Math.max(...(confusion_matrix?.flat() || [1]));

  return (
    <div className="card">
      <div className="card-header">Disease Classification Results</div>

      {/* Experiment label */}
      <div style={{ fontSize: 13, color: '#94a3b8', marginBottom: 16 }}>
        {experiment_label} · Accuracy: {accuracy}%
      </div>

      {/* Key metrics */}
      <div className="metrics-grid">
        {[
          ['Accuracy', `${accuracy}%`],
          ['Macro F1', `${metrics?.f1_macro ?? 0}%`],
          ['Weighted F1', `${metrics?.f1_weighted ?? 0}%`],
          ['Test Loss', metrics?.test_loss?.toFixed(3) ?? '-'],
        ].map(([label, value]) => (
          <div className="metric-item" key={label}>
            <div className="metric-value">{value}</div>
            <div className="metric-label">{label}</div>
          </div>
        ))}
      </div>

      {/* Per-class table */}
      <table className="per-class-table">
        <thead>
          <tr>
            <th>Class</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>F1 Score</th>
            <th>Support</th>
          </tr>
        </thead>
        <tbody>
          {per_class.map((cls) => (
            <tr key={cls.class}>
              <td>
                <span style={{
                  display: 'inline-block', width: 10, height: 10, borderRadius: '50%',
                  background: LABEL_COLORS[class_names.indexOf(cls.class)],
                  marginRight: 8,
                }} />
                {cls.class}
              </td>
              <td>{(cls.precision * 100).toFixed(1)}%</td>
              <td>{(cls.recall * 100).toFixed(1)}%</td>
              <td>{(cls.f1 * 100).toFixed(1)}%</td>
              <td>{cls.support}</td>
            </tr>
          ))}
        </tbody>
      </table>

      {/* Pie + Confusion Matrix side by side */}
      <div style={{ display: 'flex', gap: 24, flexWrap: 'wrap', alignItems: 'center' }}>
        <div style={{ width: 180, height: 180 }}>
          <Pie data={pieData} options={{ ...CHART_DEFAULTS, plugins: { legend: { display: false } } }} />
        </div>
        <div>
          <div style={{ fontSize: 12, color: '#94a3b8', marginBottom: 8 }}>Confusion Matrix (Actual → Predicted)</div>
          <div className="confusion-grid">
            <div></div>
            {class_names.map((n, i) => (
              <div key={i} style={{ textAlign: 'center', fontSize: 11, color: '#94a3b8' }}>Pred {n[0]}</div>
            ))}
            {confusion_matrix?.map((row, i) => (
              <React.Fragment key={i}>
                <div style={{ fontSize: 11, color: '#94a3b8', display: 'flex', alignItems: 'center' }}>Act {class_names[i]?.[0]}</div>
                {row.map((val, j) => (
                  <div
                    key={j}
                    className="confusion-cell"
                    style={{
                      background: i === j
                        ? LABEL_COLORS[i] + '40'
                        : `rgba(239,68,68,${(val / maxCM) * 0.3})`,
                      color: val > 0 ? '#e2e8f0' : '#475569',
                    }}
                  >
                    {val}
                  </div>
                ))}
              </React.Fragment>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
