import React from 'react';

export default function ExperimentSelector({ experiments, activeId, onChange }) {
  return (
    <select
      className="exp-select"
      value={activeId}
      onChange={(e) => onChange(Number(e.target.value))}
    >
      {experiments.map((exp) => (
        <option key={exp.id} value={exp.id}>
          Exp {exp.id}: {exp.label} — Acc {exp.accuracy}%{exp.id === 2 ? ' ★ Best' : ''}
        </option>
      ))}
    </select>
  );
}
