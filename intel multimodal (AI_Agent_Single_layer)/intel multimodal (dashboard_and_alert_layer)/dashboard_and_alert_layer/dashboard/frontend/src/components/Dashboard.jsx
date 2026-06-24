import React from 'react';
import HealthStateCard from './HealthStateCard';
import CoughCurveChart from './CoughCurveChart';
import PhysioTrendChart from './PhysioTrendChart';
import DiseaseClassification from './DiseaseClassification';
import FeatureVizPanel from './FeatureVizPanel';
import AlertStatusPanel from './AlertStatusPanel';
import AgentSuggestionsPanel from './AgentSuggestionsPanel';

export default function Dashboard({
  healthData,
  diseaseData,
  lastHealthUpdate,
  alertLog,
  lastAgentAdvice,
  agentAdviceData,
  agentAdviceLog,
  agentLoading,
  agentError,
  socket,
  onTestAlert,
}) {
  return (
    <div className="dashboard-grid">
      {/* Row 1: Health State (full width) */}
      <div className="full-width">
        <HealthStateCard
          healthData={healthData}
          lastUpdate={lastHealthUpdate}
        />
      </div>

      {/* Row 2: Cough Curve | Physio Trends */}
      <CoughCurveChart />
      <PhysioTrendChart />

      {/* Row 3: Disease Classification | Feature Viz */}
      <DiseaseClassification diseaseData={diseaseData} />
      <FeatureVizPanel />

      {/* Row 3.5: AI Agent Suggestions (full width) */}
      <div className="full-width">
        <AgentSuggestionsPanel
          lastAgentAdvice={lastAgentAdvice}
          agentAdviceData={agentAdviceData}
          adviceHistory={agentAdviceLog}
          loading={agentLoading}
          error={agentError}
        />
      </div>

      {/* Row 4: Alert Status (full width) */}
      <div className="full-width">
        <AlertStatusPanel
          alertLog={alertLog}
          socket={socket}
          onTestAlert={onTestAlert}
        />
      </div>
    </div>
  );
}
