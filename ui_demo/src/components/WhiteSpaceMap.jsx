import { useState } from 'react';
import { ScatterChart, Scatter, XAxis, YAxis, ZAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, ReferenceArea } from 'recharts';
import { MousePointer2, SlidersHorizontal, Info } from 'lucide-react';

// Mock Data for the white space map representing datasets in semantic space
const dataPoints = [
  // Red Ocean (Saturated) - e.g., QA, Sentiment
  { year: 2021, name: "SentimentX", domain: "Sentiment Analysis", x: -40, y: 60, z: 200, color: "var(--status-danger)" },
  { year: 2022, name: "SentiMulti", domain: "Sentiment Analysis", x: -35, y: 55, z: 120, color: "var(--status-danger)" },
  { year: 2019, name: "QA-Gen", domain: "Question Answering", x: -20, y: 30, z: 400, color: "var(--status-warning)" },
  { year: 2020, name: "WikiQA", domain: "Question Answering", x: -25, y: 35, z: 250, color: "var(--status-warning)" },
  { year: 2023, name: "DomainQA", domain: "Question Answering", x: -15, y: 25, z: 100, color: "var(--status-warning)" },

  // Emerging / Blue Ocean (White Space) - e.g., Agentic Planning, Low-Resource Reasoning
  { year: 2024, name: "AgentNav-1", domain: "Agentic Planning", x: 60, y: -40, z: 80, color: "var(--accent-primary)" },
  { year: 2025, name: "Swahili-Reason", domain: "Low-Resource Reasoning", x: 70, y: 20, z: 60, color: "var(--status-success)" },
  { year: 2023, name: "MathSolve- Африка", domain: "Multilingual Math", x: 45, y: 15, z: 150, color: "var(--accent-secondary)" },
  { year: 2024, name: "CodeSwitch-Dialect", domain: "Dialect Modeling", x: 80, y: 50, z: 90, color: "var(--accent-secondary)" }
];

export default function WhiteSpaceMap() {
  const [yearFilter, setYearFilter] = useState(2025);

  const filteredData = dataPoints.filter(d => d.year <= yearFilter);

  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div style={{
          background: 'var(--glass-bg)',
          backdropFilter: 'blur(12px)',
          border: '1px solid var(--glass-border)',
          borderRadius: '8px',
          padding: '12px 16px',
          boxShadow: '0 8px 32px rgba(0,0,0,0.4)'
        }}>
          <p style={{ fontWeight: 600, color: data.color, marginBottom: '4px' }}>{data.name} <span style={{fontSize: '0.8rem', color: 'var(--text-secondary)'}}>({data.year})</span></p>
          <p style={{ fontSize: '0.9rem', color: 'var(--text-primary)' }}>{data.domain}</p>
          <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '8px' }}>Adoption Proxy: {data.z}</p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="fade-in">
      <div className="page-header flex-between">
        <div>
          <h2 className="page-title">Research White Space Map</h2>
          <p className="page-description">Identifying semantic gaps and saturation in NLP datasets via PCA projections.</p>
        </div>
        <div className="flex-center" style={{ gap: '16px' }}>
          <span style={{ fontSize: '0.9rem', color: 'var(--text-secondary)' }}>Timeline: {yearFilter}</span>
          <input 
            type="range" 
            min="2019" 
            max="2025" 
            value={yearFilter} 
            onChange={(e) => setYearFilter(parseInt(e.target.value))}
            style={{ width: '150px', accentColor: 'var(--accent-primary)' }}
          />
        </div>
      </div>

      <div className="glass-panel card" style={{ height: '600px', display: 'flex', flexDirection: 'column' }}>
        <div className="flex-between" style={{ marginBottom: '16px' }}>
          <div style={{ display: 'flex', gap: '24px' }}>
            <div className="flex-center" style={{ gap: '8px', fontSize: '0.85rem' }}>
              <div style={{ width: '12px', height: '12px', borderRadius: '50%', background: 'var(--status-danger)' }}></div>
              Saturated (Red Ocean)
            </div>
            <div className="flex-center" style={{ gap: '8px', fontSize: '0.85rem' }}>
              <div style={{ width: '12px', height: '12px', borderRadius: '50%', background: 'var(--accent-primary)' }}></div>
              Emerging (White Space)
            </div>
          </div>
          <button style={{ fontSize: '0.8rem', padding: '6px 12px' }}>
            <SlidersHorizontal size={14} /> Adjust PCA Weights
          </button>
        </div>
        
        <div style={{ flex: 1, position: 'relative' }}>
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis type="number" dataKey="x" name="Semantic Dim 1" stroke="rgba(255,255,255,0.2)" tick={{fill: 'var(--text-muted)'}} />
              <YAxis type="number" dataKey="y" name="Semantic Dim 2" stroke="rgba(255,255,255,0.2)" tick={{fill: 'var(--text-muted)'}} />
              <ZAxis type="number" dataKey="z" range={[50, 400]} />
              <Tooltip content={<CustomTooltip />} cursor={{strokeDasharray: '3 3'}} />
              
              {/* Highlight Clusters */}
              <ReferenceArea x1={-50} x2={0} y1={20} y2={70} fill="rgba(239, 68, 68, 0.05)" strokeOpacity={0} />
              <ReferenceArea x1={40} x2={90} y1={-50} y2={60} fill="rgba(99, 102, 241, 0.05)" strokeOpacity={0} />

              <Scatter name="Datasets" data={filteredData} animationDuration={800}>
                {filteredData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} fillOpacity={0.8} />
                ))}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
          
          {/* Overlay Annotations */}
          <div style={{ position: 'absolute', top: '10%', left: '15%', color: 'var(--status-danger)', fontSize: '0.85rem', fontWeight: 600, letterSpacing: '0.05em', opacity: 0.7 }}>
            HIGH DENSITY (QA, SENTIMENT)
          </div>
          <div style={{ position: 'absolute', bottom: '20%', right: '15%', color: 'var(--accent-primary)', fontSize: '0.85rem', fontWeight: 600, letterSpacing: '0.05em', opacity: 0.7 }}>
            RESEARCH OPPORTUNITY
          </div>
        </div>
        
        <div style={{ marginTop: '16px', padding: '12px 16px', background: 'var(--bg-secondary)', borderRadius: 'var(--border-radius-sm)', border: '1px solid var(--glass-border)', display: 'flex', alignItems: 'center', gap: '12px' }}>
          <Info size={18} color="var(--accent-primary)" />
          <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
            <strong>Analysis:</strong> The field's centroid is actively moving from left to right. Datasets introducing Agentic Planning or Low-Resource Reasoning currently explore highly novel semantic space with low neighbor density.
          </p>
        </div>
      </div>
    </div>
  );
}
