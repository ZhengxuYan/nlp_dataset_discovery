import { Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer, Tooltip } from 'recharts';
import { PackageOpen, Scale, Globe2, Link, FileText, AlertTriangle } from 'lucide-react';

const scvData = [
  { subject: 'Novelty', A: 85, fullMark: 100 },
  { subject: 'Diversity', A: 45, fullMark: 100 },
  { subject: 'Quality', A: 30, fullMark: 100 },
  { subject: 'Adoption', A: 95, fullMark: 100 },
  { subject: 'Size', A: 70, fullMark: 100 },
];

export default function NutritionLabel() {
  return (
    <div className="fade-in">
      <div className="page-header flex-between" style={{ alignItems: 'flex-start' }}>
        <div>
          <h2 className="page-title">Dataset Nutrition Label</h2>
          <p className="page-description">Visualizing the Intrinsic Quality and Scientific Contribution Vector (SCV)</p>
        </div>
        <div style={{ textAlign: 'right' }}>
          <select style={{ 
            background: 'var(--bg-tertiary)', 
            border: '1px solid var(--glass-border)', 
            color: 'var(--text-primary)',
            padding: '10px 16px',
            borderRadius: 'var(--border-radius-sm)',
            outline: 'none',
            fontSize: '0.95rem'
          }}>
            <option>Select Dataset: GLUE Benchmark</option>
            <option>Select Dataset: SuperGLUE</option>
            <option>Select Dataset: Wikipedia (En)</option>
          </select>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: '32px' }}>
        
        {/* Left Column - SCV Radar Chart */}
        <div className="glass-panel card flex-center" style={{ flexDirection: 'column', height: '500px' }}>
          <h3 className="card-title" style={{ width: '100%', marginBottom: '0' }}>Scientific Contribution Fingerprint</h3>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', width: '100%', marginBottom: '24px' }}>Normalized metrics across 5 key dimensions</p>
          
          <div style={{ width: '100%', height: '100%', minHeight: '350px' }}>
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart cx="50%" cy="50%" outerRadius="70%" data={scvData}>
                <PolarGrid stroke="rgba(255,255,255,0.1)" />
                <PolarAngleAxis dataKey="subject" tick={{ fill: 'var(--text-secondary)', fontSize: 13, fontWeight: 500 }} />
                <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fill: 'rgba(255,255,255,0.3)', fontSize: 10 }} />
                <Tooltip 
                  contentStyle={{ 
                    background: 'var(--bg-secondary)', 
                    border: '1px solid var(--glass-border)', 
                    borderRadius: '8px',
                    boxShadow: '0 8px 32px rgba(0,0,0,0.4)'
                  }} 
                  itemStyle={{ color: 'var(--accent-primary)' }}
                />
                <Radar name="GLUE Benchmark" dataKey="A" stroke="var(--accent-primary)" fill="var(--accent-primary)" fillOpacity={0.4} />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Right Column - Warnings & Ingredients */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
          
          <div className="glass-panel card">
            <h3 className="card-title text-warning" style={{ color: 'var(--status-warning)' }}>
              <AlertTriangle size={20} /> Warning Labels
            </h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              <div style={{ padding: '12px 16px', background: 'var(--status-danger-bg)', borderRadius: 'var(--border-radius-sm)', border: '1px solid rgba(239, 68, 68, 0.2)', display: 'flex', alignItems: 'center', gap: '12px' }}>
                <Scale color="var(--status-danger)" size={20} />
                <div>
                  <h4 style={{ color: 'var(--status-danger)', fontSize: '0.9rem', marginBottom: '4px' }}>Unclear License</h4>
                  <p style={{ color: 'rgba(255,255,255,0.7)', fontSize: '0.8rem' }}>Cannot verify commercial usage rights.</p>
                </div>
              </div>
              
              <div style={{ padding: '12px 16px', background: 'var(--status-warning-bg)', borderRadius: 'var(--border-radius-sm)', border: '1px solid rgba(245, 158, 11, 0.2)', display: 'flex', alignItems: 'center', gap: '12px' }}>
                <Globe2 color="var(--status-warning)" size={20} />
                <div>
                  <h4 style={{ color: 'var(--status-warning)', fontSize: '0.9rem', marginBottom: '4px' }}>Low Language Diversity</h4>
                  <p style={{ color: 'rgba(255,255,255,0.7)', fontSize: '0.8rem' }}>Exclusively contains English language text.</p>
                </div>
              </div>
            </div>
          </div>

          <div className="glass-panel card" style={{ flex: 1 }}>
            <h3 className="card-title">
              <PackageOpen size={20} className="gradient-text"/> Detailed Ingredients
            </h3>
            
            <div style={{ display: 'flex', flexDirection: 'column', gap: '16px', marginTop: '20px' }}>
              <div className="flex-between">
                <span style={{ color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.9rem' }}>
                  <FileText size={16} /> Modality
                </span>
                <span style={{ fontWeight: 500 }}>Text-Only</span>
              </div>
              <div style={{ height: '1px', background: 'var(--glass-border)' }}></div>
              
              <div className="flex-between">
                <span style={{ color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.9rem' }}>
                  <Globe2 size={16} /> Supported Languages
                </span>
                <span style={{ fontWeight: 500 }}>1 (English)</span>
              </div>
              <div style={{ height: '1px', background: 'var(--glass-border)' }}></div>

              <div className="flex-between">
                <span style={{ color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.9rem' }}>
                  <Link size={16} /> Link Accessibility
                </span>
                <span className="badge success">Active URL</span>
              </div>
              <div style={{ height: '1px', background: 'var(--glass-border)' }}></div>

              <div className="flex-between">
                <span style={{ color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.9rem' }}>
                  <PackageOpen size={16} /> Artifact Size
                </span>
                <span style={{ fontWeight: 500 }}>~850 MB</span>
              </div>
            </div>
          </div>

        </div>
      </div>
    </div>
  );
}
