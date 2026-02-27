import { useState } from 'react';
import { Play, CheckCircle2, AlertTriangle, XCircle, Search, ArrowRight } from 'lucide-react';

const MOCK_ACUS = [
  {
    id: 1,
    text: "We introduce a new dataset for code-switching in Swahili called SwahiliCS.",
    status: "novel", // green
    matches: []
  },
  {
    id: 2,
    text: "The dataset contains 50,000 parallel sentences aligning Swahili and English.",
    status: "ambiguous", // yellow
    matches: [
      {
        paper: "Kamau et al. (2022)",
        title: "Large-Scale Parallel Corpora for African Languages",
        claim: "We release 40k parallel Swahili-English sentence pairs extracted from news.",
        similarity: "82%"
      }
    ]
  },
  {
    id: 3,
    text: "We benchmark standard Transformer models and achieve state-of-the-art results.",
    status: "entailed", // red
    matches: [
      {
        paper: "Wang et al. (2023)",
        title: "Standardizing Baselines for Low-Resource Translation",
        claim: "We benchmark standard Transformers on 10 African languages with SOTA performance.",
        similarity: "95%"
      },
      {
        paper: "Ochieng (2021)",
        title: "Baseline Models for Swahili NLP",
        claim: "We establish strong Transformer baselines achieving SOTA on Swahili translation.",
        similarity: "88%"
      }
    ]
  }
];

export default function NoveltyAudit() {
  const [abstract, setAbstract] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState(null);
  const [selectedAcu, setSelectedAcu] = useState(null);

  const handleAnalyze = () => {
    if (!abstract.trim()) return;
    setIsAnalyzing(true);
    setResults(null);
    setSelectedAcu(null);
    
    // Simulate network request
    setTimeout(() => {
      setIsAnalyzing(false);
      setResults(MOCK_ACUS);
      setSelectedAcu(MOCK_ACUS[0]);
    }, 1500);
  };

  const loadExample = () => {
    setAbstract("We introduce a new dataset for code-switching in Swahili called SwahiliCS. The dataset contains 50,000 parallel sentences aligning Swahili and English. We benchmark standard Transformer models and achieve state-of-the-art results.");
  };

  return (
    <div className="fade-in">
      <div className="page-header">
        <h2 className="page-title">Novelty Audit & Prior Art Finder</h2>
        <p className="page-description">Extract Atomic Content Units (ACUs) and check against prior work via structural entailment.</p>
      </div>

      <div className="grid-layout" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '32px' }}>
        
        {/* Left Column - Input and ACUs */}
        <div className="column">
          <div className="glass-panel card" style={{ marginBottom: '24px' }}>
            <h3 className="card-title">1. Input Abstract or Claims</h3>
            <textarea 
              rows={5} 
              placeholder="Paste your paper's abstract or core contributions here..."
              value={abstract}
              onChange={(e) => setAbstract(e.target.value)}
              style={{ marginBottom: '16px', resize: 'vertical' }}
            />
            <div className="flex-between">
              <button onClick={loadExample} style={{ fontSize: '0.85rem' }}>Load Example</button>
              <button className="primary" onClick={handleAnalyze} disabled={isAnalyzing || !abstract.trim()}>
                {isAnalyzing ? (
                  <>Analysing <div className="spinner" style={{width: '14px', height: '14px', border: '2px solid rgba(255,255,255,0.3)', borderTop: '2px solid white', borderRadius: '50%', animation: 'spin 1s linear infinite'}}/></>
                ) : (
                  <><Play size={16} /> Run Novelty Audit</>
                )}
              </button>
            </div>
          </div>

          {results && (
            <div className="glass-panel card fade-in" style={{ animationDelay: '0.1s' }}>
              <div className="flex-between" style={{ marginBottom: '16px' }}>
                <h3 className="card-title" style={{ margin: 0 }}>2. Extracted ACUs</h3>
                <span className="badge" style={{ background: 'rgba(99, 102, 241, 0.1)', color: 'var(--accent-primary)' }}>{results.length} units found</span>
              </div>
              
              <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                {results.map((acu) => (
                  <div 
                    key={acu.id}
                    className={`nav-item ${selectedAcu?.id === acu.id ? 'active' : ''}`}
                    style={{ 
                      alignItems: 'flex-start', 
                      padding: '16px', 
                      background: selectedAcu?.id === acu.id ? 'var(--bg-secondary)' : 'var(--bg-primary)',
                      border: selectedAcu?.id === acu.id ? '1px solid var(--accent-primary)' : '1px solid var(--glass-border)',
                      cursor: 'pointer'
                    }}
                    onClick={() => setSelectedAcu(acu)}
                  >
                    <div style={{ marginTop: '2px', marginRight: '8px' }}>
                      {acu.status === 'novel' && <CheckCircle2 size={18} color="var(--status-success)" />}
                      {acu.status === 'ambiguous' && <AlertTriangle size={18} color="var(--status-warning)" />}
                      {acu.status === 'entailed' && <XCircle size={18} color="var(--status-danger)" />}
                    </div>
                    <div>
                      <p style={{ fontSize: '0.95rem', marginBottom: '8px' }}>{acu.text}</p>
                      <span className={`badge ${
                        acu.status === 'novel' ? 'success' : 
                        acu.status === 'ambiguous' ? 'warning' : 'danger'
                      }`}>
                        {acu.status === 'novel' ? 'Highly Novel' : 
                         acu.status === 'ambiguous' ? 'Ambiguous' : 'Entailed (Not Novel)'}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Right Column - Prior Art Matching */}
        {results && (
          <div className="column fade-in" style={{ animationDelay: '0.2s' }}>
            <div className="glass-panel card" style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
              <h3 className="card-title" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Search size={20} className="gradient-text"/> 
                3. Prior Art Entailment Check
              </h3>
              
              {selectedAcu ? (
                <div style={{ flex: 1 }}>
                  <div style={{ 
                    padding: '16px', 
                    background: 'rgba(255,255,255,0.03)', 
                    borderRadius: 'var(--border-radius-sm)',
                    marginBottom: '24px',
                    borderLeft: '4px solid var(--accent-primary)'
                  }}>
                    <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '4px', textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 600 }}>Analyzing Claim:</p>
                    <p style={{ fontSize: '1.05rem', fontStyle: 'italic' }}>"{selectedAcu.text}"</p>
                  </div>

                  <h4 style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                    Matched Neighbors
                    <span style={{ flex: 1, height: '1px', background: 'var(--glass-border)' }}></span>
                  </h4>

                  {selectedAcu.matches.length === 0 ? (
                    <div className="flex-center" style={{ flexDirection: 'column', padding: '40px 0', textAlign: 'center' }}>
                      <CheckCircle2 size={48} color="var(--status-success)" style={{ marginBottom: '16px', opacity: 0.8 }} />
                      <h4 style={{ fontSize: '1.2rem', color: 'var(--status-success)', marginBottom: '8px' }}>No matching claims found.</h4>
                      <p style={{ color: 'var(--text-muted)', fontSize: '0.95rem', maxWidth: '80%' }}>This specific contribution appears to be highly novel based on our historical corpus.</p>
                    </div>
                  ) : (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                      {selectedAcu.matches.map((match, idx) => (
                        <div key={idx} style={{ 
                          padding: '16px', 
                          border: '1px solid var(--glass-border)', 
                          borderRadius: 'var(--border-radius-sm)',
                          background: 'var(--bg-primary)',
                          position: 'relative',
                          overflow: 'hidden'
                        }}>
                          <div style={{ 
                            position: 'absolute', 
                            top: 0, 
                            left: 0, 
                            width: '4px', 
                            height: '100%', 
                            background: selectedAcu.status === 'entailed' ? 'var(--status-danger)' : 'var(--status-warning)'
                          }}></div>
                          
                          <div className="flex-between" style={{ marginBottom: '12px' }}>
                            <span style={{ fontWeight: 600, color: 'var(--accent-primary)', fontSize: '0.95rem' }}>{match.paper}</span>
                            <span className="badge" style={{ background: 'var(--bg-tertiary)', color: 'var(--text-secondary)' }}>Similarity: {match.similarity}</span>
                          </div>
                          
                          <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '12px', paddingBottom: '12px', borderBottom: '1px solid var(--glass-border)' }}>
                            {match.title}
                          </p>
                          
                          <div style={{ display: 'flex', alignItems: 'flex-start', gap: '12px' }}>
                            <ArrowRight size={16} color="var(--text-muted)" style={{ marginTop: '2px' }} />
                            <p style={{ fontSize: '0.95rem', lineHeight: 1.4 }}>"{match.claim}"</p>
                          </div>
                        </div>
                      ))}
                      
                      <div className="flex-center" style={{ marginTop: '16px' }}>
                        <span className={`badge ${selectedAcu.status === 'entailed' ? 'danger' : 'warning'}`} style={{ padding: '8px 16px', fontSize: '0.85rem' }}>
                          {selectedAcu.status === 'entailed' 
                            ? "Conclusion: Target claim is ENTAILED by prior work. (Not Novel)"
                            : "Conclusion: Potential semantic overlap. Manual verification suggested."}
                        </span>
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <div className="flex-center" style={{ flex: 1, color: 'var(--text-muted)', padding: '40px' }}>
                  Select an extracted ACU on the left to view prior art matches.
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      <style>{`
        @keyframes spin { 100% { transform: rotate(360deg); } }
        .fade-in { animation: fadeIn 0.4s ease forwards; opacity: 0; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
      `}</style>
    </div>
  );
}
