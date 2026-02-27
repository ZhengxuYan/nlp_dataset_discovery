import { useState } from 'react';
import { Layers, Activity, Search, Compass, ShieldAlert } from 'lucide-react';
import './index.css';

// We will implement these components in separate files next
import NoveltyAudit from './components/NoveltyAudit';
import NutritionLabel from './components/NutritionLabel';
import WhiteSpaceMap from './components/WhiteSpaceMap';

function App() {
  const [activeTab, setActiveTab] = useState('audit');

  return (
    <div className="app-container">
      {/* Sidebar Navigation */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <h1 className="flex-center" style={{ justifyContent: 'flex-start', gap: '10px' }}>
            <Compass className="gradient-text" size={28} />
            Dataset Discovery
          </h1>
        </div>
        
        <nav className="sidebar-nav">
          <div 
            className={`nav-item ${activeTab === 'audit' ? 'active' : ''}`}
            onClick={() => setActiveTab('audit')}
          >
            <Search size={20} />
            Novelty Audit
          </div>
          <div 
            className={`nav-item ${activeTab === 'nutrition' ? 'active' : ''}`}
            onClick={() => setActiveTab('nutrition')}
          >
            <ShieldAlert size={20} />
            Nutrition Label
          </div>
          <div 
            className={`nav-item ${activeTab === 'map' ? 'active' : ''}`}
            onClick={() => setActiveTab('map')}
          >
            <Layers size={20} />
            White Space Map
          </div>
        </nav>
      </aside>

      {/* Main Content Area */}
      <main className="main-content">
        <div className="bg-glow"></div>
        <div className="content-wrapper">
          {activeTab === 'audit' && <NoveltyAudit />}
          {activeTab === 'nutrition' && <NutritionLabel />}
          {activeTab === 'map' && <WhiteSpaceMap />}
        </div>
      </main>
    </div>
  );
}

export default App;
