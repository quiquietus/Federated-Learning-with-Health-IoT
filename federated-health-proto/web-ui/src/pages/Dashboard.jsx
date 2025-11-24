import { useState, useEffect } from 'react'
import { Routes, Route, Link, useLocation } from 'react-router-dom'
import axios from 'axios'

const API_URL = 'http://localhost:8000'

// --- Floating Animation Component ---
function FloatingSymbols() {
  const symbols = ['+', '⚕️', '🏥', '🔬', '💊', '🧬', '🩺', '❤️', '🧪', '🩸']
  const colors = ['#6366f1', '#8b5cf6', '#ec4899', '#10b981', '#f59e0b', '#3b82f6']
  const [elements, setElements] = useState([])

  useEffect(() => {
    // Generate random floating elements
    const newElements = Array.from({ length: 20 }).map((_, i) => ({
      id: i,
      symbol: symbols[Math.floor(Math.random() * symbols.length)],
      color: colors[Math.floor(Math.random() * colors.length)],
      left: `${Math.random() * 100}vw`,
      animationDuration: `${10 + Math.random() * 20}s`,
      animationDelay: `${Math.random() * 10}s`,
      fontSize: `${2 + Math.random() * 4}rem`, // Larger sizes (2rem to 6rem)
      opacity: 0.1 + Math.random() * 0.2
    }))
    setElements(newElements)
  }, [])

  return (
    <div className="floating-container">
      {elements.map(el => (
        <div
          key={el.id}
          className="floating-symbol"
          style={{
            left: el.left,
            color: el.color,
            animation: `floatUp ${el.animationDuration} linear infinite`,
            animationDelay: el.animationDelay,
            fontSize: el.fontSize,
            opacity: el.opacity
          }}
        >
          {el.symbol}
        </div>
      ))}
    </div>
  )
}

// --- Sidebar ---
function Sidebar({ user, onLogout }) {
  const location = useLocation()
  const isActive = (path) => location.pathname.includes(path)
  
  return (
    <div style={styles.sidebar}>
      <div style={styles.sidebarHeader}>
        <h2 style={styles.logo}>FedHealth<span style={{color: '#818cf8'}}>.AI</span></h2>
        <div style={styles.userBadge}>
          <div style={styles.avatar}>{user.user_id[0].toUpperCase()}</div>
          <div style={styles.userInfo}>
            <div style={styles.userName}>{user.user_id}</div>
            <div style={styles.userRole}>{user.client_type} • {user.organization}</div>
          </div>
        </div>
      </div>

      <nav style={styles.nav}>
        <Link to="/dashboard/overview" className="nav-link" style={isActive('overview') ? styles.navLinkActive : styles.navLink}>
          📊 Overview
        </Link>
        <Link to="/dashboard/upload" className="nav-link" style={isActive('upload') ? styles.navLinkActive : styles.navLink}>
          📤 Upload Data
        </Link>
        <Link to="/dashboard/train" className="nav-link" style={isActive('train') ? styles.navLinkActive : styles.navLink}>
          🔬 Local Training
        </Link>
        <Link to="/dashboard/models" className="nav-link" style={isActive('models') ? styles.navLinkActive : styles.navLink}>
          ⬇️ Global Models
        </Link>
      </nav>

      <button onClick={onLogout} style={styles.logoutButton}>
        Sign Out
      </button>
    </div>
  )
}

// --- Overview ---
function Overview({ user }) {
  const [rounds, setRounds] = useState([])
  const [metrics, setMetrics] = useState([])
  
  const [risk, setRisk] = useState(null)

  useEffect(() => {
    const fetchData = async () => {
      try {
        const token = localStorage.getItem('token')
        const [roundsRes, metricsRes, riskRes] = await Promise.all([
          axios.get(`${API_URL}/api/rounds/${user.client_type}`, { headers: { Authorization: `Bearer ${token}` } }),
          axios.get(`${API_URL}/api/metrics/${user.client_type}`, { headers: { Authorization: `Bearer ${token}` } }),
          axios.get(`${API_URL}/api/risk-score`, { headers: { Authorization: `Bearer ${token}` } })
        ])
        setRounds(roundsRes.data)
        setMetrics(metricsRes.data)
        setRisk(riskRes.data)
      } catch (err) { console.error(err) }
    }
    fetchData()
    const interval = setInterval(fetchData, 5000)
    return () => clearInterval(interval)
  }, [user.client_type])

  const activeRound = rounds.find(r => r.status === 'active')
  const completedRounds = rounds.filter(r => r.status === 'completed')
  const latestMetric = metrics.length > 0 ? metrics[metrics.length - 1] : null

  return (
    <div style={styles.page}>
      <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '30px'}}>
        <h1 style={{...styles.pageTitle, marginBottom: 0}}>Dashboard Overview</h1>
        <button className="btn-secondary" onClick={() => alert('Client download feature coming soon!')}>
          ⬇️ Download Python Client
        </button>
      </div>
      
      <div style={styles.statsGrid}>
        <div className="hover-card" style={styles.statCard}>
          <div style={styles.statIcon}>⏳</div>
          <div>
            <h3>Active Round</h3>
            <div style={styles.statValue}>
              {activeRound ? `#${activeRound.round_number}` : 'Waiting...'}
            </div>
            <div style={styles.statLabel}>
              {activeRound ? 'In Progress' : 'Next round in ~60s'}
            </div>
          </div>
        </div>
        
        <div className="hover-card" style={styles.statCard}>
          <div style={styles.statIcon}>🎯</div>
          <div>
            <h3>Global Accuracy</h3>
            <div style={styles.statValue}>
              {latestMetric ? `${(latestMetric.accuracy * 100).toFixed(1)}%` : 'N/A'}
            </div>
            <div style={styles.statLabel}>Latest Aggregated Model</div>
          </div>
        </div>

        <div className="hover-card" style={styles.statCard}>
          <div style={styles.statIcon}>🛡️</div>
          <div>
            <h3>Risk Score</h3>
            <div style={{...styles.statValue, color: risk?.level === 'High' ? '#dc2626' : risk?.level === 'Medium' ? '#f59e0b' : '#10b981'}}>
              {risk ? risk.risk_score : '-'}
            </div>
            <div style={styles.statLabel}>{risk ? risk.level : 'Calculating...'}</div>
          </div>
        </div>
      </div>

      <h2 style={styles.sectionTitle}>Recent Activity</h2>
      <div style={styles.tableCard}>
        <table style={styles.table}>
          <thead>
            <tr>
              <th style={styles.th}>Round</th>
              <th style={styles.th}>Status</th>
              <th style={styles.th}>Participants</th>
              <th style={styles.th}>Accuracy</th>
              <th style={styles.th}>F1 Score</th>
            </tr>
          </thead>
          <tbody>
            {rounds.slice().reverse().map(round => (
              <tr key={round.round_id} style={styles.tr}>
                <td style={styles.td}>#{round.round_number}</td>
                <td style={styles.td}>
                  <span style={round.status === 'active' ? styles.badgeActive : styles.badgeCompleted}>
                    {round.status}
                  </span>
                </td>
                <td style={styles.td}>{round.num_participants}</td>
                <td style={styles.td}>{round.avg_accuracy ? `${(round.avg_accuracy * 100).toFixed(1)}%` : '-'}</td>
                <td style={styles.td}>{round.avg_f1_score ? round.avg_f1_score.toFixed(3) : '-'}</td>
              </tr>
            ))}
            {rounds.length === 0 && (
              <tr><td colSpan="5" style={{...styles.td, textAlign: 'center'}}>No rounds yet</td></tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}

// --- Upload Data ---
function UploadData({ user }) {
  const [file, setFile] = useState(null)
  const [uploading, setUploading] = useState(false)
  const [status, setStatus] = useState(null)

  useEffect(() => {
    const checkStatus = async () => {
      try {
        const res = await axios.get(`${API_URL}/api/me`, {
          headers: { Authorization: `Bearer ${localStorage.getItem('token')}` }
        })
        setStatus(res.data.dataset_uploaded)
      } catch (err) { console.error(err) }
    }
    checkStatus()
  }, [])

  const handleUpload = async () => {
    if (!file) return alert('Please select a file')
    setUploading(true)
    const formData = new FormData()
    formData.append('file', file)

    try {
      await axios.post(`${API_URL}/api/upload-dataset`, formData, {
        headers: {
          Authorization: `Bearer ${localStorage.getItem('token')}`,
          'Content-Type': 'multipart/form-data'
        }
      })
      alert('✅ Dataset uploaded successfully!')
      setStatus(true)
    } catch (err) {
      alert('❌ Upload failed: ' + (err.response?.data?.detail || err.message))
    }
    setUploading(false)
  }

  return (
    <div style={styles.page}>
      <h1 style={styles.pageTitle}>Upload Dataset</h1>
      
      <div className="hover-card" style={styles.card}>
        <div style={styles.uploadHeader}>
          <div style={styles.uploadIcon}>📂</div>
          <h3>{user.client_type.toUpperCase()} Dataset</h3>
          <p>Upload your local CSV data for training. Data never leaves your device.</p>
        </div>

        <div style={styles.uploadStatus}>
          Current Status: 
          <span style={status ? styles.statusSuccess : styles.statusPending}>
            {status ? ' ✅ Ready for Training' : ' ⚠️ No Data Uploaded'}
          </span>
        </div>

        <div style={styles.dropzone}>
          <input 
            type="file" 
            accept=".csv"
            onChange={e => setFile(e.target.files[0])}
            style={styles.fileInput}
          />
          <p style={{marginTop: '10px', color: '#6b7280'}}>
            {file ? `Selected: ${file.name}` : 'Select CSV file...'}
          </p>
        </div>

        <button 
          onClick={handleUpload} 
          disabled={uploading}
          className="btn-primary"
          style={{width: '100%'}}
        >
          {uploading ? 'Uploading...' : 'Upload Dataset'}
        </button>
      </div>
    </div>
  )
}

// --- Local Training ---
function LocalTraining({ user }) {
  const [training, setTraining] = useState(false)
  const [progress, setProgress] = useState(0)
  const [logs, setLogs] = useState([])
  const [result, setResult] = useState(null)

  const addLog = (msg) => setLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] ${msg}`])

  const handleTrain = async () => {
    setTraining(true)
    setProgress(0)
    setLogs([])
    setResult(null)
    
    addLog('Initializing local training environment...')
    
    const steps = [
      { p: 10, msg: 'Loading local dataset...' },
      { p: 30, msg: 'Preprocessing data...' },
      { p: 50, msg: `Training ${user.client_type} model...` },
      { p: 80, msg: 'Evaluating model performance...' },
      { p: 90, msg: 'Encrypting and sending update...' }
    ]

    let stepIdx = 0
    const interval = setInterval(() => {
      if (stepIdx < steps.length) {
        setProgress(steps[stepIdx].p)
        addLog(steps[stepIdx].msg)
        stepIdx++
      }
    }, 800)

    try {
      const res = await axios.post(`${API_URL}/api/train`, {}, {
        headers: { Authorization: `Bearer ${localStorage.getItem('token')}` }
      })
      
      clearInterval(interval)
      setProgress(100)
      addLog('✅ Training completed successfully!')
      setResult(res.data)
    } catch (err) {
      clearInterval(interval)
      addLog('❌ Error: ' + (err.response?.data?.detail || err.message))
    }
    setTraining(false)
  }

  return (
    <div style={styles.page}>
      <h1 style={styles.pageTitle}>Local Training</h1>
      
      <div style={styles.grid2}>
        <div className="hover-card" style={styles.card}>
          <h3>Control Panel</h3>
          <p style={{marginBottom: '20px', color: '#6b7280'}}>
            Start local training to contribute to the current federated round.
          </p>
          
          <button 
            onClick={handleTrain}
            disabled={training}
            className={training ? "btn-secondary" : "btn-primary"}
            style={{width: '100%'}}
          >
            {training ? 'Training in Progress...' : '🚀 Start Training'}
          </button>

          {training && (
            <div style={styles.progressContainer}>
              <div style={styles.progressBar}>
                <div style={{...styles.progressFill, width: `${progress}%`}}></div>
              </div>
              <div style={styles.progressText}>{progress}% Complete</div>
            </div>
          )}
        </div>

        <div className="hover-card" style={styles.card}>
          <h3>Training Logs</h3>
          <div style={styles.logBox}>
            {logs.length === 0 && <span style={{color: '#9ca3af'}}>Ready to train...</span>}
            {logs.map((log, i) => <div key={i} style={styles.logLine}>{log}</div>)}
          </div>
        </div>
      </div>

      {result && (
        <div style={styles.resultCard}>
          <h3>🎉 Training Results (Round {result.round_id})</h3>
          <div style={styles.metricsGrid}>
            <div style={styles.metricItem}>
              <div style={styles.metricLabel}>Accuracy</div>
              <div style={styles.metricValue}>{(result.metrics.accuracy * 100).toFixed(2)}%</div>
            </div>
            <div style={styles.metricItem}>
              <div style={styles.metricLabel}>F1 Score</div>
              <div style={styles.metricValue}>{result.metrics.f1_score.toFixed(4)}</div>
            </div>
            <div style={styles.metricItem}>
              <div style={styles.metricLabel}>Precision</div>
              <div style={styles.metricValue}>{result.metrics.precision.toFixed(4)}</div>
            </div>
            <div style={styles.metricItem}>
              <div style={styles.metricLabel}>Recall</div>
              <div style={styles.metricValue}>{result.metrics.recall.toFixed(4)}</div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

// --- Global Models ---
function GlobalModels({ user }) {
  const [downloading, setDownloading] = useState(false)

  const handleDownload = async () => {
    setDownloading(true)
    try {
      const res = await axios.get(`${API_URL}/api/download-model/${user.client_type}`, {
        headers: { Authorization: `Bearer ${localStorage.getItem('token')}` },
        responseType: 'blob'
      })
      
      const url = window.URL.createObjectURL(new Blob([res.data]))
      const link = document.createElement('a')
      link.href = url
      link.setAttribute('download', `global_model_${user.client_type}.txt`)
      document.body.appendChild(link)
      link.click()
      link.remove()
      alert('✅ Model downloaded successfully!')
    } catch (err) {
      alert('❌ Download failed: ' + (err.response?.data?.detail || err.message))
    }
    setDownloading(false)
  }

  return (
    <div style={styles.page}>
      <h1 style={styles.pageTitle}>Global Models</h1>
      
      <div className="hover-card" style={styles.card}>
        <div style={styles.modelHeader}>
          <div style={styles.modelIcon}>🧠</div>
          <div>
            <h3>Latest Global Model ({user.client_type})</h3>
            <p>Aggregated from all participating {user.client_type}s.</p>
          </div>
        </div>

        <div style={styles.modelInfo}>
          <p><strong>Format:</strong> LightGBM / PyTorch State Dict</p>
          <p><strong>Version:</strong> Auto-updated every 80s</p>
          <p><strong>Status:</strong> Ready for inference</p>
        </div>

        <button 
          onClick={handleDownload}
          disabled={downloading}
          className="btn-secondary"
          style={{width: '100%'}}
        >
          {downloading ? 'Downloading...' : '⬇️ Download Model'}
        </button>
      </div>
    </div>
  )
}

// --- Main Dashboard ---
export default function Dashboard({ user, onLogout }) {
  return (
    <div style={styles.container}>
      <FloatingSymbols />
      <Sidebar user={user} onLogout={onLogout} />
      <div style={styles.main}>
        <Routes>
          <Route path="overview" element={<Overview user={user} />} />
          <Route path="upload" element={<UploadData user={user} />} />
          <Route path="train" element={<LocalTraining user={user} />} />
          <Route path="models" element={<GlobalModels user={user} />} />
          <Route path="/" element={<Overview user={user} />} />
        </Routes>
      </div>
    </div>
  )
}

// --- Styles ---
const styles = {
  container: { display: 'flex', minHeight: '100vh', fontFamily: "'Inter', sans-serif", position: 'relative' },
  sidebar: {
    width: '280px',
    background: '#1f2937',
    color: 'white',
    display: 'flex',
    flexDirection: 'column',
    padding: '24px',
    boxShadow: '4px 0 10px rgba(0,0,0,0.1)',
    zIndex: 10
  },
  sidebarHeader: { marginBottom: '40px' },
  logo: { fontSize: '24px', fontWeight: 'bold', marginBottom: '20px', letterSpacing: '-0.5px' },
  userBadge: {
    display: 'flex',
    alignItems: 'center',
    background: '#374151',
    padding: '12px',
    borderRadius: '8px',
    gap: '12px'
  },
  avatar: {
    width: '36px',
    height: '36px',
    background: '#667eea',
    borderRadius: '50%',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontWeight: 'bold'
  },
  userInfo: { overflow: 'hidden' },
  userName: { fontWeight: '600', fontSize: '14px' },
  userRole: { fontSize: '12px', color: '#9ca3af', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' },
  nav: { display: 'flex', flexDirection: 'column', gap: '8px', flex: 1 },
  navLink: {
    padding: '12px 16px',
    color: '#d1d5db',
    textDecoration: 'none',
    borderRadius: '8px',
    transition: 'all 0.2s',
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
    fontSize: '15px'
  },
  navLinkActive: {
    padding: '12px 16px',
    background: '#667eea',
    color: 'white',
    textDecoration: 'none',
    borderRadius: '8px',
    fontWeight: '500',
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
    fontSize: '15px',
    boxShadow: '0 4px 6px rgba(0,0,0,0.1)'
  },
  logoutButton: {
    padding: '12px',
    background: 'transparent',
    border: '1px solid #4b5563',
    color: '#9ca3af',
    borderRadius: '8px',
    cursor: 'pointer',
    transition: 'all 0.2s',
    marginTop: 'auto'
  },
  main: { flex: 1, padding: '40px', overflowY: 'auto', zIndex: 1 },
  page: { maxWidth: '1000px', margin: '0 auto' },
  pageTitle: { fontSize: '28px', fontWeight: 'bold', color: '#111827', marginBottom: '30px' },
  statsGrid: { display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '24px', marginBottom: '40px' },
  statCard: { background: 'white', padding: '24px', borderRadius: '12px', boxShadow: '0 1px 3px rgba(0,0,0,0.1)', display: 'flex', gap: '20px', alignItems: 'center' },
  statIcon: { fontSize: '32px', background: '#f3f4f6', padding: '12px', borderRadius: '12px' },
  statValue: { fontSize: '28px', fontWeight: 'bold', color: '#111827', margin: '5px 0' },
  statLabel: { color: '#6b7280', fontSize: '14px' },
  sectionTitle: { fontSize: '20px', fontWeight: '600', color: '#374151', marginBottom: '20px' },
  tableCard: { background: 'white', borderRadius: '12px', boxShadow: '0 1px 3px rgba(0,0,0,0.1)', overflow: 'hidden' },
  table: { width: '100%', borderCollapse: 'collapse' },
  th: { textAlign: 'left', padding: '16px 24px', background: '#f9fafb', color: '#6b7280', fontSize: '12px', textTransform: 'uppercase', letterSpacing: '0.05em' },
  td: { padding: '16px 24px', borderTop: '1px solid #e5e7eb', color: '#374151' },
  badgeActive: { background: '#d1fae5', color: '#065f46', padding: '4px 10px', borderRadius: '20px', fontSize: '12px', fontWeight: '500' },
  badgeCompleted: { background: '#e5e7eb', color: '#374151', padding: '4px 10px', borderRadius: '20px', fontSize: '12px', fontWeight: '500' },
  card: { background: 'white', padding: '30px', borderRadius: '12px', boxShadow: '0 1px 3px rgba(0,0,0,0.1)' },
  uploadHeader: { textAlign: 'center', marginBottom: '30px' },
  uploadIcon: { fontSize: '48px', marginBottom: '16px' },
  uploadStatus: { textAlign: 'center', padding: '12px', background: '#f9fafb', borderRadius: '8px', marginBottom: '30px' },
  statusSuccess: { color: '#10b981', fontWeight: '600' },
  statusPending: { color: '#f59e0b', fontWeight: '600' },
  dropzone: { border: '2px dashed #e5e7eb', borderRadius: '12px', padding: '40px', textAlign: 'center', marginBottom: '30px', cursor: 'pointer', transition: 'border-color 0.2s' },
  grid2: { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px' },
  logBox: { background: '#1f2937', color: '#10b981', padding: '16px', borderRadius: '8px', height: '300px', overflowY: 'auto', fontFamily: 'monospace', fontSize: '13px' },
  logLine: { marginBottom: '8px' },
  progressContainer: { marginTop: '20px' },
  progressBar: { height: '8px', background: '#e5e7eb', borderRadius: '4px', overflow: 'hidden' },
  progressFill: { height: '100%', background: '#667eea', transition: 'width 0.3s ease' },
  progressText: { textAlign: 'right', fontSize: '12px', color: '#6b7280', marginTop: '6px' },
  resultCard: { marginTop: '24px', background: '#ecfdf5', padding: '24px', borderRadius: '12px', border: '1px solid #a7f3d0' },
  metricsGrid: { display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '16px', marginTop: '16px' },
  metricItem: { background: 'white', padding: '16px', borderRadius: '8px', textAlign: 'center', boxShadow: '0 1px 2px rgba(0,0,0,0.05)' },
  metricLabel: { fontSize: '12px', color: '#6b7280', marginBottom: '4px' },
  metricValue: { fontSize: '20px', fontWeight: 'bold', color: '#059669' },
  modelHeader: { display: 'flex', gap: '20px', alignItems: 'center', marginBottom: '24px' },
  modelIcon: { fontSize: '40px', background: '#eff6ff', padding: '16px', borderRadius: '12px' },
  modelInfo: { background: '#f9fafb', padding: '20px', borderRadius: '8px', marginBottom: '24px', lineHeight: '1.6' }
}
