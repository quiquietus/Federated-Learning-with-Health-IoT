import { useState } from 'react'
import { Link } from 'react-router-dom'
import axios from 'axios'

const API_URL = 'http://localhost:8000'

export default function Login({ onLogin }) {
  const [isRegistering, setIsRegistering] = useState(false)
  const [formData, setFormData] = useState({
    user_id: '',
    email: '',
    password: '',
    client_type: 'hospital',
    organization: ''
  })
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError('')

    try {
      let res
      if (isRegistering) {
        res = await axios.post(`${API_URL}/api/register`, {
          ...formData,
          role: 'admin'
        })
      } else {
        res = await axios.post(`${API_URL}/api/login`, {
          user_id: formData.user_id,
          password: formData.password
        })
      }
      onLogin({
        user_id: res.data.user_id,
        client_type: res.data.client_type,
        organization: formData.organization || 'Organization'
      }, res.data.access_token)
    } catch (err) {
      setError(err.response?.data?.detail || 'Authentication failed')
    }
    setLoading(false)
  }

  return (
    <div style={styles.container}>
      <div style={styles.card}>
        <div style={styles.header}>
          <h1 style={styles.title}>FedHealth<span style={{color: '#667eea'}}>.AI</span></h1>
          <p style={styles.subtitle}>Secure Federated Learning Platform</p>
        </div>

        {error && <div style={styles.error}>{error}</div>}

        <form onSubmit={handleSubmit} style={styles.form}>
          <div style={styles.inputGroup}>
            <label style={styles.label}>User ID</label>
            <input
              type="text"
              required
              style={styles.input}
              value={formData.user_id}
              onChange={e => setFormData({...formData, user_id: e.target.value})}
              placeholder="e.g. hospital_a"
            />
          </div>

          {isRegistering && (
            <>
              <div style={styles.inputGroup}>
                <label style={styles.label}>Email</label>
                <input
                  type="email"
                  required
                  style={styles.input}
                  value={formData.email}
                  onChange={e => setFormData({...formData, email: e.target.value})}
                  placeholder="admin@hospital.com"
                />
              </div>

              <div style={styles.inputGroup}>
                <label style={styles.label}>Client Type</label>
                <select
                  style={styles.select}
                  value={formData.client_type}
                  onChange={e => setFormData({...formData, client_type: e.target.value})}
                >
                  <option value="hospital">🏥 Hospital (Tabular Data)</option>
                  <option value="clinic">⚕️ Clinic (Tabular Data)</option>
                  <option value="lab">🔬 Diagnostic Lab (Images)</option>
                  <option value="iot">⌚ IoT Device (Time Series)</option>
                </select>
              </div>

              <div style={styles.inputGroup}>
                <label style={styles.label}>Organization Name</label>
                <input
                  type="text"
                  required
                  style={styles.input}
                  value={formData.organization}
                  onChange={e => setFormData({...formData, organization: e.target.value})}
                  placeholder="e.g. General Hospital"
                />
              </div>
            </>
          )}

          <div style={styles.inputGroup}>
            <label style={styles.label}>Password</label>
            <input
              type="password"
              required
              style={styles.input}
              value={formData.password}
              onChange={e => setFormData({...formData, password: e.target.value})}
              placeholder="••••••••"
            />
          </div>

          <button type="submit" disabled={loading} style={styles.button}>
            {loading ? 'Processing...' : (isRegistering ? 'Create Account' : 'Sign In')}
          </button>
        </form>

        <div style={styles.footer}>
          <button 
            onClick={() => setIsRegistering(!isRegistering)}
            style={styles.linkButton}
          >
            {isRegistering ? 'Already have an account? Sign In' : 'Need an account? Register'}
          </button>
        </div>
      </div>
    </div>
  )
}

const styles = {
  container: {
    minHeight: '100vh',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    fontFamily: "'Inter', sans-serif",
    padding: '20px'
  },
  card: {
    background: 'white',
    padding: '40px',
    borderRadius: '16px',
    boxShadow: '0 10px 25px rgba(0,0,0,0.1)',
    width: '100%',
    maxWidth: '420px'
  },
  header: { textAlign: 'center', marginBottom: '30px' },
  title: { fontSize: '32px', fontWeight: 'bold', color: '#1f2937', marginBottom: '8px' },
  subtitle: { color: '#6b7280', fontSize: '16px' },
  form: { display: 'flex', flexDirection: 'column', gap: '20px' },
  inputGroup: { display: 'flex', flexDirection: 'column', gap: '8px' },
  label: { fontSize: '14px', fontWeight: '500', color: '#374151' },
  input: {
    padding: '12px',
    borderRadius: '8px',
    border: '1px solid #d1d5db',
    fontSize: '16px',
    outline: 'none',
    transition: 'border-color 0.2s'
  },
  select: {
    padding: '12px',
    borderRadius: '8px',
    border: '1px solid #d1d5db',
    fontSize: '16px',
    background: 'white'
  },
  button: {
    padding: '14px',
    background: '#667eea',
    color: 'white',
    border: 'none',
    borderRadius: '8px',
    fontSize: '16px',
    fontWeight: '600',
    cursor: 'pointer',
    marginTop: '10px',
    transition: 'background 0.2s'
  },
  error: {
    background: '#fee2e2',
    color: '#dc2626',
    padding: '12px',
    borderRadius: '8px',
    marginBottom: '20px',
    fontSize: '14px',
    textAlign: 'center'
  },
  footer: { marginTop: '24px', textAlign: 'center' },
  linkButton: {
    background: 'none',
    border: 'none',
    color: '#667eea',
    cursor: 'pointer',
    fontSize: '14px',
    fontWeight: '500'
  }
}
