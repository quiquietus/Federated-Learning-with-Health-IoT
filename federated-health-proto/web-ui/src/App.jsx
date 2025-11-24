import { useState, useEffect } from 'react'
import { Routes, Route, Navigate, useNavigate, useLocation } from 'react-router-dom'
import axios from 'axios'
import Login from './pages/Login'
import Dashboard from './pages/Dashboard'

const API_URL = 'http://localhost:8000'

function App() {
  const [user, setUser] = useState(null)
  const [loading, setLoading] = useState(true)
  const navigate = useNavigate()
  const location = useLocation()

  useEffect(() => {
    checkAuth()
  }, [])

  const checkAuth = async () => {
    const token = localStorage.getItem('token')
    if (!token) {
      setLoading(false)
      return
    }

    try {
      const res = await axios.get(`${API_URL}/api/me`, {
        headers: { Authorization: `Bearer ${token}` }
      })
      setUser(res.data)
    } catch (err) {
      console.error(err)
      localStorage.removeItem('token')
    }
    setLoading(false)
  }

  const handleLogin = (userData, token) => {
    localStorage.setItem('token', token)
    setUser(userData)
    navigate('/dashboard')
  }

  const handleLogout = async () => {
    try {
      await axios.post(`${API_URL}/api/logout`, {}, {
        headers: { Authorization: `Bearer ${localStorage.getItem('token')}` }
      })
    } catch (err) {
      console.error(err)
    }
    localStorage.removeItem('token')
    setUser(null)
    navigate('/login')
  }

  if (loading) return <div style={styles.loading}>Loading...</div>

  return (
    <div className="app">
      <Routes>
        <Route path="/login" element={
          user ? <Navigate to="/dashboard" /> : <Login onLogin={handleLogin} />
        } />
        
        <Route path="/dashboard/*" element={
          user ? <Dashboard user={user} onLogout={handleLogout} /> : <Navigate to="/login" />
        } />
        
        <Route path="/" element={<Navigate to={user ? "/dashboard" : "/login"} />} />
      </Routes>
    </div>
  )
}

const styles = {
  loading: {
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    height: '100vh',
    background: '#f3f4f6',
    color: '#4b5563',
    fontSize: '1.2rem'
  }
}

export default App
