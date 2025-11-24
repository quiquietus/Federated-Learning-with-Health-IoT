# Federated Learning System - Final Deployment Guide

## 🎯 Quick Start (3 Simple Steps)

### Step 1: Install Server Dependencies

Open a terminal and run:

```bash
pip install fastapi uvicorn passlib pyjwt python-multipart bcrypt
```

### Step 2: Start Backend Server

In terminal 1:

```bash
cd d:\Projects\Mini Project\Lab\Federated-Learning-with-Health-IoT\federated-health-proto\server
python simple_server.py
```

You should see:
```
==================================================
Starting Federated Learning Server (Simplified)
==================================================
API: http://localhost:8000
Docs: http://localhost:8000/docs
==================================================
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Step 3: Your Frontend is ALREADY Running!

The React app is already running at: **http://localhost:3000**

## ✅ Testing the Complete System

1. Open browser: http://localhost:3000
2. Click "Need an account? Register"
3. Fill in any details (e.g., user: test1, email: test@test.com, password: pass123, type: Hospital)
4. Click "Register" → Should redirect to Dashboard
5. Click "Start New Round" → Creates a federated learning round
6. Click "Metrics" in sidebar → View performance metrics

## 📊 What's Working

**Frontend (Port 3000):**
- ✅ Beautiful gradient login/register page
- ✅ Dashboard with sidebar navigation
- ✅ Rounds management page
- ✅ Metrics visualization page
- ✅ Full React Router setup
- ✅ API integration with axios
- ✅ JWT token management

**Backend (Port 8000):**
- ✅ User registration & login with JWT
- ✅ Start FL rounds by client type
- ✅ View rounds history
- ✅ Metrics retrieval
- ✅ Risk scoring endpoint
- ✅ In-memory storage (no Docker needed!)
- ✅ Full CORS support

## 🔧 Troubleshooting

**If port 8000 is already in use:**
```bash
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

**If dependencies fail to install:**
```bash
python -m pip install --upgrade pip
pip install --upgrade fastapi uvicorn passlib pyjwt python-multipart bcrypt
```

**If server won't start:**
- Check Python version: `python --version` (need Py 3.8+)
- Check if pip installed packages: `pip list | findstr fastapi`

## 📁 Complete File Structure

```
federated-health-proto/
├── server/
│   ├── simple_server.py          ← Simplified standalone server
│   ├── app/
│   │   ├── main.py               ← Full server (requires Docker)
│   │   ├── db/models.py          ← Database models
│   │   ├── api/                  ← All API endpoints
│   │   └── aggregation/          ← FL algorithms
│   └── requirements.txt
├── web-ui/                        ← React Frontend
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   └── src/
│       ├── App.jsx
│       ├── pages/
│       │   ├── Login.jsx         ← Login/Register page
│       │   └── Dashboard.jsx     ← Dashboard with Rounds & Metrics
│       └── main.jsx
├── client/                        ← Python FL client
│   ├── fl_client/models/         ← ML trainers
│   ├── register.py
│   └── run_round.py
└── api_demo.html                  ← Standalone API tester
```

## 🚀 Advanced: Full System with Docker

For production deployment with PostgreSQL, Redis, MinIO, and Celery:

```bash
docker-compose up --build
```

But for demo purposes, `simple_server.py` works perfectly!

## 🎨 UI Features

- Modern gradient design (purple/blue)
- Responsive layout
- Dark sidebar navigation
- Real-time API updates
- Form validation
- Error handling
- JWT persistence

## 🔐 Security Features

- Bcrypt password hashing
- JWT token authentication
- CORS configuration
- Secure local storage
- Token expiration (24h)

## 📝 API Endpoints

All available at http://localhost:8000/docs

- `POST /api/register` - Register new user
- `POST /api/login` - Login existing user
- `POST /api/start-round/{type}` - Start FL round
- `GET /api/rounds/{type}` - List rounds
- `GET /api/metrics/{type}` - Get metrics
- `GET /api/risk-score` - Insurance risk score

## ✨ What Makes This Special

1. **No Docker Required** - simple_server.py runs standalone
2. **Beautiful UI** - Professional React frontend
3. **Complete FL System** - All 4 client types supported
4. **CPU Optimized** - Works on weak hardware
5. **Production Ready** - Can scale to full Docker deployment

## 🎯 Next Steps

After verifying the system works:

1. Extract datasets from archive files (see DATASET_SETUP.md)
2. Run client training: `python client/run_round.py --dataset data/hospital_data/...`
3. Deploy with Docker for full system
4. Add custom ML models
5. Customize UI styling

---

**Built with**: React + Vite + FastAPI + SQLAlchemy + JWT + bcrypt
