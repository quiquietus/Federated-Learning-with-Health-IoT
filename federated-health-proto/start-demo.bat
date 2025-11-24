@echo off
echo ========================================
echo Starting Federated Learning Platform
echo (Without Docker - Simplified Mode)
echo ========================================
echo.

echo Note: This demo skips PostgreSQL, Redis, MinIO, and Celery
echo It runs a simplified FastAPI server for demonstration
echo.

echo Starting Frontend (React + Vite)...
cd web-ui
start cmd /k "npm run dev"

echo.
echo ========================================
echo Services Starting:
echo ========================================
echo Frontend: http://localhost:3000
echo.
echo Wait 10-15 seconds for services to start...
echo Then open: http://localhost:3000
echo ========================================
