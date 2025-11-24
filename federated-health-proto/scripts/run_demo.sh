#!/bin/bash
# Demo script: Full end-to-end federated learning demo

set -e

echo "========================================="
echo "Federated Learning - Complete Demo"
echo "========================================="
echo ""

# Check if datasets exist
if [ ! -d "data" ]; then
    echo "[1/5] Extracting datasets from archive files..."
    bash scripts/download_datasets.sh
else
    echo "[1/5] ✓ Datasets already extracted"
fi

echo ""
echo "[2/5] Starting services with Docker Compose..."
docker-compose up -d

echo "Waiting for services to be ready (30s)..."
sleep 30

echo ""
echo "[3/5] Services status:"
docker-compose ps

echo ""
echo "[4/5] Creating demo clients..."

# Install client dependencies
cd client
pip install -q -r requirements.txt

# Register demo clients
python register.py http://localhost:8000 hospital_demo demo@hospital.com password123 hospital "Demo Hospital" doctor || true
python register.py http://localhost:8000 clinic_demo demo@clinic.com password123 clinic "Demo Clinic" doctor || true
python register.py http://localhost:8000 lab_demo demo@lab.com password123 lab "Demo Lab" doctor || true
python register.py http://localhost:8000 iot_demo demo@iot.com password123 iot "Demo IoT Hub" other || true

cd ..

echo ""
echo "[5/5] Demo is ready!"
echo ""
echo "========================================="
echo "ACCESS POINTS:"
echo "========================================="
echo "Frontend:      http://localhost:3000"
echo "API Docs:      http://localhost:8000/docs"
echo "Min IO Console: http://localhost:9001"
echo ""
echo "DEMO ACCOUNTS:"
echo "- hospital_demo / password123"
echo "- clinic_demo / password123"
echo "- lab_demo / password123"
echo "- iot_demo / password123"
echo ""
echo "NEXT STEPS:"
echo "1. Open http://localhost:3000 in your browser"
echo "2. Login with any demo account"
echo "3. Start a federated learning round"
echo "4. Run client training:"
echo "   cd client"
echo "   python run_round.py --dataset ../data/hospital_data/heart_failure.csv"
echo ""
echo "========================================="
