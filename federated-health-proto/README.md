# Federated Learning for Healthcare - Production System

A complete, production-grade cross-silo horizontal federated learning web application optimized for weak CPU hardware (2 cores, 4-8GB RAM). Supports 4 distinct client types with specialized models for healthcare data.

## 📋 Overview

This system implements federated learning across different healthcare organizations, keeping data local while collaboratively training global models:

- **Hospitals**: Heart failure prediction (LightGBM on tabular data)
- **Clinics**: Health status classification (LightGBM on tabular data)
- **Diagnostic Labs**: Blood cell classification (MobileNetV2 on images)  
- **IoT Device Hubs**: Activity recognition (1D-CNN on time-series)

**Key Features:**
- ✅ Optimized for CPU training (2 cores, 4-8GB RAM)
- ✅ FedAvg aggregation for PyTorch models
- ✅ Prediction-based distillation for LightGBM
- ✅ Privacy-preserving (data stays local)
- ✅ Production-ready with Docker Compose
- ✅ Risk scoring for insurance analysts
- ✅ Complete web dashboard

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- Python 3.10+
- Pre-downloaded datasets (see below)

### 1. Setup Environment

```bash
# Clone and enter directory
cd federated-health-proto

# Copy environment file
cp .env.example .env

# Edit .env with your credentials (optional for local dev)
```

### 2. Organize Datasets

The project expects the following dataset archives in the main directory:
- `archive (4).zip` - Heart Failure Clinical Data (Hospitals)
- `archive (3).zip` - BCCD Blood Cell Images (Labs)
- `archive (2).zip` - Health Status Dataset (Clinics)
- `archive (1).zip` - Student Lifestyle Dataset (IoT)

Extract and organize them:

```bash
bash scripts/download_datasets.sh
```

This will extract all archives into organized directories under `./data/`

### 4. Start Services

```bash
# Start all services
make start

# Or for CPU-limited resources
make start-cpu
```

Services will be available at:
- **Frontend**: http://localhost:3000
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **MinIO Console**: http://localhost:9001

### 5. Register a Client

```bash
cd client

# Install dependencies
pip install -r requirements.txt

# Register
python register.py \
  http://localhost:8000 \
  hospital_user_1 \
  user@hospital.com \
  password123 \
  hospital \
  "City Hospital"
```

### 6. Run a Training Round

```bash
# From federated-health-proto/client directory
python run_round.py \
  --dataset ../data/hospital_data/heart_failure.csv \
  --local-epochs 1 \
  --batch-size 8 \
  --threads 1
```

## 📁 Project Structure

```
federated-health-proto/
├── server/                  # FastAPI backend
│   ├── app/
│   │   ├── main.py         # FastAPI application
│   │   ├── api/            # API endpoints
│   │   ├── aggregation/    # FedAvg & distillation
│   │   ├── db/             # Database models
│   │   └── model_store/    # MinIO integration
│   ├── requirements.txt
│   └── Dockerfile
├── client/                  # Python client
│   ├── fl_client/
│   │   ├── models/         # ML trainers
│   │   └── client.py       # API communication
│   ├── register.py
│   ├── run_round.py
│   └── requirements.txt
├── web-ui/                  # React frontend
├── scripts/
│   ├── download_datasets.sh
│   └── run_demo.sh
├── tests/
├── docker-compose.yml
└── README.md
```

## 🔧 Configuration

### CPU Optimization

For weak hardware, adjust in `.env`:

```bash
CLIENT_THREADS=1
CLIENT_BATCH_SIZE=4      # Reduce if OOM
CLIENT_LOCAL_EPOCHS=1    # Increase for better accuracy
```

Or via command line:

```bash
python run_round.py --dataset data.csv --batch-size 4 --threads 1 --local-epochs 1
```

### Model-Specific Settings

**LightGBM (Hospitals/Clinics)**:
- Training time: <30s on 2-core CPU
- Memory: ~500MB
- Parameters: 50 boosting rounds, max_depth=4

**MobileNetV2 (Labs)**:
- Training time: 3-5 min on CPU (3 epochs)
- Memory: ~1-2GB
- Frozen backbone, only head trained

**1D-CNN (IoT)**:
- Training time: 1-2 min on CPU
- Memory: ~500MB  
- Tiny architecture: 16→32→64 channels

## 🏗️ Architecture

### Federated Learning Flow

1. **Server starts a round** for a client type
2. **Clients download** current global model
3. **Local training** on client device (data never leaves)
4. **Clients upload** model updates (compressed)
5. **Server aggregates** using FedAvg or distillation
6. **New global model** saved and made available

### Aggregation Methods

**PyTorch Models (Labs, IoT)**:
- Standard FedAvg: weighted averaging of state_dicts
- Weights: proportional to client sample counts

**LightGBM Models (Hospitals, Clinics)**:
- Prediction aggregation: clients send predictions on proxy set
- Distillation: server trains new LightGBM on aggregated soft labels
- Handles heterogeneous tree structures

## 🔒 Security & Privacy

- **JWT authentication** for all API endpoints
- **HTTPS** in production (TLS certificates required)
- **Data locality**: training data never leaves client
- **Compressed uploads**: gzip compression reduces bandwidth
- **Optional secure aggregation**: raw updates deleted after aggregation

## 📊 Dashboard Features

- **Metrics visualization**: accuracy, F1, precision, recall over rounds
- **Global model downloads**: versioned models per round
- **Risk scoring**: insurance analyst dashboard with recommendations
- **Round management**: start/monitor federated learning rounds  
- **Dark/light mode**: modern responsive UI

## 🧪 Testing

```bash
# Run all tests
make test

# Unit tests only
make test-unit

# Integration tests
make test-integration
```

## 📦 Client Packaging

Create standalone executable for users without Python:

```bash
cd client
bash package_client.sh
```

Produces: `dist/fl_client` (executable)

## 🚧 Limitations

- **LightGBM distillation** approximates tree ensembles (not exact averaging)
- **CPU training** slower than GPU (but optimized for 2-core machines)
- **Proxy set quality** affects LightGBM aggregation accuracy
- **No differential privacy** (can be added as extension)

## 🛠️ Development

```bash
# Start services in development mode
docker-compose up --build

# View logs
docker-compose logs -f server

# Stop services
docker-compose down

# Clean everything
make clean
```

## 📖 API Documentation

Once server is running, visit:
- Swagger UI: http://localhost:8000/docs  
- ReDoc: http://localhost:8000/redoc

### Key Endpoints

- `POST /api/register` - Register new user
- `POST /api/login` - Login and get JWT
- `GET /api/global-model/{client_type}/latest` - Download global model
- `POST /api/client-update` - Submit model update
- `POST /api/start-round/{client_type}` - Start new round (admin)
- `GET /api/metrics/{client_type}` - Get aggregated metrics

## 🐛 Troubleshooting

**"Out of memory" during training**:
- Reduce batch size: `--batch-size 4`
- Use CPU limits: `docker-compose -f docker-compose.cpu.yml up`

**"No active round" error**:
- Start a round via API or frontend
- Check round status: `GET /api/rounds/{client_type}`

**Model download fails**:
- Ensure at least one round has completed
- Check MinIO is running: http://localhost:9001

**Slow training**:
- Reduce local epochs: `--local-epochs 1`
- Limit threads: `--threads 1`
- Use smaller datasets for testing

## 📝 License

MIT License - see LICENSE file

## 🙏 Acknowledgments

Datasets used:
- Heart Failure Clinical Data (andrewmvd/heart-failure-clinical-data)
- BCCD Blood Cell Count and Detection (orvile/bccd-blood-cell-count-and-detection-dataset)
- Health Status Dataset (jacobhealth/health-status-dataset)
- Student Lifestyle Dataset (steve1215rogg/student-lifestyle-dataset)

## 📧 Support

For issues or questions, please open a GitHub issue.

---

**Built with**: FastAPI, PyTorch, LightGBM, React, PostgreSQL, MinIO, Docker
