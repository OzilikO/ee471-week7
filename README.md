# EE 471 Week 7: Image Classification with PyTorch, Cog & Docker

This repository contains the deliverables for **EE 471 Modern Software Development Practices and Technologies - Week 7 In-Class Exercise**.

It includes training pipelines, Hyperparameter Optimization (HPO), Cog ML containerization, and Flask web applications with side-by-side model comparison for both **Fashion-MNIST** and **CIFAR-10**.

---

## Repository Structure

```text
ee471-week7/
├── fashion-mnist/
│   ├── train_baseline.py       # Baseline 3-layer MLP training (SGD, 5 epochs)
│   ├── train_hpo.py            # HPO training with Adam optimizer (10 epochs)
│   ├── baseline_model.pth      # Saved baseline model checkpoint
│   ├── optimized_model.pth     # Saved HPO model checkpoint
│   ├── predict.py              # Cog Predictor with auto-preprocessing & live terminal logs
│   ├── cog.yaml                # Cog container specification (CPU-optimized)
│   ├── app.py                  # Flask web interface with side-by-side comparison
│   ├── Dockerfile              # Web UI Docker build
│   ├── docker-compose.yml      # Orchestration (Cog inference + Web UI)
│   ├── requirements.txt        # Cog container dependencies
│   ├── requirements-web.txt    # Web frontend dependencies
│   ├── templates/index.html    # Side-by-side comparison UI
│   ├── static/                 # CSS styles & JavaScript
│   └── sample_tshirt.png       # Test image
│
├── cifar10/
│   ├── cifar_baseline.py       # Baseline ConvNet training (SGD with momentum, 2 epochs)
│   ├── cifar_hpo.py            # HPO ConvNet training (Adam, batch 32, 5 epochs)
│   ├── baseline_model.pth      # Saved baseline ConvNet checkpoint
│   ├── optimized_model.pth     # Saved HPO ConvNet checkpoint
│   ├── predict.py              # Cog Predictor for 3-channel CIFAR images
│   ├── cog.yaml                # Cog container specification (CPU-optimized)
│   ├── app.py                  # Flask web interface (port 8081)
│   ├── Dockerfile              # Web UI Docker build
│   ├── docker-compose.yml      # Orchestration (Cog inference + Web UI)
│   ├── requirements.txt        # Cog container dependencies
│   ├── requirements-web.txt    # Web frontend dependencies
│   ├── templates/index.html    # Side-by-side comparison UI
│   ├── static/                 # CSS styles & JavaScript
│   ├── sample_car.png          # Test image
│   └── sample_ship.png         # Test image
│
└── README.md
```

---

## Setup & Running Instructions

### 1. Fashion-MNIST Garment Classifier

#### Option A: Running with Docker Compose (Recommended)
1. Build the Cog inference model image:
   ```bash
   cd fashion-mnist
   cog build -t garment-classifier-cog
   ```
2. Start both the Cog backend and Flask frontend:
   ```bash
   docker compose up -d --build
   ```
3. Open your browser at `http://localhost:8080`.
4. Inspect live container logs:
   ```bash
   docker logs -f garment-classifier-model
   ```

#### Option B: Running Locally (Direct Python)
```bash
cd fashion-mnist
pip install -r requirements-web.txt
python3 app.py
```

---

### 2. CIFAR-10 Object Classifier

#### Option A: Running with Docker Compose (Recommended)
1. Build the Cog inference model image:
   ```bash
   cd cifar10
   cog build -t cifar-classifier-cog
   ```
2. Start both the Cog backend and Flask frontend:
   ```bash
   docker compose up -d --build
   ```
3. Open your browser at `http://localhost:8081`.
4. Inspect live container logs:
   ```bash
   docker logs -f cifar-classifier-model
   ```

#### Option B: Running Locally (Direct Python)
```bash
cd cifar10
pip install -r requirements-web.txt
python3 app.py
```

---

## API Verification (Postman / cURL)

You can send a POST request with an image URL to verify the Cog inference endpoint:

### Fashion-MNIST (Port 5002):
```bash
curl -X POST http://localhost:5002/predictions \
     -H "Content-Type: application/json" \
     -d '{"input": {"image": "https://gist.githubusercontent.com/bfirsh/3c2115692682ae260932a67d93fd94a8/raw/56b19f53f7643bb6c0b822c410c366c3a6244de2/mystery.jpg"}}'
```

### CIFAR-10 (Port 5003):
```bash
curl -X POST http://localhost:5003/predictions \
     -H "Content-Type: application/json" \
     -d '{"input": {"image": "https://gist.githubusercontent.com/bfirsh/3c2115692682ae260932a67d93fd94a8/raw/56b19f53f7643bb6c0b822c410c366c3a6244de2/mystery.jpg"}}'
```
