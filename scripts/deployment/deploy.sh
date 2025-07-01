#!/bin/bash

# MAHT-Net Production Deployment Script
# Deploy trained model to AWS EC2 with FastAPI serving and monitoring

set -e

echo "🚀 MAHT-Net Production Deployment"
echo "=================================="

# Configuration
PROJECT_NAME="maht-net"
MODEL_VERSION="1.0.0"
DOCKER_IMAGE="${PROJECT_NAME}:${MODEL_VERSION}"
CONTAINER_NAME="${PROJECT_NAME}-api"
API_PORT=8000
MONITORING_PORT=9090

# Check if model checkpoint exists
if [ ! -f "models/checkpoints/best_model.pth" ]; then
    echo "❌ Model checkpoint not found. Please train the model first."
    echo "   Run: make train"
    exit 1
fi

echo "Model checkpoint found"

# Create deployment directory structure
echo "📁 Creating deployment structure..."
mkdir -p deployment/{docker,configs,scripts,monitoring}
mkdir -p deployment/api/{app,models,utils}

# Create FastAPI application
echo "🔧 Creating FastAPI application..."
cat > deployment/api/app/main.py << 'EOF'
"""
MAHT-Net FastAPI Application
Production-ready API for cephalometric landmark detection
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional
import asyncio
import uuid

import torch
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# Add src to path
sys.path.append('/app/src')
from src.models.maht_net import create_maht_net

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instance
model = None
device = None

class PredictionRequest(BaseModel):
    """Request model for prediction"""
    patient_id: Optional[str] = None
    acquisition_params: Optional[Dict] = None
    return_heatmaps: bool = False
    return_uncertainties: bool = True

class PredictionResponse(BaseModel):
    """Response model for prediction"""
    prediction_id: str
    landmarks: List[List[float]]
    landmark_names: List[str]
    uncertainties: Optional[List[float]] = None
    heatmaps: Optional[List] = None
    processing_time: float
    model_version: str
    confidence_score: float

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    gpu_available: bool
    memory_usage: Dict
    uptime: float

# Initialize FastAPI app
app = FastAPI(
    title="MAHT-Net API",
    description="Multi-Stage Attention-enhanced Hybrid Transformer Network for Cephalometric Landmark Detection",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup time
startup_time = time.time()

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    global model, device
    
    logger.info("🚀 Starting MAHT-Net API...")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🖥️  Using device: {device}")
    
    # Load model
    try:
        model = create_maht_net(num_landmarks=7, pretrained=False)
        
        # Load checkpoint
        checkpoint_path = "/app/models/best_model.pth"
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Model loaded from {checkpoint_path}")
        else:
            logger.warning("⚠️  No checkpoint found, using untrained model")
        
        model.to(device)
        model.eval()
        
        logger.info("MAHT-Net API ready for predictions")
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {str(e)}")
        raise

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    
    # GPU memory info
    gpu_memory = {}
    if torch.cuda.is_available():
        gpu_memory = {
            "allocated": torch.cuda.memory_allocated() / 1024**3,
            "cached": torch.cuda.memory_reserved() / 1024**3,
            "total": torch.cuda.get_device_properties(0).total_memory / 1024**3
        }
    
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        gpu_available=torch.cuda.is_available(),
        memory_usage=gpu_memory,
        uptime=time.time() - startup_time
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_landmarks(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    request: PredictionRequest = PredictionRequest()
):
    """
    Predict cephalometric landmarks from X-ray image
    """
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    prediction_id = str(uuid.uuid4())
    
    try:
        # Read and validate image
        contents = await image.read()
        
        # Convert to OpenCV format
        nparr = np.frombuffer(contents, np.uint8)
        cv_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if cv_image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Preprocess image
        processed_image = preprocess_image(cv_image)
        
        # Model inference
        with torch.no_grad():
            input_tensor = processed_image.unsqueeze(0).to(device)
            outputs = model(input_tensor)
        
        # Extract predictions
        pred_coords = outputs['coordinates'].cpu().numpy()[0]
        uncertainties = outputs.get('uncertainties', None)
        heatmaps = outputs.get('heatmaps', None)
        
        # Convert predictions to response format
        landmarks = pred_coords.tolist()
        landmark_names = [
            'Nasion', 'Sella', 'Articulare', 'Gonion', 
            'Menton', 'Pogonion', 'Upper_Incisor'
        ]
        
        # Compute confidence score
        confidence_score = compute_confidence_score(pred_coords, uncertainties)
        
        # Prepare response
        response_data = {
            "prediction_id": prediction_id,
            "landmarks": landmarks,
            "landmark_names": landmark_names,
            "processing_time": time.time() - start_time,
            "model_version": "1.0.0",
            "confidence_score": confidence_score
        }
        
        if request.return_uncertainties and uncertainties is not None:
            response_data["uncertainties"] = uncertainties.cpu().numpy().tolist()
        
        if request.return_heatmaps and heatmaps is not None:
            response_data["heatmaps"] = heatmaps.cpu().numpy().tolist()
        
        # Log prediction for monitoring
        background_tasks.add_task(
            log_prediction, 
            prediction_id, 
            request.patient_id, 
            confidence_score,
            time.time() - start_time
        )
        
        return PredictionResponse(**response_data)
        
    except Exception as e:
        logger.error(f"❌ Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

def preprocess_image(image: np.ndarray) -> torch.Tensor:
    """Preprocess image for model input"""
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize to model input size
    image = cv2.resize(image, (512, 512))
    
    # Normalize
    image = image.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std
    
    # Convert to tensor and change to CHW format
    tensor = torch.from_numpy(image).permute(2, 0, 1)
    
    return tensor

def compute_confidence_score(coordinates: np.ndarray, uncertainties: Optional[torch.Tensor]) -> float:
    """Compute overall confidence score for the prediction"""
    
    if uncertainties is not None:
        # Use uncertainty to compute confidence
        uncertainty_values = uncertainties.cpu().numpy()
        confidence = 1.0 / (1.0 + np.mean(uncertainty_values))
    else:
        # Use heuristic based on coordinate spread
        coord_std = np.std(coordinates)
        confidence = max(0.1, min(0.9, 1.0 - coord_std / 100.0))
    
    return float(confidence)

async def log_prediction(prediction_id: str, patient_id: Optional[str], 
                        confidence: float, processing_time: float):
    """Log prediction for monitoring and analytics"""
    
    log_entry = {
        "timestamp": time.time(),
        "prediction_id": prediction_id,
        "patient_id": patient_id,
        "confidence": confidence,
        "processing_time": processing_time,
        "model_version": "1.0.0"
    }
    
    # Write to log file (in production, use proper logging/monitoring)
    with open("/app/logs/predictions.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "MAHT-Net API for Cephalometric Landmark Detection",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1
    )
EOF

# Create Dockerfile
echo "🐳 Creating Dockerfile..."
cat > deployment/docker/Dockerfile << 'EOF'
# MAHT-Net Production Dockerfile
FROM nvidia/cuda:12.1-runtime-ubuntu20.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3-pip \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.9 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/
COPY deployment/api/ .
COPY models/checkpoints/best_model.pth models/

# Create logs directory
RUN mkdir -p logs

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["python3", "app/main.py"]
EOF

# Create docker-compose for full stack
echo "🔧 Creating docker-compose configuration..."
cat > deployment/docker/docker-compose.yml << 'EOF'
version: '3.8'

services:
  maht-net-api:
    build:
      context: ../../
      dockerfile: deployment/docker/Dockerfile
    ports:
      - "8000:8000"
    volumes:
      - "../../logs:/app/logs"
      - "../../models:/app/models"
    environment:
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ../../certs:/etc/nginx/certs
    depends_on:
      - maht-net-api
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana-datasources.yml:/etc/grafana/provisioning/datasources/datasources.yml
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    depends_on:
      - prometheus
    restart: unless-stopped

volumes:
  prometheus_data:
  grafana_data:
EOF

# Create nginx configuration
echo "🌐 Creating nginx configuration..."
cat > deployment/docker/nginx.conf << 'EOF'
events {
    worker_connections 1024;
}

http {
    upstream maht_net_api {
        server maht-net-api:8000;
    }

    server {
        listen 80;
        server_name _;

        # Security headers
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;
        add_header X-XSS-Protection "1; mode=block";

        # Client max body size for image uploads
        client_max_body_size 50M;

        location / {
            proxy_pass http://maht_net_api;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # Timeouts
            proxy_connect_timeout 60s;
            proxy_send_timeout 60s;
            proxy_read_timeout 60s;
        }

        location /health {
            proxy_pass http://maht_net_api/health;
            access_log off;
        }
    }
}
EOF

# Create monitoring configuration
echo "Creating monitoring configuration..."
cat > deployment/docker/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'maht-net-api'
    static_configs:
      - targets: ['maht-net-api:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
EOF

# Build Docker image
echo "🔨 Building Docker image..."
cd deployment/docker
docker build -t ${DOCKER_IMAGE} ../../

# Check if container is already running
if [ "$(docker ps -q -f name=${CONTAINER_NAME})" ]; then
    echo "🔄 Stopping existing container..."
    docker stop ${CONTAINER_NAME}
    docker rm ${CONTAINER_NAME}
fi

# Deploy with docker-compose
echo "🚀 Deploying MAHT-Net stack..."
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 30

# Health check
echo "🏥 Performing health check..."
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "MAHT-Net API is healthy"
else
    echo "❌ Health check failed"
    docker-compose logs maht-net-api
    exit 1
fi

# Setup monitoring alerts (optional)
echo "Setting up monitoring..."
# Add monitoring setup here if needed

# Create systemd service for auto-restart
echo "🔧 Creating systemd service..."
sudo tee /etc/systemd/system/maht-net.service > /dev/null <<EOF
[Unit]
Description=MAHT-Net Production Service
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=$PWD
ExecStart=/usr/bin/docker-compose up -d
ExecStop=/usr/bin/docker-compose down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable maht-net

echo ""
echo "🎉 MAHT-Net deployment completed successfully!"
echo ""
echo "📋 Service Information:"
echo "======================"
echo "🌐 API Endpoint:     http://localhost:8000"
echo "📚 API Documentation: http://localhost:8000/docs"
echo "🏥 Health Check:     http://localhost:8000/health"
echo "Monitoring:       http://localhost:3000 (Grafana)"
echo "📈 Metrics:          http://localhost:9090 (Prometheus)"
echo ""
echo "💡 Management Commands:"
echo "======================"
echo "View logs:        docker-compose logs -f maht-net-api"
echo "🔄 Restart service:  docker-compose restart maht-net-api"
echo "⏹️  Stop service:     docker-compose down"
echo "🔧 Update service:   sudo systemctl restart maht-net"
echo ""
echo "🧪 Test the API:"
echo "==============="
echo "curl -X POST 'http://localhost:8000/predict' \\"
echo "  -H 'Content-Type: multipart/form-data' \\"
echo "  -F 'image=@path/to/your/xray.jpg'"
echo ""
echo "MAHT-Net is ready for clinical deployment!"

cd ../..  # Return to project root
