#!/bin/bash
# ============================================================================
# Cognisom NVIDIA Brev Setup Script - MAXED OUT
# ============================================================================
# For L40S/H100 GPU instance
# Includes: Cognisom + Omniverse Nucleus + LLM + ESMFold + RAPIDS
# ============================================================================

set -e

echo "=========================================="
echo "Cognisom Brev GPU Setup - FULL STACK"
echo "=========================================="

# Update system
sudo apt-get update
sudo apt-get install -y git curl wget build-essential python3-pip python3-venv \
    docker.io docker-compose nvidia-container-toolkit

# Add user to docker group
sudo usermod -aG docker $USER

# Verify NVIDIA GPU
echo ""
echo "Checking NVIDIA GPU..."
nvidia-smi

# Get GPU memory to determine what we can run
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
echo "GPU Memory: ${GPU_MEM} MiB"

# Clone Cognisom repository
echo ""
echo "Cloning Cognisom..."
cd ~
if [ -d "cognisom" ]; then
    cd cognisom && git pull
else
    git clone https://github.com/eyentelligence/cognisom.git
    cd cognisom
fi

# Create Python virtual environment
echo ""
echo "Setting up Python environment..."
python3 -m venv venv
source venv/bin/activate

# Install base requirements
pip install --upgrade pip
pip install -r requirements.txt

# Install GPU-specific packages
echo ""
echo "Installing GPU packages..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install cupy-cuda12x
pip install nvidia-ml-py3

# Install RAPIDS for GPU-accelerated data science
echo ""
echo "Installing RAPIDS (GPU DataFrame/ML)..."
pip install cudf-cu12 cuml-cu12 cugraph-cu12 --extra-index-url=https://pypi.nvidia.com

# Install vLLM for efficient LLM serving
echo ""
echo "Installing vLLM for LLM inference..."
pip install vllm

# Create data directories
mkdir -p data/auth data/scrna data/simulation data/models

# Set environment variables
cat > .env << 'EOF'
# Cognisom Environment Configuration
COGNISOM_ENV=production
COGNISOM_GPU_ENABLED=true
CUDA_VISIBLE_DEVICES=0

# Omniverse Configuration
OMNIVERSE_URL=omniverse://localhost/cognisom
OMNIVERSE_NUCLEUS_PORT=3009

# LLM Configuration (vLLM)
LLM_MODEL=meta-llama/Llama-3.2-8B-Instruct
LLM_PORT=8000
LLM_MAX_MODEL_LEN=4096

# Dashboard Configuration
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0

# API Configuration
API_HOST=0.0.0.0
API_PORT=5000
EOF

# Create Docker Compose for all services
cat > docker-compose.gpu.yml << 'EOF'
version: '3.8'

services:
  # Omniverse Nucleus Server
  nucleus:
    image: nvcr.io/nvidia/omniverse/nucleus:2023.2.0
    container_name: cognisom-nucleus
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - ACCEPT_EULA=Y
    ports:
      - "3009:3009"   # Web UI
      - "3019:3019"   # API
      - "3030:3030"   # Collaboration
    volumes:
      - nucleus_data:/var/lib/omniverse/nucleus
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  # vLLM Server for Research Agent
  llm:
    image: vllm/vllm-openai:latest
    container_name: cognisom-llm
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    ports:
      - "8000:8000"
    command: >
      --model meta-llama/Llama-3.2-8B-Instruct
      --max-model-len 4096
      --gpu-memory-utilization 0.4
      --dtype half
    volumes:
      - model_cache:/root/.cache/huggingface
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  # ESMFold for protein structure prediction
  esmfold:
    image: nvcr.io/nvidia/clara/esmfold:1.0
    container_name: cognisom-esmfold
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    ports:
      - "8001:8001"
    restart: unless-stopped
    profiles:
      - protein  # Only start when explicitly requested
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

volumes:
  nucleus_data:
  model_cache:
EOF

# Create service management scripts
cat > start_all.sh << 'EOF'
#!/bin/bash
# Start all Cognisom GPU services
set -e

cd ~/cognisom
source venv/bin/activate

echo "Starting GPU services..."

# Start Docker services (Nucleus + LLM)
sudo docker-compose -f docker-compose.gpu.yml up -d nucleus llm

# Wait for services to be ready
echo "Waiting for services to initialize..."
sleep 10

# Start Cognisom API
python -m api.rest_server &
echo $! > /tmp/cognisom_api.pid

# Start Streamlit dashboard
streamlit run cognisom/dashboard/app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 &
echo $! > /tmp/cognisom_dashboard.pid

echo ""
echo "=========================================="
echo "All services started!"
echo "=========================================="
echo "Dashboard:    http://localhost:8501"
echo "LLM API:      http://localhost:8000"
echo "Nucleus:      http://localhost:3009"
echo "=========================================="
EOF
chmod +x start_all.sh

cat > stop_all.sh << 'EOF'
#!/bin/bash
# Stop all Cognisom services
cd ~/cognisom

echo "Stopping services..."

# Stop Python processes
[ -f /tmp/cognisom_api.pid ] && kill $(cat /tmp/cognisom_api.pid) 2>/dev/null
[ -f /tmp/cognisom_dashboard.pid ] && kill $(cat /tmp/cognisom_dashboard.pid) 2>/dev/null

# Stop Docker services
sudo docker-compose -f docker-compose.gpu.yml down

echo "All services stopped."
EOF
chmod +x stop_all.sh

cat > start_cognisom.sh << 'EOF'
#!/bin/bash
# Start just Cognisom (no Docker services)
source ~/cognisom/venv/bin/activate
cd ~/cognisom

python -m api.rest_server &
API_PID=$!

streamlit run cognisom/dashboard/app.py --server.port 8501 --server.address 0.0.0.0

kill $API_PID 2>/dev/null
EOF
chmod +x start_cognisom.sh

cat > start_protein.sh << 'EOF'
#!/bin/bash
# Start ESMFold service (uses ~8GB VRAM)
cd ~/cognisom
sudo docker-compose -f docker-compose.gpu.yml --profile protein up -d esmfold
echo "ESMFold available at http://localhost:8001"
EOF
chmod +x start_protein.sh

# GPU monitoring script
cat > gpu_monitor.sh << 'EOF'
#!/bin/bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi
EOF
chmod +x gpu_monitor.sh

# GPU verification script
cat > verify_gpu.py << 'EOF'
#!/usr/bin/env python3
"""Verify GPU setup for Cognisom."""
import sys

print("=" * 60)
print("Cognisom GPU Verification - FULL STACK")
print("=" * 60)

# Check PyTorch CUDA
try:
    import torch
    print(f"\n✓ PyTorch {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM: {vram:.1f} GB")

        # Recommend configuration based on VRAM
        print(f"\n  Recommended config for {vram:.0f}GB VRAM:")
        if vram >= 80:
            print("    → Can run: 70B LLM + Nucleus + ESMFold simultaneously")
        elif vram >= 48:
            print("    → Can run: 8B LLM + Nucleus always")
            print("    → ESMFold: on-demand (stop LLM first for large proteins)")
        elif vram >= 24:
            print("    → Can run: 8B LLM OR Nucleus (not both)")
        else:
            print("    → Limited: Run services one at a time")
except ImportError:
    print("✗ PyTorch not installed")

# Check CuPy
try:
    import cupy as cp
    print(f"\n✓ CuPy {cp.__version__}")
except ImportError:
    print("\n✗ CuPy not installed")

# Check RAPIDS
try:
    import cudf
    print(f"✓ RAPIDS cuDF {cudf.__version__}")
except ImportError:
    print("✗ RAPIDS not installed")

# Check vLLM
try:
    import vllm
    print(f"✓ vLLM {vllm.__version__}")
except ImportError:
    print("⚠ vLLM not installed (LLM serving disabled)")

# Check NVIDIA ML
try:
    import pynvml
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    name = pynvml.nvmlDeviceGetName(handle)
    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print(f"\n✓ NVIDIA ML Python")
    print(f"  GPU: {name}")
    print(f"  Memory: {mem.used/1e9:.1f} / {mem.total/1e9:.1f} GB used")
    pynvml.nvmlShutdown()
except:
    print("✗ NVIDIA ML not available")

# Check Docker
import subprocess
try:
    result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
    print(f"\n✓ {result.stdout.strip()}")
except:
    print("\n✗ Docker not available")

print("\n" + "=" * 60)
print("Quick Start Commands:")
print("=" * 60)
print("  ./start_all.sh      - Start everything (Cognisom+LLM+Nucleus)")
print("  ./start_cognisom.sh - Start just Cognisom dashboard")
print("  ./start_protein.sh  - Add ESMFold service")
print("  ./stop_all.sh       - Stop all services")
print("  ./gpu_monitor.sh    - Watch GPU usage")
print("=" * 60)
EOF

# Run verification
echo ""
echo "Verifying GPU setup..."
python3 verify_gpu.py

echo ""
echo "=========================================="
echo "Setup Complete - FULL GPU STACK!"
echo "=========================================="
echo ""
echo "Services included:"
echo "  ✓ Cognisom Dashboard (port 8501)"
echo "  ✓ Cognisom API (port 5000)"
echo "  ✓ Omniverse Nucleus (port 3009)"
echo "  ✓ LLM via vLLM (port 8000)"
echo "  ✓ ESMFold protein prediction (port 8001)"
echo "  ✓ RAPIDS GPU data science"
echo ""
echo "To start everything:"
echo "  cd ~/cognisom && ./start_all.sh"
echo ""
echo "Expose these ports in Brev:"
echo "  8501 - Dashboard"
echo "  8000 - LLM API"
echo "  3009 - Nucleus"
echo "=========================================="
