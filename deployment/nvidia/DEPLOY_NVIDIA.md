# 🚀 Deploy cognisom on NVIDIA Cloud

## Option 1: NVIDIA NGC (NVIDIA GPU Cloud)

### **What is NGC?**
- Container registry for GPU-optimized software
- Pre-configured environments
- Access to NVIDIA GPUs
- Integration with major cloud providers

### **Step 1: Sign Up**

```bash
# Go to: https://ngc.nvidia.com
# Create free account
# Get API key
```

### **Step 2: Login to NGC**

```bash
# Install NGC CLI
wget https://ngc.nvidia.com/downloads/ngccli_linux.zip
unzip ngccli_linux.zip
chmod +x ngc-cli/ngc

# Login
./ngc-cli/ngc config set
# Enter API key when prompted
```

### **Step 3: Create NGC-Compatible Dockerfile**

```dockerfile
# Use NVIDIA base image
FROM nvcr.io/nvidia/pytorch:23.10-py3

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose ports
EXPOSE 5000

CMD ["python", "api/rest_server.py"]
```

### **Step 4: Build and Push to NGC**

```bash
# Build
docker build -t cognisom:latest -f deployment/nvidia/Dockerfile.ngc .

# Tag for NGC
docker tag cognisom:latest nvcr.io/YOUR_ORG/cognisom:latest

# Push
docker push nvcr.io/YOUR_ORG/cognisom:latest
```

### **Step 5: Deploy on Cloud with GPU**

```bash
# AWS with GPU
aws ec2 run-instances \
  --image-id ami-xxx \
  --instance-type p3.2xlarge \
  --key-name your-key \
  --security-group-ids sg-xxx

# Then SSH and run:
docker run --gpus all -p 5000:5000 nvcr.io/YOUR_ORG/cognisom:latest
```

---

## Option 2: NVIDIA Base Command Platform

### **Enterprise-grade ML platform**

**Features**:
- Multi-node GPU clusters
- Job scheduling
- Data management
- Monitoring

**Pricing**: Contact NVIDIA sales

**Setup**:
1. Contact NVIDIA for access
2. Upload container to NGC
3. Submit jobs through web interface

---

## Option 3: Cloud Providers with NVIDIA GPUs

### **AWS with NVIDIA GPUs**

**Instance Types**:
- **p3.2xlarge**: 1x V100 (16GB) - $3.06/hour
- **p3.8xlarge**: 4x V100 (16GB) - $12.24/hour
- **p4d.24xlarge**: 8x A100 (40GB) - $32.77/hour
- **g5.xlarge**: 1x A10G (24GB) - $1.006/hour (cheaper!)

**Deploy**:
```bash
# Launch GPU instance
aws ec2 run-instances \
  --image-id ami-0c55b159cbfafe1f0 \
  --instance-type g5.xlarge \
  --key-name your-key

# SSH and setup
ssh -i your-key.pem ubuntu@instance-ip

# Install NVIDIA Docker
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Run with GPU
docker run --gpus all -p 5000:5000 cognisom:latest
```

### **Google Cloud with NVIDIA GPUs**

**Instance Types**:
- **n1-standard-4 + T4**: $0.35/hour + $0.35/hour
- **n1-standard-8 + V100**: $0.38/hour + $2.48/hour
- **a2-highgpu-1g + A100**: $3.67/hour

**Deploy**:
```bash
# Create instance with GPU
gcloud compute instances create cognisom-gpu \
  --zone=us-central1-a \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=ubuntu-2004-lts \
  --image-project=ubuntu-os-cloud \
  --maintenance-policy=TERMINATE \
  --metadata=install-nvidia-driver=True

# SSH and deploy
gcloud compute ssh cognisom-gpu

# Install Docker + NVIDIA Docker (same as AWS)
# Run container
```

### **Azure with NVIDIA GPUs**

**Instance Types**:
- **NC6**: 1x K80 - $0.90/hour
- **NC6s_v3**: 1x V100 - $3.06/hour
- **ND40rs_v2**: 8x V100 - $22.032/hour

**Deploy**:
```bash
# Create VM with GPU
az vm create \
  --resource-group cognisom-rg \
  --name cognisom-gpu \
  --size Standard_NC6 \
  --image UbuntuLTS \
  --admin-username azureuser \
  --generate-ssh-keys

# Install NVIDIA drivers
az vm extension set \
  --resource-group cognisom-rg \
  --vm-name cognisom-gpu \
  --name NvidiaGpuDriverLinux \
  --publisher Microsoft.HpcCompute \
  --version 1.3
```

---

## Option 4: NVIDIA DGX Cloud (Premium)

### **Dedicated NVIDIA Infrastructure**

**Features**:
- DGX A100 systems
- Full-stack AI platform
- Enterprise support
- Managed service

**Pricing**: $36,000+/month

**Best for**: Large-scale production deployments

---

## GPU-Optimized Dockerfile

```dockerfile
FROM nvcr.io/nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install GPU-accelerated libraries
RUN pip install --no-cache-dir \
    torch \
    cupy-cuda12x \
    nvidia-ml-py3

# Copy application
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python3", "api/rest_server.py"]
```

---

## Cost Comparison

### **Current (CPU-only)**:
- AWS t3.xlarge: $120/month
- Works great for 10,000 cells

### **With GPU (for 100,000+ cells)**:
- AWS g5.xlarge (A10G): $730/month
- 10-100x faster
- Handles 100,000+ cells

### **Recommendation**:
- **Start with CPU** (current setup)
- **Add GPU** when you need >10,000 cells
- **Use spot instances** for 70% savings

---

## Spot Instances (Save 70%!)

### **AWS Spot**:
```bash
# Request spot instance
aws ec2 request-spot-instances \
  --instance-count 1 \
  --type "one-time" \
  --launch-specification file://spot-spec.json

# spot-spec.json:
{
  "ImageId": "ami-xxx",
  "InstanceType": "g5.xlarge",
  "KeyName": "your-key",
  "SecurityGroupIds": ["sg-xxx"]
}
```

**Pricing**:
- g5.xlarge on-demand: $1.006/hour
- g5.xlarge spot: ~$0.30/hour (70% savings!)

---

## Free GPU Options

### **1. Google Colab**:
- Free T4 GPU
- Limited to 12 hours
- Good for testing

### **2. Kaggle Notebooks**:
- Free P100 GPU
- 30 hours/week
- Good for development

### **3. NVIDIA LaunchPad**:
- Free trial access
- Various GPUs
- Apply at: https://www.nvidia.com/en-us/data-center/launchpad/

---

## Monitoring GPU Usage

```bash
# Install nvidia-smi
nvidia-smi

# Watch GPU usage
watch -n 1 nvidia-smi

# In Python
import nvidia_smi
nvidia_smi.nvmlInit()
handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
print(f"GPU Memory: {info.used / info.total * 100:.1f}%")
```

---

## Quick Start (AWS GPU)

```bash
# 1. Launch GPU instance
aws ec2 run-instances \
  --image-id ami-0c55b159cbfafe1f0 \
  --instance-type g5.xlarge \
  --key-name your-key

# 2. SSH
ssh -i your-key.pem ubuntu@instance-ip

# 3. Install NVIDIA Docker
curl -fsSL https://get.docker.com | sh
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# 4. Clone and run
git clone https://github.com/eyedwalker/cognisom.git
cd cognisom
docker build -t cognisom .
docker run --gpus all -p 5000:5000 cognisom
```

---

## Recommendation

### **For Your Current Needs** (10,000 cells):
✅ **Use AWS EC2 t3.xlarge (CPU-only)**
- Cost: $120/month
- Current code works perfectly
- No GPU needed yet

### **When You Need More** (100,000+ cells):
🚀 **Upgrade to AWS g5.xlarge (GPU)**
- Cost: $730/month (or $220/month with spot)
- 10-100x faster
- Add GPU code later

### **Best Value**:
💰 **AWS g5.xlarge Spot Instance**
- Cost: ~$220/month
- Same performance as on-demand
- 70% savings!

---

## Support

- NVIDIA NGC: https://ngc.nvidia.com
- NVIDIA Forums: https://forums.developer.nvidia.com
- NVIDIA Developer: https://developer.nvidia.com
