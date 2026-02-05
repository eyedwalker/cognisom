# 🚀 Deploy cognisom on AWS

## Option 1: AWS EC2 (Recommended for Full Control)

### **Step 1: Launch EC2 Instance**

```bash
# Recommended instance types:
# - t3.xlarge (4 vCPU, 16GB RAM) - $0.1664/hour - Good for testing
# - c6i.2xlarge (8 vCPU, 16GB RAM) - $0.34/hour - Better performance
# - c6i.4xlarge (16 vCPU, 32GB RAM) - $0.68/hour - Production

# AMI: Ubuntu 22.04 LTS
# Storage: 50GB SSD
# Security Group: Open ports 22 (SSH), 5000 (API), 8080 (Web)
```

### **Step 2: Connect and Setup**

```bash
# SSH into instance
ssh -i your-key.pem ubuntu@your-instance-ip

# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Logout and login again for docker group
exit
ssh -i your-key.pem ubuntu@your-instance-ip
```

### **Step 3: Deploy cognisom**

```bash
# Clone repository
git clone https://github.com/eyedwalker/cognisom.git
cd cognisom

# Build and run
docker-compose up -d

# Check status
docker-compose ps
docker-compose logs -f
```

### **Step 4: Access**

```
API: http://your-instance-ip:5000
Web Dashboard: http://your-instance-ip:8080
```

### **Step 5: Setup Domain (Optional)**

```bash
# Install nginx
sudo apt-get install nginx certbot python3-certbot-nginx

# Configure nginx
sudo nano /etc/nginx/sites-available/cognisom

# Add:
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://localhost:8080;
    }
    
    location /api {
        proxy_pass http://localhost:5000;
    }
}

# Enable site
sudo ln -s /etc/nginx/sites-available/cognisom /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx

# Get SSL certificate
sudo certbot --nginx -d your-domain.com
```

---

## Option 2: AWS ECS (Elastic Container Service)

### **Step 1: Push to ECR**

```bash
# Create ECR repository
aws ecr create-repository --repository-name cognisom

# Login to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com

# Build and push
docker build -t cognisom .
docker tag cognisom:latest YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/cognisom:latest
docker push YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/cognisom:latest
```

### **Step 2: Create ECS Task Definition**

```json
{
  "family": "cognisom",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "4096",
  "containerDefinitions": [
    {
      "name": "cognisom-api",
      "image": "YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/cognisom:latest",
      "portMappings": [
        {
          "containerPort": 5000,
          "protocol": "tcp"
        }
      ],
      "essential": true,
      "environment": [
        {
          "name": "FLASK_ENV",
          "value": "production"
        }
      ]
    }
  ]
}
```

### **Step 3: Create ECS Service**

```bash
# Create cluster
aws ecs create-cluster --cluster-name cognisom-cluster

# Create service
aws ecs create-service \
  --cluster cognisom-cluster \
  --service-name cognisom-service \
  --task-definition cognisom \
  --desired-count 1 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx],securityGroups=[sg-xxx],assignPublicIp=ENABLED}"
```

---

## Option 3: AWS Lambda + API Gateway (Serverless)

### **For API-only deployment**

```bash
# Install Serverless Framework
npm install -g serverless

# Create serverless.yml
# See deployment/aws/serverless.yml

# Deploy
serverless deploy
```

---

## Cost Estimates

### **EC2 Pricing** (us-east-1):
- **t3.xlarge**: ~$120/month (24/7)
- **c6i.2xlarge**: ~$245/month (24/7)
- **c6i.4xlarge**: ~$490/month (24/7)

### **ECS Fargate Pricing**:
- **2 vCPU, 4GB RAM**: ~$60/month (24/7)
- **4 vCPU, 8GB RAM**: ~$120/month (24/7)

### **Lambda Pricing**:
- **Free tier**: 1M requests/month
- **After**: $0.20 per 1M requests

---

## Recommended Setup

### **For Development/Testing**:
```
EC2 t3.xlarge + Docker Compose
Cost: ~$120/month
```

### **For Production**:
```
ECS Fargate + ALB + CloudFront
Cost: ~$150-200/month
Auto-scaling enabled
```

### **For API-only**:
```
Lambda + API Gateway
Cost: ~$10-50/month (usage-based)
```

---

## Monitoring

### **CloudWatch**

```bash
# Enable CloudWatch logs
aws logs create-log-group --log-group-name /ecs/cognisom

# Set up alarms
aws cloudwatch put-metric-alarm \
  --alarm-name cognisom-cpu-high \
  --alarm-description "CPU usage > 80%" \
  --metric-name CPUUtilization \
  --namespace AWS/ECS \
  --statistic Average \
  --period 300 \
  --threshold 80 \
  --comparison-operator GreaterThanThreshold
```

---

## Security Best Practices

1. **Use VPC**: Deploy in private subnet
2. **Use ALB**: Application Load Balancer for HTTPS
3. **Use Secrets Manager**: Store API keys
4. **Enable CloudTrail**: Audit logging
5. **Use IAM roles**: No hardcoded credentials
6. **Enable encryption**: EBS volumes, S3 buckets

---

## Backup Strategy

```bash
# Automated snapshots
aws ec2 create-snapshot \
  --volume-id vol-xxx \
  --description "cognisom-backup-$(date +%Y%m%d)"

# S3 backup
aws s3 sync /app/data s3://cognisom-backups/$(date +%Y%m%d)/
```

---

## Quick Start Commands

```bash
# One-line deployment
git clone https://github.com/eyedwalker/cognisom.git && \
cd cognisom && \
docker-compose up -d

# Check logs
docker-compose logs -f cognisom-api

# Stop
docker-compose down

# Update
git pull && docker-compose up -d --build
```

---

## Support

For AWS-specific issues:
- AWS Support: https://console.aws.amazon.com/support
- AWS Documentation: https://docs.aws.amazon.com
