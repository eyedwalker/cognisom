# 🆓 FREE Deployment Options

## ✅ Best Free Options (No Credit Card!)

---

## Option 1: Google Colab (FREE GPU!) ⭐ RECOMMENDED

### **What You Get**:
- ✅ FREE Tesla T4 GPU
- ✅ 12GB RAM
- ✅ 12 hours runtime
- ✅ No credit card needed
- ✅ Perfect for testing

### **Deploy in 2 Minutes**:

1. **Go to**: https://colab.research.google.com
2. **Create new notebook**
3. **Run this**:

```python
# Install and run cognisom
!git clone https://github.com/eyedwalker/cognisom.git
%cd cognisom

# Install dependencies
!pip install -q numpy scipy matplotlib flask flask-cors

# Test the platform
!python3 test_platform.py
```

4. **Run scenarios**:
```python
# Run immunotherapy scenario
!python3 scenarios/immunotherapy.py

# Run hypoxia scenario
!python3 scenarios/hypoxia.py
```

5. **Start API server**:
```python
# Start API in background
!python3 api/rest_server.py &

# Access via Colab's built-in tools
from google.colab import output
output.serve_kernel_port_as_window(5000)
```

### **Limitations**:
- 12 hour limit (then restart)
- Can't run 24/7
- Perfect for testing and demos

---

## Option 2: Replit (FREE, Always On) ⭐ GREAT FOR DEMOS

### **What You Get**:
- ✅ FREE hosting
- ✅ Public URL
- ✅ Web IDE
- ✅ No credit card
- ✅ Always accessible

### **Deploy in 3 Minutes**:

1. **Go to**: https://replit.com
2. **Sign up** (free)
3. **Create new Repl** → Import from GitHub
4. **Enter**: `https://github.com/eyedwalker/cognisom`
5. **Click "Import"**
6. **Run**:

```bash
# In Replit shell:
pip install -r requirements.txt
python3 api/rest_server.py
```

7. **Access**: Replit gives you a public URL!

### **Limitations**:
- Limited CPU/RAM
- Slower than paid options
- Good for demos

---

## Option 3: Render (FREE Tier)

### **What You Get**:
- ✅ FREE web service
- ✅ Public URL
- ✅ Auto-deploy from GitHub
- ✅ HTTPS included

### **Deploy in 5 Minutes**:

1. **Go to**: https://render.com
2. **Sign up** (free, no credit card)
3. **New** → **Web Service**
4. **Connect GitHub**: `eyedwalker/cognisom`
5. **Settings**:
   - Name: `cognisom`
   - Environment: `Python 3`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `python api/rest_server.py`
6. **Create Web Service**

### **Your URL**: `https://cognisom.onrender.com`

### **Limitations**:
- Spins down after 15 min idle
- Takes 30s to wake up
- 750 hours/month free

---

## Option 4: Railway (FREE $5 Credit)

### **What You Get**:
- ✅ $5 free credit/month
- ✅ ~100 hours runtime
- ✅ Fast deployment
- ✅ Good performance

### **Deploy in 3 Minutes**:

1. **Go to**: https://railway.app
2. **Sign up** with GitHub
3. **New Project** → **Deploy from GitHub**
4. **Select**: `cognisom` repo
5. **Auto-deploys!**

### **Limitations**:
- $5/month limit
- Need credit card for verification (not charged)

---

## Option 5: Hugging Face Spaces (FREE)

### **What You Get**:
- ✅ FREE hosting
- ✅ Public URL
- ✅ Great for ML demos
- ✅ Community visibility

### **Deploy in 5 Minutes**:

1. **Go to**: https://huggingface.co/spaces
2. **Create new Space**
3. **Choose**: Gradio or Streamlit
4. **Upload** cognisom files
5. **Create** `app.py`:

```python
import gradio as gr
from core import SimulationEngine, SimulationConfig
from modules import CellularModule, ImmuneModule

def run_simulation(duration):
    engine = SimulationEngine(SimulationConfig(dt=0.01, duration=duration))
    engine.register_module('cellular', CellularModule)
    engine.register_module('immune', ImmuneModule)
    engine.initialize()
    
    immune = engine.modules['immune']
    cellular = engine.modules['cellular']
    immune.set_cellular_module(cellular)
    
    engine.run()
    state = engine.get_state()
    
    return f"""
    Time: {state['time']:.2f}h
    Cancer cells: {state['cellular']['n_cancer']}
    Immune kills: {state['immune']['total_kills']}
    """

demo = gr.Interface(
    fn=run_simulation,
    inputs=gr.Slider(0.1, 24, value=1, label="Duration (hours)"),
    outputs="text",
    title="cognisom Simulator"
)

demo.launch()
```

---

## Option 6: AWS Free Tier (12 Months)

### **What You Get**:
- ✅ 750 hours/month t2.micro
- ✅ 12 months free
- ✅ Need credit card (not charged unless you exceed limits)

### **Deploy**:

1. **Sign up**: https://aws.amazon.com/free
2. **Launch EC2** → t2.micro (free tier)
3. **SSH and deploy**:

```bash
ssh -i your-key.pem ubuntu@instance-ip

# Install Docker
curl -fsSL https://get.docker.com | sh

# Deploy
git clone https://github.com/eyedwalker/cognisom.git
cd cognisom
docker-compose up -d
```

### **Limitations**:
- t2.micro is small (1GB RAM)
- Need credit card
- Only free for 12 months

---

## Option 7: Google Cloud Free Tier

### **What You Get**:
- ✅ $300 credit (90 days)
- ✅ Always free tier after
- ✅ e2-micro instance free forever

### **Deploy**:

1. **Sign up**: https://cloud.google.com/free
2. **Create VM** → e2-micro
3. **Deploy** (same as AWS)

---

## 🎯 MY RECOMMENDATION FOR FREE

### **For Testing & Development**:
**Google Colab** ⭐
```
✅ FREE GPU
✅ No setup
✅ Start in 2 minutes
✅ Perfect for learning
```

### **For Public Demos**:
**Replit** ⭐
```
✅ Always accessible
✅ Public URL
✅ Easy to share
✅ Web IDE included
```

### **For Production (Free Tier)**:
**Render** ⭐
```
✅ Auto-deploy from GitHub
✅ HTTPS included
✅ 750 hours/month
✅ Professional URL
```

---

## 📊 Comparison

| Platform | Cost | GPU | Always On | Setup Time | Best For |
|----------|------|-----|-----------|------------|----------|
| **Google Colab** | FREE | ✅ T4 | ❌ 12h | 2 min | Testing ⭐ |
| **Replit** | FREE | ❌ | ✅ | 3 min | Demos ⭐ |
| **Render** | FREE | ❌ | ⚠️ Sleeps | 5 min | Production ⭐ |
| **Railway** | $5/mo | ❌ | ✅ | 3 min | Good value |
| **HF Spaces** | FREE | ❌ | ✅ | 5 min | ML demos |
| **AWS Free** | FREE* | ❌ | ✅ | 15 min | 12 months |

*Requires credit card

---

## 🚀 Quick Start (Right Now!)

### **Option 1: Google Colab** (Fastest!)

1. Go to: https://colab.research.google.com
2. New notebook
3. Paste and run:

```python
!git clone https://github.com/eyedwalker/cognisom.git
%cd cognisom
!pip install -q numpy scipy matplotlib flask flask-cors
!python3 test_platform.py
```

**Done! Running in 2 minutes!**

### **Option 2: Replit** (Public URL!)

1. Go to: https://replit.com
2. Import from GitHub: `https://github.com/eyedwalker/cognisom`
3. Run: `python3 api/rest_server.py`

**Done! Public URL in 3 minutes!**

---

## 💡 Tips for Free Tiers

### **Google Colab**:
- Save your work to Google Drive
- Restart every 12 hours
- Use GPU runtime for faster simulations

### **Replit**:
- Keep tab open to prevent sleep
- Use "Always On" (paid) if needed
- Great for sharing demos

### **Render**:
- Wakes up in 30s from sleep
- Perfect for occasional use
- Upgrade to paid for always-on

---

## 🎓 Learning Path (All Free!)

### **Day 1: Test Locally**
```bash
python3 test_platform.py
```

### **Day 2: Try Google Colab**
- Run simulations
- Test scenarios
- Learn the API

### **Day 3: Deploy to Replit**
- Public demo
- Share with others
- Get feedback

### **Day 4: Deploy to Render**
- Production-ready
- Custom domain
- HTTPS

---

## ✅ What to Do RIGHT NOW

### **Fastest Option** (2 minutes):

1. **Open**: https://colab.research.google.com
2. **New notebook**
3. **Copy/paste**:

```python
# Clone and setup
!git clone https://github.com/eyedwalker/cognisom.git
%cd cognisom
!pip install -q numpy scipy matplotlib flask flask-cors

# Test everything
!python3 test_platform.py

# Run a scenario
!python3 scenarios/immunotherapy.py

# Start API
!python3 api/rest_server.py &
```

4. **Run!**

**You're now running cognisom for FREE with a GPU!** 🎉

---

## 📞 Support

- Google Colab: https://colab.research.google.com
- Replit: https://replit.com
- Render: https://render.com
- Railway: https://railway.app
- Hugging Face: https://huggingface.co

---

## 🎯 Summary

**Best FREE options**:
1. ⭐ **Google Colab** - FREE GPU, perfect for testing
2. ⭐ **Replit** - Public URL, great for demos
3. ⭐ **Render** - Production-ready, auto-deploy

**Start with Google Colab RIGHT NOW - it's the fastest!**

No credit card needed. No setup. Just run! 🚀
