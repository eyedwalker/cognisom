# 🖥️ GitHub Codespaces GUI Setup (Fixed)

## ✅ Working Setup for GUI in Codespaces

### **Problem You're Seeing**:
- noVNC gives "404 File not found"
- VNC server not properly started

### **Solution** (Step-by-Step):

---

## 📋 Complete Working Setup

### **Step 1: Install Dependencies**

```bash
# Update and install packages
sudo apt-get update
sudo apt-get install -y python3-tk x11-apps

# Install VNC server
sudo apt-get install -y tigervnc-standalone-server tigervnc-common

# Install lightweight window manager
sudo apt-get install -y xfce4 xfce4-goodies
```

---

### **Step 2: Configure VNC Password**

```bash
# Set VNC password
vncpasswd
# Enter password: cognisom
# Verify: cognisom
# View-only password: n
```

---

### **Step 3: Create VNC Startup Script**

```bash
# Create xstartup file
mkdir -p ~/.vnc
cat > ~/.vnc/xstartup << 'EOF'
#!/bin/bash
xrdb $HOME/.Xresources
startxfce4 &
EOF

# Make executable
chmod +x ~/.vnc/xstartup
```

---

### **Step 4: Start VNC Server**

```bash
# Kill any existing VNC servers
vncserver -kill :1 2>/dev/null

# Start VNC server
vncserver :1 -geometry 1920x1080 -depth 24 -localhost no

# Verify it's running
ps aux | grep vnc
```

You should see output like:
```
Xvnc :1 -geometry 1920x1080 -depth 24 ...
```

---

### **Step 5: Install and Start noVNC**

```bash
# Go to home directory
cd ~

# Clone noVNC (if not already done)
if [ ! -d "noVNC" ]; then
    git clone https://github.com/novnc/noVNC.git
fi

cd noVNC

# Install websockify
git clone https://github.com/novnc/websockify.git

# Start noVNC proxy
./utils/novnc_proxy --vnc localhost:5901
```

---

### **Step 6: Forward Port in Codespaces**

1. Click **PORTS** tab (bottom panel)
2. Find port **6080**
3. Click **globe icon** to make public
4. Click **link** to open in browser

---

### **Step 7: Connect to Desktop**

1. Browser opens to noVNC page
2. Click **Connect**
3. Enter password: `cognisom`
4. You should see XFCE desktop!

---

### **Step 8: Run cognisom GUI**

In the VNC desktop, open terminal and run:

```bash
cd /workspaces/cognisom
export DISPLAY=:1
python3 ui/control_panel.py
```

**GUI should appear!** 🎉

---

## 🔧 Troubleshooting

### **Issue: "404 File not found"**

**Cause**: VNC server not running or wrong port

**Fix**:
```bash
# Check VNC is running
vncserver -list

# Should show:
# TigerVNC server sessions:
# X DISPLAY #     PROCESS ID
# :1              12345

# If not running, start it:
vncserver :1 -geometry 1920x1080 -depth 24 -localhost no
```

---

### **Issue: "Connection refused"**

**Cause**: noVNC not connected to right port

**Fix**:
```bash
# VNC runs on port 5901 (display :1)
# noVNC should connect to localhost:5901

cd ~/noVNC
./utils/novnc_proxy --vnc localhost:5901
```

---

### **Issue: "Black screen in VNC"**

**Cause**: No window manager

**Fix**:
```bash
# Install XFCE
sudo apt-get install -y xfce4

# Restart VNC
vncserver -kill :1
vncserver :1 -geometry 1920x1080 -depth 24
```

---

## 🚀 Quick One-Line Setup

Copy and paste this entire block:

```bash
# Complete setup in one go
sudo apt-get update && \
sudo apt-get install -y python3-tk x11-apps tigervnc-standalone-server tigervnc-common xfce4 xfce4-goodies && \
mkdir -p ~/.vnc && \
echo -e "cognisom\ncognisom\nn\n" | vncpasswd && \
cat > ~/.vnc/xstartup << 'EOF'
#!/bin/bash
xrdb $HOME/.Xresources
startxfce4 &
EOF
chmod +x ~/.vnc/xstartup && \
vncserver -kill :1 2>/dev/null ; \
vncserver :1 -geometry 1920x1080 -depth 24 -localhost no && \
cd ~ && \
git clone https://github.com/novnc/noVNC.git 2>/dev/null ; \
cd noVNC && \
git clone https://github.com/novnc/websockify.git 2>/dev/null ; \
echo "✅ Setup complete! Now run: cd ~/noVNC && ./utils/novnc_proxy --vnc localhost:5901"
```

Then run:
```bash
cd ~/noVNC && ./utils/novnc_proxy --vnc localhost:5901
```

---

## 📝 Alternative: Simpler Setup (No Desktop)

If you just want to run the GUI without full desktop:

```bash
# Install minimal X11
sudo apt-get update
sudo apt-get install -y python3-tk xvfb x11vnc

# Start virtual display
Xvfb :99 -screen 0 1920x1080x24 &
export DISPLAY=:99

# Start VNC server on that display
x11vnc -display :99 -forever -shared -rfbport 5900 &

# Install noVNC
cd ~
git clone https://github.com/novnc/noVNC.git
cd noVNC
./utils/novnc_proxy --vnc localhost:5900
```

Then forward port 6080 and connect!

---

## 🎯 What Should Happen

### **Correct Flow**:
1. VNC server starts → Creates display :1 on port 5901
2. noVNC proxy starts → Connects to port 5901, serves on port 6080
3. Browser connects to port 6080 → Shows VNC desktop
4. You see XFCE desktop
5. Open terminal in desktop
6. Run cognisom GUI
7. GUI appears!

### **Your Current Issue**:
- noVNC is running but can't find VNC server
- Need to ensure VNC server is running first

---

## ✅ Verification Steps

### **1. Check VNC is running**:
```bash
vncserver -list
# Should show :1 with a process ID
```

### **2. Check VNC port**:
```bash
netstat -tlnp | grep 5901
# Should show LISTEN on port 5901
```

### **3. Check noVNC**:
```bash
# Should be running on port 6080
netstat -tlnp | grep 6080
```

### **4. Test VNC connection**:
```bash
# Try connecting with vncviewer (if available)
vncviewer localhost:5901
```

---

## 💡 Recommended Approach

**Use the one-line setup above**, then:

1. Wait for it to complete
2. Run: `cd ~/noVNC && ./utils/novnc_proxy --vnc localhost:5901`
3. Forward port 6080 in Codespaces
4. Open in browser
5. Connect with password: `cognisom`
6. Open terminal in desktop
7. Run: `cd /workspaces/cognisom && python3 ui/control_panel.py`

---

## 🆘 Still Not Working?

### **Alternative: Use X11 Forwarding from Local Mac**

If VNC is too complex, you can use X11 forwarding:

1. **On your Mac**, install XQuartz:
```bash
brew install --cask xquartz
```

2. **Restart your Mac**

3. **SSH into Codespaces with X11**:
```bash
# Get your codespace name from GitHub
gh codespace ssh --codespace YOUR_CODESPACE_NAME -- -Y
```

4. **Run GUI**:
```bash
cd /workspaces/cognisom
python3 ui/control_panel.py
```

GUI appears on your Mac!

---

## 📊 Summary

**Issue**: noVNC can't find VNC server (404 error)

**Solution**: 
1. Start VNC server first: `vncserver :1`
2. Verify it's running: `vncserver -list`
3. Then start noVNC: `./utils/novnc_proxy --vnc localhost:5901`
4. Forward port 6080
5. Connect in browser

**Easier Alternative**: Use the one-line setup script above!

---

**Try the one-line setup - it handles everything automatically!** 🚀
