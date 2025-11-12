"""
Google Colab Setup Script
==========================

Run this in Google Colab to set up cognisom without cloning.

Usage in Colab:
1. Upload this file to Colab
2. Run: !python colab_setup.py
3. Or copy/paste the code below
"""

# Install dependencies
print("Installing dependencies...")
import subprocess
subprocess.run(["pip", "install", "-q", "numpy", "scipy", "matplotlib", "flask", "flask-cors"])

# Create directory structure
import os
os.makedirs("cognisom", exist_ok=True)
os.chdir("cognisom")

print("✅ Setup complete!")
print("\nNow you can:")
print("1. Upload cognisom files to this directory")
print("2. Or copy/paste the modules directly")
print("\nAlternatively, use this direct installation:")

# Direct installation code
installation_code = """
# OPTION 1: Direct file upload
# Upload these files from your local cognisom folder:
# - core/ (entire folder)
# - modules/ (entire folder)  
# - scenarios/ (entire folder)
# - api/ (entire folder)

# OPTION 2: Copy/paste code
# See the code cells below for direct implementation
"""

print(installation_code)
