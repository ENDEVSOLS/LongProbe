#!/bin/bash
# Setup script for demo environment

set -e

echo "Setting up demo environment..."

# Create fresh venv
rm -rf demos/.demo_venv
python3 -m venv demos/.demo_venv

# Activate venv
source demos/.demo_venv/bin/activate

# Install longprobe from PyPI
pip install -q --upgrade pip
pip install -q longprobe[chroma,huggingface]

# Verify installation
which longprobe
longprobe --help | head -5

echo "✓ Demo environment ready!"
echo "To use: source demos/.demo_venv/bin/activate"
