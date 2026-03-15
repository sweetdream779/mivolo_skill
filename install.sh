#!/bin/bash
set -e

SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SKILL_DIR/.venv"

echo "=== MiVOLO Skill — Installation ==="
echo ""

# ── Check Python 3 ────────────────────────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
    echo "❌ Error: python3 is not installed or not in PATH."
    echo "   Please install Python 3.8+ and try again."
    echo "   macOS:  brew install python"
    echo "   Ubuntu: sudo apt install python3 python3-venv"
    exit 1
fi

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PYTHON_MAJOR=$(python3 -c "import sys; print(sys.version_info.major)")
PYTHON_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")

if [ "$PYTHON_MAJOR" -lt 3 ] || { [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]; }; then
    echo "❌ Error: Python 3.8+ is required, found Python $PYTHON_VERSION"
    exit 1
fi

echo "✅ Python $PYTHON_VERSION found"

# ── Create virtual environment ─────────────────────────────────────────────────
if [ -d "$VENV_DIR" ]; then
    echo "♻️  Virtual environment already exists at $VENV_DIR, skipping creation."
else
    echo "📦 Creating virtual environment at $VENV_DIR ..."
    python3 -m venv "$VENV_DIR"
    echo "✅ Virtual environment created"
fi

# ── Install dependencies inside venv ──────────────────────────────────────────
echo ""
echo "📥 Installing dependencies..."

"$VENV_DIR/bin/pip" install --upgrade pip --quiet

# PyTorch 2.5.1: try CUDA first, fall back to CPU-only
"$VENV_DIR/bin/pip" install "torch==2.5.1" torchvision \
    --index-url https://download.pytorch.org/whl/cu118 --quiet 2>/dev/null || \
"$VENV_DIR/bin/pip" install "torch==2.5.1" torchvision --quiet

"$VENV_DIR/bin/pip" install \
    "transformers==4.51.0" \
    "accelerate==1.8.1" \
    "ultralytics==8.1.0" \
    "huggingface_hub" \
    "Pillow" \
    "numpy" \
    --quiet

echo "✅ Dependencies installed"

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "=== Installation complete! ==="
echo ""
echo "Models will be downloaded from HuggingFace automatically on first run:"
echo "  - iitolstykh/YOLO-Face-Person-Detector"
echo "  - iitolstykh/mivolo_v2"
echo ""
echo "Usage:"
echo "  $VENV_DIR/bin/python $SKILL_DIR/mivolo_inference.py --image photo.jpg"
echo "  $VENV_DIR/bin/python $SKILL_DIR/mivolo_inference.py --image photo.jpg --draw --output result.jpg"
echo "  $VENV_DIR/bin/python $SKILL_DIR/mivolo_inference.py --image ./photos/ --device cpu"
