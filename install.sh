#!/bin/bash
set -e

SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SKILL_DIR/.venv"

echo "=== MiVOLO Skill — Installation ==="
echo ""

# ── Check Python 3 ────────────────────────────────────────────────────────────
PYTHON_CMD=""
for cmd in python3.10 python3; do
    if command -v "$cmd" &>/dev/null; then
        ver=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        if [ "$ver" = "3.10" ]; then
            PYTHON_CMD="$cmd"
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "❌ Error: Python 3.10 is required but not found."
    echo "   macOS:  brew install python@3.10"
    echo "   Ubuntu: sudo apt install python3.10 python3.10-venv"
    exit 1
fi

PYTHON_VERSION=$("$PYTHON_CMD" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "✅ Python $PYTHON_VERSION found ($PYTHON_CMD)"

# ── Create virtual environment ─────────────────────────────────────────────────
if [ -d "$VENV_DIR" ]; then
    echo "♻️  Virtual environment already exists at $VENV_DIR, skipping creation."
else
    echo "📦 Creating virtual environment at $VENV_DIR ..."
    "$PYTHON_CMD" -m venv "$VENV_DIR"
    echo "✅ Virtual environment created"
fi

# ── Install dependencies inside venv ──────────────────────────────────────────
echo ""
echo "📥 Installing dependencies..."

"$VENV_DIR/bin/pip" install --upgrade pip "setuptools<71" --quiet

# PyTorch 2.5.1: try CUDA first, fall back to CPU-only
"$VENV_DIR/bin/pip" install "torch==2.5.1" torchvision \
    --index-url https://download.pytorch.org/whl/cu118 --quiet 2>/dev/null || \
"$VENV_DIR/bin/pip" install "torch==2.5.1" torchvision --quiet

"$VENV_DIR/bin/pip" install \
    "transformers==4.57.1" \
    "accelerate==1.8.1" \
    "ultralytics==8.1.0" \
    "huggingface_hub" \
    "Pillow" \
    "numpy" \
    "requests" \
    --quiet

# MiVOLO uses pkg_resources in setup.py; --no-build-isolation avoids
# pip creating an isolated env with the latest setuptools (which dropped pkg_resources).
"$VENV_DIR/bin/pip" install \
    "git+https://github.com/WildChlamydia/MiVOLO.git" \
    --no-build-isolation --quiet

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
