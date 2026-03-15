#!/bin/bash
set -e

echo "Installing MiVOLO Skill dependencies..."

# Check Python version
python3 -c "import sys; assert sys.version_info >= (3, 8), 'Python 3.8+ required'" || {
    echo "Error: Python 3.8+ is required"
    exit 1
}

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 2>/dev/null || \
pip install torch torchvision  # fallback to CPU version

pip install "transformers==4.51.0" "accelerate==1.8.1" Pillow

# Install MiVOLO library from GitHub
pip install git+https://github.com/WildChlamydia/MiVOLO.git

echo ""
echo "✅ MiVOLO Skill installed successfully!"
echo ""
echo "Usage:"
echo "  python mivolo_inference.py --image path/to/image.jpg"
echo "  python mivolo_inference.py --image path/to/image.jpg --draw --output result.jpg"
echo "  python mivolo_inference.py --image path/to/folder/ --device cpu"
