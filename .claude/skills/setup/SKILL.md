---
name: setup
description: Install MiVOLO skill for Claude Code. Run this after cloning the repo. Handles Python check, venv creation, dependency installation, and skill registration.
user-invocable: true
---

# MiVOLO Setup

Automatically install the MiVOLO age & gender detection skill for Claude Code.

## Steps

### 1. Check Python 3.10

```bash
python3.10 --version
```

If Python 3.10 is not found, tell the user to install it:
- macOS: `brew install python@3.10`
- Ubuntu: `sudo apt install python3.10 python3.10-venv`

### 2. Register skill files

Copy the skill to the user's global Claude Code skills directory:

```bash
mkdir -p ~/.claude/skills/mivolo_skill
cp SKILL.md mivolo_inference.py install.sh requirements.txt ~/.claude/skills/mivolo_skill/
```

### 3. Run install script from the skills directory

```bash
bash ~/.claude/skills/mivolo_skill/install.sh
```

This creates an isolated `.venv/` inside `~/.claude/skills/mivolo_skill/` and installs all dependencies (PyTorch, Transformers, ultralytics, MiVOLO). Takes 2-5 minutes depending on network speed.

If installation fails:
- **"No module named venv"** → `sudo apt install python3.10-venv` (Linux)
- **PyTorch CUDA install fails** → normal on CPU-only machines, script falls back to CPU automatically
- **pip timeout** → retry, or check network connection

### 4. Verify

Run a quick test to confirm everything works:

```bash
~/.claude/skills/mivolo_skill/.venv/bin/python -c "
import torch
from transformers import AutoConfig
print(f'PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
config = AutoConfig.from_pretrained('iitolstykh/mivolo_v2', trust_remote_code=True)
print('MiVOLO config loaded successfully')
print()
print('Setup complete! Try: Determine the age and gender of people in photo.jpg')
"
```

If this fails, check the install logs above for errors.

### 5. Done

Tell the user:

> MiVOLO skill is installed and ready. You can now use it from any Claude Code session:
>
> *"Determine the age and gender of people in this image: photo.jpg"*
>
> Models (~500MB) will download automatically on first real inference run.
