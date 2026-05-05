#!/bin/bash
# Install all dependencies for Multi-Level-OT-MTA
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/.venv"

# 1. Virtual environment
if [ ! -f "$VENV/bin/activate" ]; then
    rm -rf "$VENV"
    python3 -m venv "$VENV" 2>/dev/null || {
        python3 -m venv --without-pip "$VENV"
        curl -sS https://bootstrap.pypa.io/get-pip.py | "$VENV/bin/python3"
    }
fi
source "$VENV/bin/activate"
pip install --upgrade pip --quiet

# 2. PyTorch — auto-detect CUDA version
CUDA_MAJOR=$(nvidia-smi 2>/dev/null | grep -oP "CUDA Version: \K[0-9]+" | head -1)
if   [ "${CUDA_MAJOR:-0}" -ge 13 ]; then IDX="cu130"; TORCH="torch==2.9.0"; TVISION="torchvision==0.24.0"
elif [ "${CUDA_MAJOR:-0}" -eq 12 ];  then IDX="cu121"; TORCH="torch"; TVISION="torchvision"
elif [ "${CUDA_MAJOR:-0}" -eq 11 ];  then IDX="cu118"; TORCH="torch"; TVISION="torchvision"
else IDX="cpu"; TORCH="torch"; TVISION="torchvision"; fi
echo "Installing ${TORCH} (CUDA ${CUDA_MAJOR:-?} → ${IDX})..."
pip install "$TORCH" "$TVISION" --index-url "https://download.pytorch.org/whl/${IDX}" --quiet

# 3. Training & evaluation libraries
pip install transformers peft datasets accelerate tqdm wandb numpy \
    rouge-score bert-score evaluate nltk scikit-learn pandas pyarrow --quiet

# 4. spaCy + English model
pip install spacy --quiet
python -m spacy download en_core_web_sm --quiet

echo "Done! Run: bash run.sh"
