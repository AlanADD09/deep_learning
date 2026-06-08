#!/bin/bash
# ============================================================
# setup.sh — Crea venv e instala dependencias del proyecto
# Requiere: Python 3.12, driver NVIDIA instalado
# ============================================================

set -e

VENV_DIR=".venv"
PYTHON="python3.12"

echo "============================================================"
echo "  Setup del entorno virtual (GPU)"
echo "============================================================"

if ! command -v $PYTHON &> /dev/null; then
    echo "[ERROR] Python 3.12 no encontrado."
    echo "  Instálalo con:"
    echo "    sudo add-apt-repository ppa:deadsnakes/ppa"
    echo "    sudo apt install python3.12 python3.12-venv python3.12-dev"
    exit 1
fi

PYTHON_VERSION=$($PYTHON --version 2>&1)
echo "  Python encontrado: $PYTHON_VERSION"

# Verificar GPU
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "desconocida")
    echo "  GPU detectada: $GPU_NAME"
else
    echo "  ⚠ nvidia-smi no encontrado. GPU puede no estar disponible."
fi

if [ -d "$VENV_DIR" ]; then
    echo "  [!] El venv ya existe en '$VENV_DIR', omitiendo creación."
else
    echo "  Creando entorno virtual en '$VENV_DIR'..."
    $PYTHON -m venv $VENV_DIR
    echo "  ✓ Venv creado."
fi

echo "  Activando entorno virtual..."
source $VENV_DIR/bin/activate

echo "  Actualizando pip..."
pip install --upgrade pip --quiet

echo ""
echo "  Instalando TensorFlow con soporte GPU + dependencias..."
echo "------------------------------------------------------------"

pip install \
    "tensorflow[and-cuda]" \
    numpy \
    matplotlib \
    scikit-learn \
    seaborn \
    Pillow \
    tqdm

echo ""
echo "============================================================"
echo "  ✓ Instalación completa."
echo ""
echo "  Para activar el entorno con soporte GPU:"
echo "    source activate.sh"
echo ""
echo "  Para desactivarlo:"
echo "    deactivate"
echo "============================================================"
