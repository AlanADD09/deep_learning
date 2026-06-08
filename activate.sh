#!/bin/bash
# ============================================================
# activate.sh — Activa el venv con soporte GPU
# Uso: source activate.sh
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "[ERROR] No se encontró el venv en '$VENV_DIR'"
    echo "  Ejecuta primero: bash setup.sh"
    return 1 2>/dev/null || exit 1
fi

source "$VENV_DIR/bin/activate"

# Configurar LD_LIBRARY_PATH para CUDA (pip-installed NVIDIA libs)
NVIDIA_LIBS="$VENV_DIR/lib/python3.12/site-packages/nvidia"
export LD_LIBRARY_PATH="$NVIDIA_LIBS/cuda_runtime/lib:$NVIDIA_LIBS/cudnn/lib:$NVIDIA_LIBS/cublas/lib:$NVIDIA_LIBS/cufft/lib:$NVIDIA_LIBS/curand/lib:$NVIDIA_LIBS/cusolver/lib:$NVIDIA_LIBS/cusparse/lib:$NVIDIA_LIBS/nccl/lib:$NVIDIA_LIBS/nvjitlink/lib:${LD_LIBRARY_PATH:-}"

# Persistir caché JIT de PTX. La RTX 5060 Ti (Blackwell, compute capability
# sm_120) no tiene kernels precompilados en TensorFlow 2.21, así que se compilan
# desde PTX en la primera ejecución. Ampliar el caché evita recompilar cada vez:
# solo la primera corrida es lenta, las siguientes arrancan rápido.
export CUDA_CACHE_MAXSIZE=4294967296   # 4 GB
export CUDA_CACHE_PATH="$HOME/.nv/ComputeCache"

echo "✓ Entorno activado (Python $(python --version 2>&1 | cut -d' ' -f2))"
echo "  GPU libs en LD_LIBRARY_PATH"
echo "  Caché JIT CUDA: $CUDA_CACHE_PATH (máx 4GB)"
