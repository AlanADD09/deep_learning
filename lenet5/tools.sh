#!/bin/bash
# ============================================================
# setup.sh — Crea venv e instala dependencias del proyecto
# ============================================================

set -e  # Detiene el script si cualquier comando falla

VENV_DIR=".venv"
PYTHON="python3"

echo "============================================================"
echo "  Setup del entorno virtual"
echo "============================================================"

# Verificar que Python 3 esté instalado
if ! command -v $PYTHON &> /dev/null; then
    echo "[ERROR] Python 3 no encontrado. Instálalo primero."
    exit 1
fi

PYTHON_VERSION=$($PYTHON --version 2>&1)
echo "  Python encontrado: $PYTHON_VERSION"

# Crear el venv si no existe
if [ -d "$VENV_DIR" ]; then
    echo "  [!] El venv ya existe en '$VENV_DIR', omitiendo creación."
else
    echo "  Creando entorno virtual en '$VENV_DIR'..."
    $PYTHON -m venv $VENV_DIR
    echo "  ✓ Venv creado."
fi

# Activar el venv
echo "  Activando entorno virtual..."
source $VENV_DIR/bin/activate

# Actualizar pip
echo "  Actualizando pip..."
pip install --upgrade pip --quiet

echo ""
echo "  Instalando dependencias..."
echo "------------------------------------------------------------"

# Dependencias del proyecto
pip install \
    tensorflow==2.16.1 \
    numpy==1.26.4 \
    matplotlib==3.9.0 \
    scikit-learn==1.5.0 \
    seaborn==0.13.2 \
    Pillow==10.3.0 \
    tqdm==4.66.4

echo ""
echo "============================================================"
echo "  ✓ Instalación completa."
echo ""
echo "  Para activar el entorno en tu terminal:"
echo "    source $VENV_DIR/bin/activate"
echo ""
echo "  Para desactivarlo:"
echo "    deactivate"
echo "============================================================"