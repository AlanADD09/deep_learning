# ============================================================
# config.py — Configuración centralizada de rutas del proyecto
# ============================================================

import os
from pathlib import Path

# Raíz del proyecto (directorio donde vive este archivo)
PROJECT_ROOT = Path(__file__).resolve().parent

# --- Datos ---
DATA_DIR     = PROJECT_ROOT / "data"
RAW_DIR      = DATA_DIR / "LSCIDMR"        # Imágenes crudas del satélite
DATASET_DIR  = DATA_DIR / "dataset"         # Dataset preparado (train/val/test)
TRAIN_DIR    = DATASET_DIR / "train"
VAL_DIR      = DATASET_DIR / "val"
TEST_DIR     = DATASET_DIR / "test"

# --- Modelos guardados ---
MODELS_DIR       = PROJECT_ROOT / "models"
LENET5_MODELS    = MODELS_DIR / "lenet5"
ALEXNET_MODELS   = MODELS_DIR / "alexnet"
VGG16_MODELS     = MODELS_DIR / "vgg16"
VGG19_MODELS     = MODELS_DIR / "vgg19"
RESNET50_MODELS  = MODELS_DIR / "resnet50"
RESNET101_MODELS = MODELS_DIR / "resnet101"
INCEPTIONV3_MODELS = MODELS_DIR / "inceptionv3"

# --- Salidas (gráficas, grad-cam, reportes) ---
OUTPUTS_DIR  = PROJECT_ROOT / "outputs"
GRADCAM_DIR  = OUTPUTS_DIR / "gradcam"
PLOTS_DIR    = OUTPUTS_DIR / "plots"

# --- Hiperparámetros compartidos ---
SEED         = 42
EPOCHS       = 30
DROPOUT      = 0.5
NUM_CLASSES  = 1  # Clasificación binaria → sigmoide

# Experimentos comunes
EXPERIMENTS = [
    {"lr": 0.001,  "batch_size": 32, "name": "lr1e-3_bs32"},
    {"lr": 0.001,  "batch_size": 64, "name": "lr1e-3_bs64"},
    {"lr": 0.0001, "batch_size": 32, "name": "lr1e-4_bs32"},
    {"lr": 0.0001, "batch_size": 64, "name": "lr1e-4_bs64"},
]
