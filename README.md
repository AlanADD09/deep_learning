# CNN - Clasificación de Contaminación Satelital

Clasificación binaria de imágenes satelitales (Himawari-8) usando redes
convolucionales. Proyecto de tesis para predicción de picos de contaminantes.

## Estructura del Proyecto

```
deep_learning/
├── config.py                  ← Rutas y parámetros compartidos
├── setup.sh                   ← Instala venv + dependencias
├── data/
│   ├── LSCIDMR/              ← Imágenes crudas del satélite
│   └── dataset/              ← Dataset preparado (train/val/test)
├── src/
│   ├── alexnet.py            ← Entrena AlexNet
│   ├── lenet5.py             ← Entrena LeNet-5
│   ├── dataset_preparation.py← Etiqueta y divide el dataset
│   └── calibrate.py          ← Calibra umbral de brillo
├── models/
│   ├── alexnet/              ← Modelos AlexNet guardados
│   └── lenet5/              ← Modelos LeNet-5 guardados
└── outputs/
    ├── gradcam/              ← Visualizaciones Grad-CAM
    └── plots/                ← Curvas de entrenamiento, matrices
```

## Arquitecturas

| Arquitectura | Entrada    | Parámetros | Uso recomendado              |
|-------------|------------|------------|------------------------------|
| LeNet-5     | 32×32 RGB  | ~60K       | Pruebas rápidas, datasets pequeños |
| AlexNet     | 224×224 RGB| ~60M       | Mayor precisión              |
| AlexNet-Light| 224×224 RGB| ~15M      | Datasets < 10K imágenes      |

## Uso

```bash
# 1. Configurar entorno
bash setup.sh
source .venv/bin/activate

# 2. Colocar imágenes LSCIDMR en data/LSCIDMR/

# 3. Calibrar umbral (opcional)
python src/calibrate.py

# 4. Preparar dataset
python src/dataset_preparation.py

# 5. Entrenar
python src/lenet5.py
python src/alexnet.py
python src/alexnet.py --light      # versión ligera
python src/alexnet.py --gradcam    # con visualizaciones Grad-CAM

# 6. Solo Grad-CAM sobre modelo existente
python src/alexnet.py --gradcam-only models/alexnet/best_lr1e-3_bs32.keras
```

## Dataset

Estructura generada por `dataset_preparation.py`:

```
data/dataset/
├── train/
│   ├── contaminado/
│   └── no_contaminado/
├── val/
│   ├── contaminado/
│   └── no_contaminado/
└── test/
    ├── contaminado/
    └── no_contaminado/
```
