# ============================================================
# dataset_preparation.py
# Etiqueta y divide el dataset LSCIDMR automáticamente
# ============================================================

import os
import sys
import shutil
import numpy as np
from PIL import Image
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# Agregar directorio raíz al path para importar config
sys.path.insert(0, str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import SEED, RAW_DIR, DATASET_DIR

# ============================================================
# CONFIGURACIÓN
# ============================================================

SOURCE_DIR = str(RAW_DIR)
OUTPUT_DIR = str(DATASET_DIR)

SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}

# Umbral de "contaminación" basado en brillo medio
BRIGHTNESS_THRESHOLD = 80  # 0-255


# ============================================================
# 1. FUNCIÓN DE ETIQUETADO
# ============================================================

def compute_features(img_path: str) -> dict:
    """
    Extrae features visuales de una imagen satelital
    para determinar si está contaminada o no.

    Criterios:
    - Brillo medio bajo → contaminado (neblina/smog oscurece el cielo)
    - Alta proporción de píxeles grises → contaminado
    - Baja saturación → contaminado (colores apagados)
    """
    img = Image.open(img_path).convert("RGB")
    arr = np.array(img, dtype=np.float32)

    # Brillo medio
    gray = np.mean(arr, axis=2)
    brightness = gray.mean()

    # Proporción de píxeles "grises"
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    diff_rg = np.abs(r - g)
    diff_gb = np.abs(g - b)
    gray_mask = (diff_rg < 25) & (diff_gb < 25) & (gray < 160)
    gray_ratio = gray_mask.mean()

    # Saturación media
    arr_hsv = np.array(Image.open(img_path).convert("HSV"), dtype=np.float32)
    saturation = arr_hsv[:, :, 1].mean()

    return {
        "brightness": brightness,
        "gray_ratio": gray_ratio,
        "saturation": saturation,
    }


def label_image(img_path: str, threshold: float = BRIGHTNESS_THRESHOLD) -> str:
    """
    Retorna 'contaminado' o 'no_contaminado' según heurística.

    Lógica combinada:
    - Si brillo < threshold: probable contaminación
    - Si gray_ratio > 0.4: refuerza contaminación
    - Si saturación < 30: refuerza contaminación
    """
    features = compute_features(img_path)

    score = 0
    if features["brightness"] < threshold:
        score += 2
    if features["gray_ratio"] > 0.40:
        score += 1
    if features["saturation"] < 30:
        score += 1

    return "contaminado" if score >= 2 else "no_contaminado"


# ============================================================
# 2. RECOPILAR Y ETIQUETAR IMÁGENES
# ============================================================

# Tamaño mínimo en bytes para considerar un archivo como imagen válida
# (filtra archivos vacíos o corruptos sin abrir cada uno)
MIN_FILE_SIZE = 1024  # 1 KB


def collect_images(source_dir: str) -> list[str]:
    """Recopila imágenes válidas (> MIN_FILE_SIZE) del directorio fuente."""
    extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    paths = []
    skipped = 0
    for root, _, files in os.walk(source_dir):
        for f in files:
            if Path(f).suffix.lower() in extensions:
                full_path = os.path.join(root, f)
                if os.path.getsize(full_path) >= MIN_FILE_SIZE:
                    paths.append(full_path)
                else:
                    skipped += 1
    if skipped:
        print(f"  [!] {skipped} archivos omitidos (< {MIN_FILE_SIZE} bytes, probablemente corruptos)")
    return sorted(paths)


def _label_single(path: str) -> tuple[str, str | None]:
    """Worker function for multiprocessing. Returns (path, label) or (path, None) on error."""
    try:
        return (path, label_image(path))
    except Exception:
        return (path, None)


def label_all_images(image_paths: list[str], n_workers: int = None) -> dict[str, list[str]]:
    """
    Etiqueta todas las imágenes en paralelo y retorna un dict:
    { "contaminado": [...paths], "no_contaminado": [...paths] }
    """
    labeled = {"contaminado": [], "no_contaminado": []}
    errors = 0

    if n_workers is None:
        n_workers = min(cpu_count(), 8)

    print(f"Etiquetando {len(image_paths)} imágenes ({n_workers} workers)...")

    with Pool(processes=n_workers) as pool:
        results = pool.imap_unordered(_label_single, image_paths, chunksize=200)
        for path, label in tqdm(results, total=len(image_paths)):
            if label is None:
                errors += 1
            else:
                labeled[label].append(path)

    print(f"\n  Contaminado:    {len(labeled['contaminado'])}")
    print(f"  No contaminado: {len(labeled['no_contaminado'])}")
    if errors:
        print(f"  Errores:        {errors} (omitidos)")

    # Advertencia de desbalance
    total = sum(len(v) for v in labeled.values())
    if total > 0:
        for cls, paths in labeled.items():
            ratio = len(paths) / total
            if ratio < 0.25 or ratio > 0.75:
                print(f"\n  ⚠ Desbalance detectado en '{cls}': {ratio:.1%}")
                print("    Considera ajustar BRIGHTNESS_THRESHOLD")

    return labeled


# ============================================================
# 3. DIVIDIR Y COPIAR AL DIRECTORIO DE SALIDA
# ============================================================

def split_and_copy(
    labeled: dict[str, list[str]],
    output_dir: str,
    splits: dict = SPLITS,
):
    """
    Divide cada clase en train/val/test y copia los archivos.
    Mantiene proporciones dentro de cada clase (stratified).
    """
    for cls, paths in labeled.items():
        if len(paths) == 0:
            print(f"[!] Clase '{cls}' vacía, omitiendo.")
            continue

        train_paths, temp_paths = train_test_split(
            paths,
            test_size=(splits["val"] + splits["test"]),
            random_state=SEED,
        )
        val_ratio_adjusted = splits["val"] / (splits["val"] + splits["test"])
        val_paths, test_paths = train_test_split(
            temp_paths,
            test_size=(1 - val_ratio_adjusted),
            random_state=SEED,
        )

        split_data = {
            "train": train_paths,
            "val": val_paths,
            "test": test_paths,
        }

        for split_name, split_paths in split_data.items():
            dest_dir = os.path.join(output_dir, split_name, cls)
            os.makedirs(dest_dir, exist_ok=True)

            for src in tqdm(split_paths, desc=f"{split_name}/{cls}"):
                fname = Path(src).name
                dest = os.path.join(dest_dir, fname)
                if os.path.exists(dest):
                    stem = Path(src).stem
                    suffix = Path(src).suffix
                    dest = os.path.join(
                        dest_dir, f"{stem}_{hash(src) % 10000}{suffix}"
                    )
                shutil.copy2(src, dest)

    print(f"\n✓ Dataset dividido en: {output_dir}")
    _print_summary(output_dir)


def _print_summary(output_dir: str):
    """Imprime resumen de la estructura generada."""
    for split in ["train", "val", "test"]:
        for cls in ["contaminado", "no_contaminado"]:
            d = os.path.join(output_dir, split, cls)
            if os.path.exists(d):
                n = len(os.listdir(d))
                print(f"  {split:6s}/{cls}: {n} imágenes")


# ============================================================
# 4. CALIBRACIÓN DEL UMBRAL (OPCIONAL)
# ============================================================

def calibrate_threshold(
    source_dir: str,
    n_samples: int = 200,
    thresholds: list[int] = None,
):
    """
    Muestra distribución de brightness para ayudar a elegir
    el umbral más adecuado visualmente.
    """
    import matplotlib.pyplot as plt

    if thresholds is None:
        thresholds = [80, 100, 110, 120, 140]

    paths = collect_images(source_dir)
    sample = np.random.choice(paths, min(n_samples, len(paths)), replace=False)

    brightnesses = []
    for p in tqdm(sample, desc="Calculando brillo"):
        try:
            arr = np.array(Image.open(p).convert("L"), dtype=np.float32)
            brightnesses.append(arr.mean())
        except Exception:
            pass

    brightnesses = np.array(brightnesses)

    plt.figure(figsize=(10, 4))
    plt.hist(brightnesses, bins=50, color="steelblue", alpha=0.7)
    for t in thresholds:
        pct = (brightnesses < t).mean()
        plt.axvline(t, linestyle="--", label=f"t={t} ({pct:.1%} contaminado)")
    plt.xlabel("Brillo medio")
    plt.ylabel("Frecuencia")
    plt.title("Distribución de brillo — ayuda a elegir umbral")
    plt.legend()
    plt.tight_layout()
    plt.savefig("threshold_calibration.png", dpi=150)
    plt.show()

    return brightnesses


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    # 1. Recopilar imágenes
    all_images = collect_images(SOURCE_DIR)
    print(f"Total de imágenes encontradas: {len(all_images)}")

    if len(all_images) == 0:
        print(f"\n[!] No se encontraron imágenes en: {SOURCE_DIR}")
        print(f"    Coloca las imágenes LSCIDMR en: {RAW_DIR}")
        exit(1)

    # 2. Etiquetar
    labeled = label_all_images(all_images)

    # 3. Dividir y copiar
    split_and_copy(labeled, OUTPUT_DIR)
