# calibrate.py
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# SOURCE_DIR = "./lscidmr_raw"
SOURCE_DIR = "./LSCIDMR"

def get_brightness(path):
    arr = np.array(Image.open(path).convert("L"), dtype=np.float32)
    return arr.mean()

paths = []
for root, _, files in os.walk(SOURCE_DIR):
    for f in files:
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif")):
            paths.append(os.path.join(root, f))

print(f"Total imágenes: {len(paths)}")

brightnesses = []
for p in tqdm(paths, desc="Calculando brillo"):
    try:
        brightnesses.append(get_brightness(p))
    except Exception:
        pass

brightnesses = np.array(brightnesses)

print(f"\nBrillo mínimo:  {brightnesses.min():.1f}")
print(f"Brillo máximo:  {brightnesses.max():.1f}")
print(f"Brillo medio:   {brightnesses.mean():.1f}")
print(f"Percentil  25:  {np.percentile(brightnesses, 25):.1f}")
print(f"Percentil  50:  {np.percentile(brightnesses, 50):.1f}")
print(f"Percentil  75:  {np.percentile(brightnesses, 75):.1f}")

# Simular distintos thresholds
print("\nSimulación de thresholds:")
print(f"{'Threshold':>10} {'Contaminado':>12} {'No contaminado':>15} {'Ratio':>8}")
for t in [80, 100, 120, 140, 160, 180, 200, 220]:
    cont = (brightnesses < t).sum()
    no_cont = (brightnesses >= t).sum()
    ratio = cont / len(brightnesses)
    print(f"{t:>10}   {cont:>10}   {no_cont:>13}   {ratio:>7.1%}")

# Histograma
plt.figure(figsize=(10, 4))
plt.hist(brightnesses, bins=60, color="steelblue", alpha=0.7)
for t, color in [(110, "red"), (150, "orange"), (180, "green")]:
    pct = (brightnesses < t).mean()
    plt.axvline(t, color=color, linestyle="--", label=f"t={t} → {pct:.1%} contaminado")
plt.xlabel("Brillo medio")
plt.ylabel("Frecuencia")
plt.title("Distribución de brillo del dataset")
plt.legend()
plt.tight_layout()
plt.savefig("calibration.png", dpi=150)
plt.show()