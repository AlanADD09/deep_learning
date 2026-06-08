# ============================================================
# AlexNet para clasificación binaria de cielos satelitales
# Dataset: LSCIDMR (Himawari-8)
# Tesis: Predicción de picos de contaminantes
# ============================================================

import os
import sys
import gc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Agregar directorio raíz al path para importar config
sys.path.insert(0, str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import (
    SEED, EPOCHS, DROPOUT, EXPERIMENTS,
    DATASET_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR,
    ALEXNET_MODELS, GRADCAM_DIR, PLOTS_DIR,
)

# ============================================================
# CONFIGURACIÓN GPU (automática) - OPTIMIZADA PARA WSL2
# ============================================================

def setup_gpu(memory_limit_mb: int = None):
    """
    Configura GPU si está disponible.

    Args:
        memory_limit_mb: Límite de memoria en MB (None = memory growth dinámico)
                        Para GTX 1650 4GB se recomienda ~3500MB
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                if memory_limit_mb:
                    tf.config.set_logical_device_configuration(
                        gpu,
                        [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit_mb)]
                    )
                    print(f"✓ GPU detectada: {gpus[0].name}")
                    print(f"  Memoria limitada a: {memory_limit_mb}MB")
                else:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    print(f"✓ GPU detectada: {gpus[0].name}")
                    print(f"  Memoria: se asignará dinámicamente")
            return True
        except RuntimeError as e:
            print(f"[!] Error configurando GPU: {e}")
            return False
    else:
        print("⚠ No se detectó GPU. Usando CPU (será más lento).")
        print("  Para habilitar GPU: pip install tensorflow[and-cuda]")
        return False


GPU_AVAILABLE = None

def init_gpu(memory_limit_mb: int = 3000):
    """Inicializa GPU con el límite especificado."""
    global GPU_AVAILABLE
    GPU_AVAILABLE = setup_gpu(memory_limit_mb=memory_limit_mb)


# ============================================================
# 1. CONFIGURACIÓN GENERAL
# ============================================================

tf.random.set_seed(SEED)
np.random.seed(SEED)

# AlexNet usa 224×224
IMG_SIZE = (224, 224)

# ============================================================
# 2. DATA PIPELINE (tf.data optimizado para GPU)
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE


def build_augmentation():
    """Capa de augmentation que corre en GPU."""
    return keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(20 / 360),
        layers.RandomZoom(0.15),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomBrightness(0.2),
    ], name="data_augmentation")


def build_datasets(batch_size: int):
    """Construye pipelines tf.data optimizados con prefetch paralelo.

    Usa image_dataset_from_directory + prefetch para mantener el GPU
    alimentado sin esperar I/O de disco.
    """
    train_ds = keras.utils.image_dataset_from_directory(
        str(TRAIN_DIR),
        image_size=IMG_SIZE,
        batch_size=batch_size,
        label_mode="binary",
        seed=SEED,
        shuffle=True,
    )

    val_ds = keras.utils.image_dataset_from_directory(
        str(VAL_DIR),
        image_size=IMG_SIZE,
        batch_size=batch_size,
        label_mode="binary",
        seed=SEED,
        shuffle=False,
    )

    test_ds = keras.utils.image_dataset_from_directory(
        str(TEST_DIR),
        image_size=IMG_SIZE,
        batch_size=batch_size,
        label_mode="binary",
        seed=SEED,
        shuffle=False,
    )

    # Normalización (0-255 → 0-1)
    rescale = layers.Rescaling(1.0 / 255)

    # Augmentation para train
    augment = build_augmentation()

    # IMPORTANTE: augmentar sobre los píxeles crudos [0,255] y DESPUÉS rescalar.
    # RandomBrightness usa value_range=(0,255) por defecto; si se aplica sobre
    # imágenes ya normalizadas a [0,1] corrompe la entrada (valores 0 o >>1),
    # creando un desajuste train/test que descalibra las capas BatchNorm.
    train_ds = train_ds.map(
        lambda x, y: (rescale(augment(x, training=True)), y),
        num_parallel_calls=AUTOTUNE,
    ).prefetch(AUTOTUNE)

    val_ds = val_ds.map(
        lambda x, y: (rescale(x), y),
        num_parallel_calls=AUTOTUNE,
    ).prefetch(AUTOTUNE)

    test_ds = test_ds.map(
        lambda x, y: (rescale(x), y),
        num_parallel_calls=AUTOTUNE,
    ).prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds


# ============================================================
# 3. ARQUITECTURA AlexNet MODIFICADA
# ============================================================

def build_alexnet(input_shape=(224, 224, 3), dropout_rate=0.5) -> Model:
    """
    AlexNet adaptado para:
    - Entrada RGB de imágenes satelitales
    - Clasificación binaria (sigmoide en salida)
    - BatchNormalization en lugar de LRN
    - Dropout en capas densas

    Arquitectura original (Krizhevsky et al., 2012):
    - Conv1: 96 filtros 11×11, stride 4, ReLU, MaxPool 3×3 stride 2
    - Conv2: 256 filtros 5×5, pad same, ReLU, MaxPool 3×3 stride 2
    - Conv3: 384 filtros 3×3, pad same, ReLU
    - Conv4: 384 filtros 3×3, pad same, ReLU
    - Conv5: 256 filtros 3×3, pad same, ReLU, MaxPool 3×3 stride 2
    - FC6: 4096 neuronas, ReLU, Dropout
    - FC7: 4096 neuronas, ReLU, Dropout
    - FC8: salida (1 en nuestro caso binario)
    """
    inputs = keras.Input(shape=input_shape, name="input_layer")

    # --- Bloque 1: Conv1 + Pool1 ---
    x = layers.Conv2D(
        96, kernel_size=11, strides=4, padding="valid",
        activation="relu", name="conv1"
    )(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2, name="pool1")(x)

    # --- Bloque 2: Conv2 + Pool2 ---
    x = layers.Conv2D(
        256, kernel_size=5, padding="same",
        activation="relu", name="conv2"
    )(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2, name="pool2")(x)

    # --- Bloque 3: Conv3 ---
    x = layers.Conv2D(
        384, kernel_size=3, padding="same",
        activation="relu", name="conv3"
    )(x)
    x = layers.BatchNormalization(name="bn3")(x)

    # --- Bloque 4: Conv4 ---
    x = layers.Conv2D(
        384, kernel_size=3, padding="same",
        activation="relu", name="conv4"
    )(x)
    x = layers.BatchNormalization(name="bn4")(x)

    # --- Bloque 5: Conv5 + Pool5 ---
    x = layers.Conv2D(
        256, kernel_size=3, padding="same",
        activation="relu", name="conv5"
    )(x)
    x = layers.BatchNormalization(name="bn5")(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2, name="pool5")(x)

    # --- Flatten ---
    x = layers.Flatten(name="flatten")(x)

    # --- FC6 ---
    x = layers.Dense(4096, activation="relu", name="fc6")(x)
    x = layers.Dropout(dropout_rate, name="dropout_fc6")(x)

    # --- FC7 ---
    x = layers.Dense(4096, activation="relu", name="fc7")(x)
    x = layers.Dropout(dropout_rate, name="dropout_fc7")(x)

    # --- FC8: Output binario ---
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = Model(inputs, outputs, name="AlexNet_Binary")
    return model


def build_alexnet_light(input_shape=(224, 224, 3), dropout_rate=0.5) -> Model:
    """
    Versión ligera de AlexNet para datasets pequeños.
    Reduce filtros y neuronas para evitar overfitting.
    Recomendada si el dataset tiene menos de 10,000 imágenes.
    """
    inputs = keras.Input(shape=input_shape, name="input_layer")

    x = layers.Conv2D(48, kernel_size=11, strides=4, padding="valid",
                      activation="relu", name="conv1")(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2, name="pool1")(x)

    x = layers.Conv2D(128, kernel_size=5, padding="same",
                      activation="relu", name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2, name="pool2")(x)

    x = layers.Conv2D(192, kernel_size=3, padding="same",
                      activation="relu", name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)

    x = layers.Conv2D(192, kernel_size=3, padding="same",
                      activation="relu", name="conv4")(x)
    x = layers.BatchNormalization(name="bn4")(x)

    x = layers.Conv2D(128, kernel_size=3, padding="same",
                      activation="relu", name="conv5")(x)
    x = layers.BatchNormalization(name="bn5")(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2, name="pool5")(x)

    x = layers.Flatten(name="flatten")(x)

    x = layers.Dense(2048, activation="relu", name="fc6")(x)
    x = layers.Dropout(dropout_rate, name="dropout_fc6")(x)

    x = layers.Dense(2048, activation="relu", name="fc7")(x)
    x = layers.Dropout(dropout_rate, name="dropout_fc7")(x)

    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = Model(inputs, outputs, name="AlexNet_Light_Binary")
    return model


# ============================================================
# 4. CALLBACKS
# ============================================================

class GPUMemoryCallback(keras.callbacks.Callback):
    """Limpia caché de GPU entre epochs para estabilidad en WSL2."""
    def __init__(self, clear_every_n_epochs=2):
        super().__init__()
        self.clear_every_n_epochs = clear_every_n_epochs

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.clear_every_n_epochs == 0:
            gc.collect()


class GradCAMProgressCallback(keras.callbacks.Callback):
    """
    Genera Grad-CAMs en épocas específicas para visualizar
    cómo evoluciona el aprendizaje del modelo.

    Genera n_samples imágenes por clase en las épocas indicadas.
    Los resultados se guardan en subcarpetas epoch_XX/.
    """
    def __init__(self, experiment_name: str, sample_images: list = None,
                 every_n_epochs: int = 5, n_samples: int = 3,
                 last_conv: str = "conv5"):
        super().__init__()
        self.experiment_name = experiment_name
        self.every_n_epochs = every_n_epochs
        self.n_samples = n_samples
        self.last_conv = last_conv
        self._sample_images = sample_images

    def _get_sample_images(self):
        """Selecciona imágenes fijas para comparar entre épocas."""
        if self._sample_images is not None:
            return self._sample_images

        import random
        from pathlib import Path
        random.seed(SEED)

        images = []
        for class_name in ["contaminado", "no_contaminado"]:
            class_dir = TEST_DIR / class_name
            if not class_dir.exists():
                continue
            extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
            available = [
                str(class_dir / f) for f in os.listdir(class_dir)
                if Path(f).suffix.lower() in extensions
            ]
            if available:
                images.extend(random.sample(available, min(self.n_samples, len(available))))

        self._sample_images = images
        return images

    def on_epoch_end(self, epoch, logs=None):
        # Generar en época 1, y luego cada every_n_epochs
        if epoch == 0 or (epoch + 1) % self.every_n_epochs == 0:
            epoch_dir = str(GRADCAM_DIR / f"{self.experiment_name}" / f"epoch_{epoch+1:02d}")
            os.makedirs(epoch_dir, exist_ok=True)

            samples = self._get_sample_images()
            for img_path in samples:
                try:
                    display_gradcam(
                        img_path, self.model,
                        last_conv=self.last_conv,
                        save_dir=epoch_dir,
                        show=False,
                    )
                except Exception:
                    pass


def build_callbacks(experiment_name: str, use_gpu_memory_callback: bool = True,
                    gradcam_every: int = None, gradcam_samples: int = 3):
    """Early stopping + reducción de LR + guardado del mejor modelo."""
    os.makedirs(ALEXNET_MODELS, exist_ok=True)

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=7,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(ALEXNET_MODELS / f"best_{experiment_name}.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        ),
    ]

    if use_gpu_memory_callback and GPU_AVAILABLE:
        callbacks.append(GPUMemoryCallback(clear_every_n_epochs=2))

    if gradcam_every:
        callbacks.append(GradCAMProgressCallback(
            experiment_name=experiment_name,
            every_n_epochs=gradcam_every,
            n_samples=gradcam_samples,
        ))

    return callbacks


# ============================================================
# 5. ENTRENAMIENTO Y EXPERIMENTACIÓN
# ============================================================

def run_experiment(config: dict, use_light_model: bool = False,
                   gradcam_every: int = None, gradcam_samples: int = 3) -> dict:
    """Ejecuta un experimento completo con la configuración dada."""
    model_type = "AlexNet-Light" if use_light_model else "AlexNet"
    print(f"\n{'='*60}")
    print(f"  Experimento: {config['name']} ({model_type})")
    print(f"  LR={config['lr']} | Batch={config['batch_size']}")
    print(f"{'='*60}")

    # Generadores
    train_ds, val_ds, test_ds = build_datasets(config["batch_size"])

    # Modelo
    if use_light_model:
        model = build_alexnet_light(
            input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), dropout_rate=DROPOUT
        )
    else:
        model = build_alexnet(
            input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), dropout_rate=DROPOUT
        )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config["lr"]),
        loss="binary_crossentropy",
        metrics=["accuracy",
                 keras.metrics.AUC(name="auc"),
                 keras.metrics.Precision(name="precision"),
                 keras.metrics.Recall(name="recall")],
    )

    if config == EXPERIMENTS[0]:
        model.summary()

    # Entrenamiento
    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=build_callbacks(config["name"],
                                  gradcam_every=gradcam_every,
                                  gradcam_samples=gradcam_samples),
        verbose=1,
    )

    # Evaluación en test
    test_loss, test_acc, test_auc, test_prec, test_rec = model.evaluate(
        test_ds, verbose=0
    )
    print(f"\n[TEST] Loss={test_loss:.4f} | Acc={test_acc:.4f} "
          f"| AUC={test_auc:.4f}")

    # Reporte de clasificación
    y_true = np.concatenate([y.numpy() for _, y in test_ds]).flatten()
    y_pred = (model.predict(test_ds) > 0.5).astype(int).flatten()
    print("\n", classification_report(
        y_true, y_pred,
        target_names=["contaminado", "no_contaminado"]
    ))

    return {
        "name": config["name"],
        "model": model,
        "history": history,
        "test_ds": test_ds,
        "test_acc": test_acc,
        "test_auc": test_auc,
    }


# ============================================================
# 6. GRAD-CAM
# ============================================================

def make_gradcam_heatmap(img_array: np.ndarray,
                          model: Model,
                          last_conv_layer_name: str = "conv5") -> np.ndarray:
    """Genera un heatmap Grad-CAM para visualizar qué región activa la clasificación."""
    grad_model = Model(
        inputs=model.input,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output,
        ],
    )

    with tf.GradientTape() as tape:
        inputs = tf.cast(img_array, tf.float32)
        conv_outputs, predictions = grad_model(inputs)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()


def display_gradcam(img_path: str,
                    model: Model,
                    alpha: float = 0.4,
                    last_conv: str = "conv5",
                    save_dir: str = None,
                    show: bool = True) -> dict:
    """Superpone el heatmap Grad-CAM sobre la imagen satelital."""
    from pathlib import Path

    output_dir = save_dir or str(GRADCAM_DIR)
    os.makedirs(output_dir, exist_ok=True)

    img = keras.utils.load_img(img_path, target_size=IMG_SIZE)
    img_array = keras.utils.img_to_array(img) / 255.0
    img_array_exp = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array_exp, verbose=0)[0][0]
    # image_dataset_from_directory ordena las clases alfabéticamente:
    # índice 0 = contaminado, índice 1 = no_contaminado.
    # La sigmoide predice la probabilidad de la clase 1 (no_contaminado).
    label = "NO CONTAMINADO" if pred > 0.5 else "CONTAMINADO"
    label_short = "no_cont" if pred > 0.5 else "cont"
    confidence = pred if pred > 0.5 else 1 - pred

    heatmap = make_gradcam_heatmap(img_array_exp, model, last_conv)
    heatmap_resized = np.array(
        keras.utils.array_to_img(heatmap[..., np.newaxis])
        .resize((IMG_SIZE[1], IMG_SIZE[0]))
    ) / 255.0

    colormap = cm.get_cmap("jet")
    heatmap_colored = colormap(heatmap_resized)[:, :, :3]

    superimposed = heatmap_colored * alpha + img_array * (1 - alpha)
    superimposed = np.clip(superimposed, 0, 1)

    img_name = Path(img_path).stem
    output_filename = f"alexnet_gradcam_{label_short}_{confidence:.0%}_{img_name}.png"
    output_path = os.path.join(output_dir, output_filename)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].imshow(img_array)
    axes[0].set_title("Imagen original\n(Satelital Himawari-8)")
    axes[0].axis("off")

    axes[1].imshow(heatmap_resized, cmap="jet")
    axes[1].set_title("Heatmap Grad-CAM\n(zonas activadas)")
    axes[1].axis("off")

    axes[2].imshow(superimposed)
    axes[2].set_title(f"Superposición\nPredicción: {label} ({confidence:.1%})")
    axes[2].axis("off")

    plt.suptitle("Análisis Grad-CAM — AlexNet Clasificación de Cielo Satelital",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return {
        "image": img_path,
        "prediction": label,
        "confidence": confidence,
        "saved_to": output_path,
    }


def generate_gradcam_batch(model: Model, n_samples: int = 20,
                           last_conv: str = "conv5", save_dir: str = None):
    """Genera visualizaciones Grad-CAM para múltiples imágenes del test set."""
    from pathlib import Path
    import random

    output_dir = save_dir or str(GRADCAM_DIR)
    os.makedirs(output_dir, exist_ok=True)

    results = []

    for class_name in ["contaminado", "no_contaminado"]:
        class_dir = TEST_DIR / class_name

        if not class_dir.exists():
            print(f"[!] Directorio no encontrado: {class_dir}")
            continue

        extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
        images = [
            str(class_dir / f)
            for f in os.listdir(class_dir)
            if Path(f).suffix.lower() in extensions
        ]

        if not images:
            print(f"[!] No se encontraron imágenes en: {class_dir}")
            continue

        sample_size = min(n_samples, len(images))
        sampled = random.sample(images, sample_size)

        print(f"\nProcesando {sample_size} imágenes de '{class_name}'...")

        for img_path in sampled:
            try:
                result = display_gradcam(
                    img_path, model, last_conv=last_conv,
                    save_dir=output_dir, show=False
                )
                results.append(result)
                print(f"  ✓ {Path(img_path).name} → {result['prediction']} ({result['confidence']:.1%})")
            except Exception as e:
                print(f"  [!] Error procesando {img_path}: {e}")

    print(f"\n{'='*50}")
    print(f"Grad-CAM generado para {len(results)} imágenes")
    print(f"Guardado en: {output_dir}")
    print(f"{'='*50}")

    return results


# ============================================================
# 7. VISUALIZACIÓN DE CURVAS DE ENTRENAMIENTO
# ============================================================

def plot_training_curves(results: list):
    """Compara curvas de accuracy y loss de todos los experimentos."""
    os.makedirs(PLOTS_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for r in results:
        hist = r["history"].history
        label = r["name"]
        axes[0].plot(hist["val_accuracy"], label=label)
        axes[1].plot(hist["val_loss"],     label=label)

    axes[0].set_title("Validation Accuracy por experimento")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].set_title("Validation Loss por experimento")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("Loss")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    plt.suptitle("Comparación de experimentos — AlexNet LSCIDMR",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(PLOTS_DIR / "alexnet_training_curves.png"), dpi=150)
    plt.show()


def plot_confusion_matrix(result: dict):
    """Matriz de confusión del mejor experimento."""
    os.makedirs(PLOTS_DIR, exist_ok=True)

    test_ds = result["test_ds"]
    model   = result["model"]

    y_true = np.concatenate([y.numpy() for _, y in test_ds]).flatten()
    y_pred = (model.predict(test_ds) > 0.5).astype(int).flatten()

    cm_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm_matrix, annot=True, fmt="d", cmap="Blues",
        xticklabels=["contaminado", "no_contaminado"],
        yticklabels=["contaminado", "no_contaminado"],
    )
    plt.title(f"Matriz de Confusión — AlexNet\n{result['name']}")
    plt.ylabel("Real")
    plt.xlabel("Predicho")
    plt.tight_layout()
    plt.savefig(str(PLOTS_DIR / f"alexnet_confusion_{result['name']}.png"), dpi=150)
    plt.show()


# ============================================================
# 8. COMPARACIÓN LENET VS ALEXNET
# ============================================================

def compare_architectures_summary():
    """Imprime resumen comparativo de las arquitecturas."""
    print("\n" + "="*70)
    print("COMPARACIÓN DE ARQUITECTURAS: LeNet-5 vs AlexNet")
    print("="*70)

    comparison = """
    | Característica       | LeNet-5              | AlexNet               |
    |---------------------|----------------------|-----------------------|
    | Año                 | 1998                 | 2012                  |
    | Entrada             | 32×32×3              | 224×224×3             |
    | Capas conv          | 3                    | 5                     |
    | Capas FC            | 2                    | 3                     |
    | Total parámetros    | ~60K                 | ~60M (full) / ~15M (light) |
    | Activación          | ReLU (adaptado)      | ReLU                  |
    | Normalización       | -                    | BatchNorm             |
    | Dropout             | 0.5 en FC            | 0.5 en FC             |
    | Pooling             | Average              | Max                   |

    Ventajas AlexNet para imágenes satelitales:
    - Mayor resolución de entrada → captura más detalles atmosféricos
    - Más capas → aprende patrones más complejos de contaminación
    - BatchNorm → entrenamiento más estable

    Desventajas:
    - Mayor costo computacional
    - Requiere más datos para evitar overfitting
    - Mayor tiempo de inferencia
    """
    print(comparison)


# ============================================================
# 9. MAIN
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Entrenar AlexNet para clasificación de contaminación"
    )
    parser.add_argument(
        "--light", action="store_true",
        help="Usar versión ligera de AlexNet (recomendado para datasets pequeños)"
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Mostrar comparación de arquitecturas y salir"
    )
    parser.add_argument(
        "--gradcam", action="store_true",
        help="Generar visualizaciones Grad-CAM después del entrenamiento"
    )
    parser.add_argument(
        "--gradcam-only", type=str, metavar="MODEL_PATH",
        help="Cargar modelo existente y generar solo Grad-CAM (sin entrenar)"
    )
    parser.add_argument(
        "--n-samples", type=int, default=20,
        help="Número de imágenes por clase para Grad-CAM (default: 20)"
    )
    parser.add_argument(
        "--single-experiment", type=int, metavar="INDEX",
        help="Ejecutar solo un experimento específico (0-3). Útil tras crash."
    )
    parser.add_argument(
        "--gpu-memory", type=int, default=12000,
        help="Límite de memoria GPU en MB (default: 12000 para RTX 5060 Ti 16GB)"
    )
    parser.add_argument(
        "--gradcam-progress", type=int, metavar="N", default=None,
        help="Generar Grad-CAMs cada N épocas para visualizar aprendizaje (ej: 5)"
    )
    parser.add_argument(
        "--gradcam-progress-samples", type=int, default=3,
        help="Imágenes por clase para gradcam-progress (default: 3)"
    )
    args = parser.parse_args()

    # Inicializar GPU
    init_gpu(memory_limit_mb=args.gpu_memory)

    if args.compare:
        compare_architectures_summary()
        exit(0)

    # Modo solo Grad-CAM
    if args.gradcam_only:
        if not os.path.exists(args.gradcam_only):
            print(f"[ERROR] Modelo no encontrado: {args.gradcam_only}")
            exit(1)

        print(f"\n{'#'*60}")
        print(f"#  GENERACIÓN GRAD-CAM - ALEXNET")
        print(f"#  Modelo: {args.gradcam_only}")
        print(f"#  Muestras por clase: {args.n_samples}")
        print(f"{'#'*60}")

        loaded_model = keras.models.load_model(args.gradcam_only)
        generate_gradcam_batch(loaded_model, n_samples=args.n_samples)
        exit(0)

    all_results = []

    print(f"\n{'#'*60}")
    print(f"#  ENTRENAMIENTO ALEXNET - CLASIFICACIÓN DE CONTAMINACIÓN")
    print(f"#  Modelo: {'AlexNet-Light' if args.light else 'AlexNet (completo)'}")
    print(f"#  Input size: {IMG_SIZE[0]}×{IMG_SIZE[1]}")
    print(f"#  Memoria GPU limitada a: {args.gpu_memory}MB")
    print(f"{'#'*60}")

    # Seleccionar experimentos a ejecutar
    if args.single_experiment is not None:
        if 0 <= args.single_experiment < len(EXPERIMENTS):
            experiments_to_run = [EXPERIMENTS[args.single_experiment]]
            print(f"\n[!] Ejecutando solo experimento {args.single_experiment}: "
                  f"{experiments_to_run[0]['name']}")
        else:
            print(f"[ERROR] Índice inválido. Use 0-{len(EXPERIMENTS)-1}")
            exit(1)
    else:
        experiments_to_run = EXPERIMENTS

    # Correr experimentos
    for config in experiments_to_run:
        try:
            result = run_experiment(config, use_light_model=args.light,
                                   gradcam_every=args.gradcam_progress,
                                   gradcam_samples=args.gradcam_progress_samples)
            all_results.append(result)
        except Exception as e:
            print(f"\n[ERROR] Experimento {config['name']} falló: {e}")
            print(f"[!] Para reanudar: python3 alexnet.py --gradcam-only "
                  f"{ALEXNET_MODELS}/best_{config['name']}.keras")
            gc.collect()
            tf.keras.backend.clear_session()
            continue

    if not all_results:
        print("\n[ERROR] Ningún experimento completó exitosamente.")
        print("Sugerencias:")
        print("  1. Reducir batch size: editar EXPERIMENTS en config.py")
        print("  2. Usar modelo ligero: python3 alexnet.py --light")
        print("  3. Reducir memoria GPU: python3 alexnet.py --gpu-memory 2500")
        exit(1)

    # Comparar experimentos
    plot_training_curves(all_results)

    # Seleccionar mejor modelo por AUC en test
    best = max(all_results, key=lambda r: r["test_auc"])
    print(f"\n✓ Mejor experimento: {best['name']} "
          f"(AUC={best['test_auc']:.4f})")

    # Matriz de confusión del mejor
    plot_confusion_matrix(best)

    # Generar Grad-CAM si se solicitó
    if args.gradcam:
        print(f"\n{'#'*60}")
        print(f"#  GENERANDO VISUALIZACIONES GRAD-CAM")
        print(f"{'#'*60}")
        generate_gradcam_batch(best["model"], n_samples=args.n_samples)
    else:
        sample_image = str(TEST_DIR / "contaminado" / "sample_001.png")
        if os.path.exists(sample_image):
            display_gradcam(sample_image, best["model"])
        else:
            print(f"\n[!] Usa --gradcam para generar visualizaciones Grad-CAM")

    compare_architectures_summary()
