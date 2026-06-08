# ============================================================
# ResNet-50 / ResNet-101 para clasificación binaria de cielos satelitales
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
    RESNET50_MODELS, RESNET101_MODELS, GRADCAM_DIR, PLOTS_DIR,
)

# ============================================================
# CONFIGURACIÓN GPU (automática)
# ============================================================

def setup_gpu(memory_limit_mb: int = None):
    """Configura GPU si está disponible."""
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

# ResNet usa 224×224
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
    """Construye pipelines tf.data optimizados con prefetch paralelo."""
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

    rescale = layers.Rescaling(1.0 / 255)
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
# 3. BLOQUES RESIDUALES
# ============================================================

def identity_block(x, filters, stage, block):
    """
    Bloque identidad (sin cambio de dimensión).
    Usado cuando input y output tienen la misma forma.

    Estructura: Conv1×1 → BN → ReLU → Conv3×3 → BN → ReLU → Conv1×1 → BN → Add → ReLU
    """
    f1, f2, f3 = filters
    prefix = f"stage{stage}_block{block}"

    shortcut = x

    # Primera capa 1×1
    x = layers.Conv2D(f1, 1, padding="valid", kernel_initializer="he_normal", name=f"{prefix}_conv1")(x)
    x = layers.BatchNormalization(name=f"{prefix}_bn1")(x)
    x = layers.Activation("relu", name=f"{prefix}_relu1")(x)

    # Segunda capa 3×3
    x = layers.Conv2D(f2, 3, padding="same", kernel_initializer="he_normal", name=f"{prefix}_conv2")(x)
    x = layers.BatchNormalization(name=f"{prefix}_bn2")(x)
    x = layers.Activation("relu", name=f"{prefix}_relu2")(x)

    # Tercera capa 1×1
    x = layers.Conv2D(f3, 1, padding="valid", kernel_initializer="he_normal", name=f"{prefix}_conv3")(x)
    x = layers.BatchNormalization(name=f"{prefix}_bn3")(x)

    # Conexión residual
    x = layers.Add(name=f"{prefix}_add")([x, shortcut])
    x = layers.Activation("relu", name=f"{prefix}_relu_out")(x)

    return x


def conv_block(x, filters, stage, block, strides=2):
    """
    Bloque convolucional (con cambio de dimensión).
    Usado cuando hay downsampling o cambio de profundidad.

    Estructura: Conv1×1(s) → BN → ReLU → Conv3×3 → BN → ReLU → Conv1×1 → BN → Add → ReLU
    El shortcut pasa por Conv1×1(s) para igualar dimensiones.
    """
    f1, f2, f3 = filters
    prefix = f"stage{stage}_block{block}"

    shortcut = x

    # Primera capa 1×1 con stride
    x = layers.Conv2D(f1, 1, strides=strides, padding="valid", kernel_initializer="he_normal", name=f"{prefix}_conv1")(x)
    x = layers.BatchNormalization(name=f"{prefix}_bn1")(x)
    x = layers.Activation("relu", name=f"{prefix}_relu1")(x)

    # Segunda capa 3×3
    x = layers.Conv2D(f2, 3, padding="same", kernel_initializer="he_normal", name=f"{prefix}_conv2")(x)
    x = layers.BatchNormalization(name=f"{prefix}_bn2")(x)
    x = layers.Activation("relu", name=f"{prefix}_relu2")(x)

    # Tercera capa 1×1
    x = layers.Conv2D(f3, 1, padding="valid", kernel_initializer="he_normal", name=f"{prefix}_conv3")(x)
    x = layers.BatchNormalization(name=f"{prefix}_bn3")(x)

    # Shortcut con Conv1×1 para igualar dimensiones
    shortcut = layers.Conv2D(f3, 1, strides=strides, padding="valid", kernel_initializer="he_normal",
                             name=f"{prefix}_shortcut_conv")(shortcut)
    shortcut = layers.BatchNormalization(name=f"{prefix}_shortcut_bn")(shortcut)

    # Conexión residual
    x = layers.Add(name=f"{prefix}_add")([x, shortcut])
    x = layers.Activation("relu", name=f"{prefix}_relu_out")(x)

    return x


# ============================================================
# 4. ARQUITECTURA ResNet-50
# ============================================================

def build_resnet50(input_shape=(224, 224, 3), dropout_rate=0.5) -> Model:
    """
    ResNet-50 adaptado para clasificación binaria.

    Arquitectura original (He et al., 2015):
    - Conv 7×7, stride 2 + MaxPool 3×3, stride 2
    - Stage 2: 3 bloques bottleneck [64, 64, 256]
    - Stage 3: 4 bloques bottleneck [128, 128, 512]
    - Stage 4: 6 bloques bottleneck [256, 256, 1024]
    - Stage 5: 3 bloques bottleneck [512, 512, 2048]
    - GlobalAveragePooling + FC(1)
    - ~25.6M parámetros (mucho menor que VGG gracias a bottlenecks)
    """
    inputs = keras.Input(shape=input_shape, name="input_layer")

    # --- Stage 1: Conv + Pool inicial ---
    x = layers.Conv2D(64, 7, strides=2, padding="same", kernel_initializer="he_normal", name="conv1")(inputs)
    x = layers.BatchNormalization(name="bn_conv1")(x)
    x = layers.Activation("relu", name="relu_conv1")(x)
    x = layers.MaxPooling2D(3, strides=2, padding="same", name="pool1")(x)

    # --- Stage 2: 3 bloques [64, 64, 256] ---
    x = conv_block(x, [64, 64, 256], stage=2, block=1, strides=1)
    x = identity_block(x, [64, 64, 256], stage=2, block=2)
    x = identity_block(x, [64, 64, 256], stage=2, block=3)

    # --- Stage 3: 4 bloques [128, 128, 512] ---
    x = conv_block(x, [128, 128, 512], stage=3, block=1, strides=2)
    x = identity_block(x, [128, 128, 512], stage=3, block=2)
    x = identity_block(x, [128, 128, 512], stage=3, block=3)
    x = identity_block(x, [128, 128, 512], stage=3, block=4)

    # --- Stage 4: 6 bloques [256, 256, 1024] ---
    x = conv_block(x, [256, 256, 1024], stage=4, block=1, strides=2)
    x = identity_block(x, [256, 256, 1024], stage=4, block=2)
    x = identity_block(x, [256, 256, 1024], stage=4, block=3)
    x = identity_block(x, [256, 256, 1024], stage=4, block=4)
    x = identity_block(x, [256, 256, 1024], stage=4, block=5)
    x = identity_block(x, [256, 256, 1024], stage=4, block=6)

    # --- Stage 5: 3 bloques [512, 512, 2048] ---
    x = conv_block(x, [512, 512, 2048], stage=5, block=1, strides=2)
    x = identity_block(x, [512, 512, 2048], stage=5, block=2)
    x = identity_block(x, [512, 512, 2048], stage=5, block=3)

    # --- Global Average Pooling + Clasificador ---
    x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
    x = layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = Model(inputs, outputs, name="ResNet50_Binary")
    return model


# ============================================================
# 5. ARQUITECTURA ResNet-101
# ============================================================

def build_resnet101(input_shape=(224, 224, 3), dropout_rate=0.5) -> Model:
    """
    ResNet-101 adaptado para clasificación binaria.

    Diferencia con ResNet-50: Stage 4 tiene 23 bloques en vez de 6.
    Total: ~44.5M parámetros.

    - Stage 2: 3 bloques [64, 64, 256]
    - Stage 3: 4 bloques [128, 128, 512]
    - Stage 4: 23 bloques [256, 256, 1024]
    - Stage 5: 3 bloques [512, 512, 2048]
    """
    inputs = keras.Input(shape=input_shape, name="input_layer")

    # --- Stage 1: Conv + Pool inicial ---
    x = layers.Conv2D(64, 7, strides=2, padding="same", kernel_initializer="he_normal", name="conv1")(inputs)
    x = layers.BatchNormalization(name="bn_conv1")(x)
    x = layers.Activation("relu", name="relu_conv1")(x)
    x = layers.MaxPooling2D(3, strides=2, padding="same", name="pool1")(x)

    # --- Stage 2: 3 bloques [64, 64, 256] ---
    x = conv_block(x, [64, 64, 256], stage=2, block=1, strides=1)
    x = identity_block(x, [64, 64, 256], stage=2, block=2)
    x = identity_block(x, [64, 64, 256], stage=2, block=3)

    # --- Stage 3: 4 bloques [128, 128, 512] ---
    x = conv_block(x, [128, 128, 512], stage=3, block=1, strides=2)
    x = identity_block(x, [128, 128, 512], stage=3, block=2)
    x = identity_block(x, [128, 128, 512], stage=3, block=3)
    x = identity_block(x, [128, 128, 512], stage=3, block=4)

    # --- Stage 4: 23 bloques [256, 256, 1024] ---
    x = conv_block(x, [256, 256, 1024], stage=4, block=1, strides=2)
    for i in range(2, 24):
        x = identity_block(x, [256, 256, 1024], stage=4, block=i)

    # --- Stage 5: 3 bloques [512, 512, 2048] ---
    x = conv_block(x, [512, 512, 2048], stage=5, block=1, strides=2)
    x = identity_block(x, [512, 512, 2048], stage=5, block=2)
    x = identity_block(x, [512, 512, 2048], stage=5, block=3)

    # --- Global Average Pooling + Clasificador ---
    x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
    x = layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = Model(inputs, outputs, name="ResNet101_Binary")
    return model


# ============================================================
# 6. CALLBACKS
# ============================================================

class GPUMemoryCallback(keras.callbacks.Callback):
    """Limpia caché de GPU entre epochs para estabilidad."""
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
    """
    def __init__(self, experiment_name: str, model_name: str = "resnet50",
                 sample_images: list = None,
                 every_n_epochs: int = 5, n_samples: int = 3,
                 last_conv: str = "stage5_block3_relu_out"):
        super().__init__()
        self.experiment_name = experiment_name
        self.model_name = model_name
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
        if epoch == 0 or (epoch + 1) % self.every_n_epochs == 0:
            epoch_dir = str(GRADCAM_DIR / f"{self.model_name}_{self.experiment_name}" / f"epoch_{epoch+1:02d}")
            os.makedirs(epoch_dir, exist_ok=True)

            samples = self._get_sample_images()
            for img_path in samples:
                try:
                    display_gradcam(
                        img_path, self.model,
                        last_conv=self.last_conv,
                        model_name=self.model_name,
                        save_dir=epoch_dir,
                        show=False,
                    )
                except Exception:
                    pass


def lr_warmup_schedule(target_lr, warmup_epochs=5):
    """Devuelve función de schedule con warmup lineal."""
    def schedule(epoch, lr):
        if epoch < warmup_epochs:
            return target_lr * (epoch + 1) / warmup_epochs
        return lr  # después del warmup, ReduceLROnPlateau controla
    return schedule


def build_callbacks(experiment_name: str, models_dir, use_gpu_memory_callback: bool = True,
                    gradcam_every: int = None, gradcam_samples: int = 3,
                    model_name: str = "resnet50", last_conv: str = "stage5_block3_relu_out",
                    target_lr: float = 1e-3, warmup_epochs: int = 5):
    """Early stopping + warmup + reducción de LR + guardado del mejor modelo."""
    os.makedirs(models_dir, exist_ok=True)

    callbacks = [
        keras.callbacks.LearningRateScheduler(
            lr_warmup_schedule(target_lr, warmup_epochs),
            verbose=1,
        ),
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
            filepath=str(models_dir / f"best_{experiment_name}.keras"),
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
            model_name=model_name,
            every_n_epochs=gradcam_every,
            n_samples=gradcam_samples,
            last_conv=last_conv,
        ))

    return callbacks


# ============================================================
# 7. ENTRENAMIENTO Y EXPERIMENTACIÓN
# ============================================================

def run_experiment(config: dict, use_resnet101: bool = False,
                   gradcam_every: int = None, gradcam_samples: int = 3) -> dict:
    """Ejecuta un experimento completo con la configuración dada."""
    model_name = "ResNet-101" if use_resnet101 else "ResNet-50"
    models_dir = RESNET101_MODELS if use_resnet101 else RESNET50_MODELS
    last_conv = "stage5_block3_relu_out"

    print(f"\n{'='*60}")
    print(f"  Experimento: {config['name']} ({model_name})")
    print(f"  LR={config['lr']} | Batch={config['batch_size']}")
    print(f"{'='*60}")

    # Datasets
    train_ds, val_ds, test_ds = build_datasets(config["batch_size"])

    # Modelo
    if use_resnet101:
        model = build_resnet101(
            input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), dropout_rate=DROPOUT
        )
    else:
        model = build_resnet50(
            input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), dropout_rate=DROPOUT
        )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config["lr"], clipnorm=1.0),
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
        callbacks=build_callbacks(
            config["name"], models_dir,
            gradcam_every=gradcam_every,
            gradcam_samples=gradcam_samples,
            model_name=model_name.lower().replace("-", ""),
            last_conv=last_conv,
            target_lr=config["lr"],
        ),
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
        "model_name": model_name,
        "model": model,
        "history": history,
        "test_ds": test_ds,
        "test_acc": test_acc,
        "test_auc": test_auc,
        "last_conv": last_conv,
    }


# ============================================================
# 8. GRAD-CAM
# ============================================================

def make_gradcam_heatmap(img_array: np.ndarray,
                          model: Model,
                          last_conv_layer_name: str = "stage5_block3_relu_out") -> np.ndarray:
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
                    last_conv: str = "stage5_block3_relu_out",
                    model_name: str = "resnet",
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
    output_filename = f"{model_name}_gradcam_{label_short}_{confidence:.0%}_{img_name}.png"
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

    plt.suptitle(f"Análisis Grad-CAM — {model_name.upper()} Clasificación de Cielo Satelital",
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
                           last_conv: str = "stage5_block3_relu_out",
                           model_name: str = "resnet",
                           save_dir: str = None):
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
                    model_name=model_name,
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
# 9. VISUALIZACIÓN DE CURVAS DE ENTRENAMIENTO
# ============================================================

def plot_training_curves(results: list, model_name: str = "ResNet"):
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

    plt.suptitle(f"Comparación de experimentos — {model_name} LSCIDMR",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(PLOTS_DIR / f"{model_name.lower().replace('-','')}_training_curves.png"), dpi=150)
    plt.show()


def plot_confusion_matrix(result: dict):
    """Matriz de confusión del mejor experimento."""
    os.makedirs(PLOTS_DIR, exist_ok=True)

    test_ds = result["test_ds"]
    model   = result["model"]
    model_name = result["model_name"]

    y_true = np.concatenate([y.numpy() for _, y in test_ds]).flatten()
    y_pred = (model.predict(test_ds) > 0.5).astype(int).flatten()

    cm_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm_matrix, annot=True, fmt="d", cmap="Blues",
        xticklabels=["contaminado", "no_contaminado"],
        yticklabels=["contaminado", "no_contaminado"],
    )
    plt.title(f"Matriz de Confusión — {model_name}\n{result['name']}")
    plt.ylabel("Real")
    plt.xlabel("Predicho")
    plt.tight_layout()
    plt.savefig(str(PLOTS_DIR / f"{model_name.lower().replace('-','')}_confusion_{result['name']}.png"), dpi=150)
    plt.show()


# ============================================================
# 10. MAIN
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Entrenar ResNet-50/ResNet-101 para clasificación de contaminación"
    )
    parser.add_argument(
        "--resnet101", action="store_true",
        help="Usar ResNet-101 en lugar de ResNet-50 (default: ResNet-50)"
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

    model_name = "ResNet-101" if args.resnet101 else "ResNet-50"
    models_dir = RESNET101_MODELS if args.resnet101 else RESNET50_MODELS
    last_conv = "stage5_block3_relu_out"

    # Modo solo Grad-CAM
    if args.gradcam_only:
        if not os.path.exists(args.gradcam_only):
            print(f"[ERROR] Modelo no encontrado: {args.gradcam_only}")
            exit(1)

        print(f"\n{'#'*60}")
        print(f"#  GENERACIÓN GRAD-CAM - {model_name}")
        print(f"#  Modelo: {args.gradcam_only}")
        print(f"#  Muestras por clase: {args.n_samples}")
        print(f"{'#'*60}")

        loaded_model = keras.models.load_model(args.gradcam_only)
        generate_gradcam_batch(loaded_model, n_samples=args.n_samples,
                               last_conv=last_conv,
                               model_name=model_name.lower().replace("-", ""))
        exit(0)

    all_results = []

    print(f"\n{'#'*60}")
    print(f"#  ENTRENAMIENTO {model_name} - CLASIFICACIÓN DE CONTAMINACIÓN")
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
            result = run_experiment(config, use_resnet101=args.resnet101,
                                   gradcam_every=args.gradcam_progress,
                                   gradcam_samples=args.gradcam_progress_samples)
            all_results.append(result)
        except Exception as e:
            print(f"\n[ERROR] Experimento {config['name']} falló: {e}")
            print(f"[!] Para reanudar: python3 src/resnet.py --gradcam-only "
                  f"{models_dir}/best_{config['name']}.keras"
                  f"{' --resnet101' if args.resnet101 else ''}")
            gc.collect()
            tf.keras.backend.clear_session()
            continue

    if not all_results:
        print("\n[ERROR] Ningún experimento completó exitosamente.")
        print("Sugerencias:")
        print("  1. Reducir batch size: editar EXPERIMENTS en config.py")
        print(f"  2. Reducir memoria GPU: python3 src/resnet.py --gpu-memory 2500")
        exit(1)

    # Comparar experimentos
    plot_training_curves(all_results, model_name=model_name)

    # Seleccionar mejor modelo por AUC en test
    best = max(all_results, key=lambda r: r["test_auc"])
    print(f"\n✓ Mejor experimento: {best['name']} "
          f"(AUC={best['test_auc']:.4f})")

    # Matriz de confusión del mejor
    plot_confusion_matrix(best)

    # Generar Grad-CAM si se solicitó
    if args.gradcam:
        print(f"\n{'#'*60}")
        print(f"#  GENERANDO VISUALIZACIONES GRAD-CAM — {model_name}")
        print(f"{'#'*60}")
        generate_gradcam_batch(best["model"], n_samples=args.n_samples,
                               last_conv=best["last_conv"],
                               model_name=model_name.lower().replace("-", ""))
    else:
        print(f"\n[!] Usa --gradcam para generar visualizaciones Grad-CAM")
