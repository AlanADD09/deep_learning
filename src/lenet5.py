# ============================================================
# LeNet-5 para clasificación binaria de cielos satelitales
# Dataset: LSCIDMR (Himawari-8)
# Tesis: Predicción de picos de contaminantes
# ============================================================

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Agregar directorio raíz al path para importar config
sys.path.insert(0, str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import (
    SEED, EPOCHS, DROPOUT, EXPERIMENTS,
    DATASET_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR,
    LENET5_MODELS, GRADCAM_DIR, PLOTS_DIR,
)

# ============================================================
# 1. CONFIGURACIÓN GENERAL
# ============================================================

tf.random.set_seed(SEED)
np.random.seed(SEED)

# LeNet-5 clásico usa 32×32
IMG_SIZE = (32, 32)

# ============================================================
# 2. DATA AUGMENTATION Y GENERADORES
# ============================================================

def build_generators(batch_size: int):
    """Construye generadores con data augmentation para train
    y sin augmentation para val/test."""

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.15,
        brightness_range=[0.8, 1.2],
        fill_mode="reflect",
    )

    val_test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        str(TRAIN_DIR),
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="binary",
        seed=SEED,
        shuffle=True,
    )

    val_gen = val_test_datagen.flow_from_directory(
        str(VAL_DIR),
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="binary",
        seed=SEED,
        shuffle=False,
    )

    test_gen = val_test_datagen.flow_from_directory(
        str(TEST_DIR),
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="binary",
        seed=SEED,
        shuffle=False,
    )

    return train_gen, val_gen, test_gen


# ============================================================
# 3. ARQUITECTURA LeNet-5 MODIFICADA
# ============================================================

def build_lenet5(input_shape=(32, 32, 3), dropout_rate=0.5) -> Model:
    """
    LeNet-5 adaptado para:
    - Entrada RGB (3 canales) en lugar de escala de grises
    - Clasificación binaria
    - Dropout 0.5 en capas densas
    - Activación ReLU moderna (original usaba tanh/sigmoid)
    """
    inputs = keras.Input(shape=input_shape, name="input_layer")

    # --- Bloque 1: C1 + S2 ---
    x = layers.Conv2D(
        6, kernel_size=5, padding="valid",
        activation="relu", name="C1_conv"
    )(inputs)

    x = layers.AveragePooling2D(
        pool_size=2, strides=2, name="S2_pool"
    )(x)

    # --- Bloque 2: C3 + S4 ---
    x = layers.Conv2D(
        16, kernel_size=5, padding="valid",
        activation="relu", name="C3_conv"
    )(x)

    x = layers.AveragePooling2D(
        pool_size=2, strides=2, name="S4_pool"
    )(x)

    # --- C5: Capa convolucional densa ---
    x = layers.Conv2D(
        120, kernel_size=5, padding="valid",
        activation="relu", name="C5_conv"
    )(x)

    x = layers.Flatten(name="flatten")(x)

    # --- F6: Fully connected ---
    x = layers.Dense(84, activation="relu", name="F6_dense")(x)
    x = layers.Dropout(dropout_rate, name="dropout_F6")(x)

    # --- Output binario ---
    outputs = layers.Dense(
        1, activation="sigmoid", name="output"
    )(x)

    model = Model(inputs, outputs, name="LeNet5_Binary")
    return model


# ============================================================
# 4. CALLBACKS
# ============================================================

def build_callbacks(experiment_name: str):
    """Early stopping + reducción de LR + guardado del mejor modelo."""
    os.makedirs(LENET5_MODELS, exist_ok=True)

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
            filepath=str(LENET5_MODELS / f"best_{experiment_name}.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        ),
    ]
    return callbacks


# ============================================================
# 5. ENTRENAMIENTO Y EXPERIMENTACIÓN
# ============================================================

def run_experiment(config: dict) -> dict:
    """Ejecuta un experimento completo con la configuración dada."""
    print(f"\n{'='*60}")
    print(f"  Experimento: {config['name']}")
    print(f"  LR={config['lr']} | Batch={config['batch_size']}")
    print(f"{'='*60}")

    # Generadores
    train_gen, val_gen, test_gen = build_generators(config["batch_size"])

    # Modelo
    model = build_lenet5(input_shape=(32, 32, 3), dropout_rate=DROPOUT)
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
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=build_callbacks(config["name"]),
        verbose=1,
    )

    # Evaluación en test
    test_loss, test_acc, test_auc, test_prec, test_rec = model.evaluate(
        test_gen, verbose=0
    )
    print(f"\n[TEST] Loss={test_loss:.4f} | Acc={test_acc:.4f} "
          f"| AUC={test_auc:.4f}")

    # Reporte de clasificación
    test_gen.reset()
    y_pred = (model.predict(test_gen) > 0.5).astype(int).flatten()
    y_true = test_gen.classes
    print("\n", classification_report(
        y_true, y_pred,
        target_names=["contaminado", "no_contaminado"]
    ))

    return {
        "name": config["name"],
        "model": model,
        "history": history,
        "test_gen": test_gen,
        "test_acc": test_acc,
        "test_auc": test_auc,
    }


# ============================================================
# 6. GRAD-CAM
# ============================================================

def make_gradcam_heatmap(img_array: np.ndarray,
                          model: Model,
                          last_conv_layer_name: str = "C3_conv") -> np.ndarray:
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
                    last_conv: str = "C3_conv"):
    """Superpone el heatmap Grad-CAM sobre la imagen satelital."""
    os.makedirs(GRADCAM_DIR, exist_ok=True)

    img = keras.utils.load_img(img_path, target_size=IMG_SIZE)
    img_array = keras.utils.img_to_array(img) / 255.0
    img_array_exp = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array_exp, verbose=0)[0][0]
    # image_dataset_from_directory ordena las clases alfabéticamente:
    # índice 0 = contaminado, índice 1 = no_contaminado.
    # La sigmoide predice la probabilidad de la clase 1 (no_contaminado).
    label = "NO CONTAMINADO" if pred > 0.5 else "CONTAMINADO"
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

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].imshow(img_array)
    axes[0].set_title("Imagen original\n(Satelital Himawari-8)")
    axes[0].axis("off")

    axes[1].imshow(heatmap_resized, cmap="jet")
    axes[1].set_title("Heatmap Grad-CAM\n(zonas activadas)")
    axes[1].axis("off")

    axes[2].imshow(superimposed)
    axes[2].set_title(
        f"Superposición\nPredicción: {label} ({confidence:.1%})"
    )
    axes[2].axis("off")

    plt.suptitle("Análisis Grad-CAM — LeNet-5 Clasificación de Cielo Satelital",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    output_path = str(GRADCAM_DIR / f"lenet5_gradcam_{label.lower().replace(' ', '_')}.png")
    plt.savefig(output_path, dpi=150)
    plt.show()


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

    plt.suptitle("Comparación de experimentos — LeNet-5 LSCIDMR",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(PLOTS_DIR / "lenet5_training_curves.png"), dpi=150)
    plt.show()


def plot_confusion_matrix(result: dict):
    """Matriz de confusión del mejor experimento."""
    os.makedirs(PLOTS_DIR, exist_ok=True)

    test_gen = result["test_gen"]
    model    = result["model"]

    test_gen.reset()
    y_pred = (model.predict(test_gen) > 0.5).astype(int).flatten()
    y_true = test_gen.classes

    cm_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm_matrix, annot=True, fmt="d", cmap="Blues",
        xticklabels=["contaminado", "no_contaminado"],
        yticklabels=["contaminado", "no_contaminado"],
    )
    plt.title(f"Matriz de Confusión\n{result['name']}")
    plt.ylabel("Real")
    plt.xlabel("Predicho")
    plt.tight_layout()
    plt.savefig(str(PLOTS_DIR / f"lenet5_confusion_{result['name']}.png"), dpi=150)
    plt.show()


# ============================================================
# 8. MAIN
# ============================================================

if __name__ == "__main__":
    all_results = []

    for config in EXPERIMENTS:
        result = run_experiment(config)
        all_results.append(result)

    plot_training_curves(all_results)

    best = max(all_results, key=lambda r: r["test_auc"])
    print(f"\n✓ Mejor experimento: {best['name']} "
          f"(AUC={best['test_auc']:.4f})")

    plot_confusion_matrix(best)

    sample_image = str(TEST_DIR / "contaminado" / "sample_001.png")
    if os.path.exists(sample_image):
        display_gradcam(sample_image, best["model"])
    else:
        print(f"\n[!] Coloca una imagen en: {sample_image} "
              "para visualizar Grad-CAM")
