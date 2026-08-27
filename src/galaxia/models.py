"""
Constructores de modelos para las fases 2-4.

Separado de los notebooks para que la CNN pueda lanzarse desde terminal
(scripts/train_cnn.py) mientras el notebook sigue trabajando en otra cosa.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# Pesos de clase
# --------------------------------------------------------------------------- #
def pesos_clase(y: np.ndarray) -> dict[int, float]:
    """class_weight='balanced' en formato dict, que es lo que espera Keras."""
    from sklearn.utils.class_weight import compute_class_weight

    clases = np.unique(y)
    w = compute_class_weight("balanced", classes=clases, y=y)
    return {int(c): float(p) for c, p in zip(clases, w)}


# --------------------------------------------------------------------------- #
# Pipelines clásicos
# --------------------------------------------------------------------------- #
def pipeline_pca(clf, n_componentes: float | int = 0.95, cache_dir: str | None = None):
    """
    Pipeline estándar del proyecto: escalado -> PCA -> clasificador.

    cache_dir : str | None
        Carpeta de caché de joblib. En GridSearchCV evita recalcular el PCA
        para cada combinación de hiperparámetros del clasificador: con 30
        combinaciones y 5 pliegues, la diferencia es de ~30 min a ~2 min.
    """
    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pasos = [
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_componentes, random_state=42)),
        ("clf", clf),
    ]
    if cache_dir:
        from joblib import Memory

        return Pipeline(pasos, memory=Memory(cache_dir, verbose=0))
    return Pipeline(pasos)


def grid_knn() -> dict:
    """Rejilla acotada para el sprint de 4 días. 5x2x3 = 30 combinaciones."""
    return {
        "clf__n_neighbors": [3, 7, 15, 25, 41],
        "clf__weights": ["uniform", "distance"],
        "clf__metric": ["euclidean", "manhattan", "cosine"],
    }


# --------------------------------------------------------------------------- #
# Redes densas (Keras)
# --------------------------------------------------------------------------- #
def build_mlp(input_dim: int, n_clases: int, capas=(256, 128), dropout=0.3, lr=1e-3):
    """MLP sobre vectores de características (PCA o parámetros morfológicos)."""
    import keras

    capas_lista = [keras.layers.Input(shape=(input_dim,))]
    for u in capas:
        capas_lista += [
            keras.layers.Dense(u, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(dropout),
        ]
    # dtype float32 explícito: si hay mixed_float16 activo, la salida softmax
    # debe quedarse en float32 o la pérdida se desestabiliza.
    capas_lista.append(
        keras.layers.Dense(n_clases, activation="softmax", dtype="float32")
    )

    m = keras.Sequential(capas_lista)
    m.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return m


# --------------------------------------------------------------------------- #
# CNN
# --------------------------------------------------------------------------- #
def capas_aumentacion():
    """
    Aumentación con justificación física, no por convención de ML.

    Las galaxias no tienen orientación privilegiada en el cielo: rotarlas o
    reflejarlas produce imágenes igual de válidas. Por eso se usa rotación
    completa de 360 grados (factor=1.0), que en fotografía natural sería
    absurdo pero aquí es exactamente correcto.

    NO se altera el color: es señal física real (poblaciones estelares viejas
    rojas vs. formación estelar joven azul). Distorsionarlo destruiría el
    segundo discriminante morfológico más fuerte.
    """
    import keras

    return keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal_and_vertical"),
            keras.layers.RandomRotation(1.0, fill_mode="constant", fill_value=0.0),
            keras.layers.RandomZoom(0.1, fill_mode="constant", fill_value=0.0),
            keras.layers.RandomTranslation(0.05, 0.05, fill_mode="constant",
                                           fill_value=0.0),
        ],
        name="aumentacion",
    )


def build_cnn(n_clases: int, img: int = 128, filtros=(32, 64, 128, 256),
              dropout=0.4, lr=1e-3, aumentar: bool = True):
    """CNN desde cero. Bloques Conv-Conv-Pool con BatchNorm y GAP final."""
    import keras

    x_in = keras.layers.Input(shape=(img, img, 3))
    x = keras.layers.Rescaling(1.0 / 255)(x_in)
    if aumentar:
        x = capas_aumentacion()(x)

    for f in filtros:
        x = keras.layers.Conv2D(f, 3, padding="same", use_bias=False)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.Conv2D(f, 3, padding="same", use_bias=False)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.MaxPooling2D()(x)

    # Nombre fijo: Grad-CAM necesita localizar la última capa convolucional.
    x = keras.layers.Conv2D(256, 3, padding="same", activation="relu",
                            name="ultima_conv")(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(dropout)(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dropout(dropout * 0.75)(x)
    salida = keras.layers.Dense(n_clases, activation="softmax", dtype="float32")(x)

    m = keras.Model(x_in, salida, name="cnn_galaxia")
    m.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return m


def callbacks_estandar(ruta_modelo: str, paciencia: int = 10):
    import keras

    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=paciencia, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=max(paciencia // 2, 3),
            min_lr=1e-6, verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            ruta_modelo, monitor="val_loss", save_best_only=True, verbose=0
        ),
    ]
