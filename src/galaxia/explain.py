"""
Interpretabilidad de la CNN mediante Grad-CAM.

Doble propósito:

1. Científico: verificar que la red mira el bulbo y los brazos, y no artefactos
   del fondo o bordes del recorte. Sin esto, un F1-macro de 0.95 no prueba que
   el modelo aprendió morfología.
2. Divulgación: es el elemento visual más convincente de la interfaz.
"""
from __future__ import annotations

import numpy as np


def ultima_capa_conv(modelo) -> str:
    """Localiza la última capa convolucional del modelo."""
    for capa in reversed(modelo.layers):
        if len(getattr(capa, "output_shape", getattr(capa, "output", None).shape) or []) == 4:
            if "conv" in capa.name.lower():
                return capa.name
    raise ValueError("No se encontró capa convolucional")


def grad_cam(modelo, img: np.ndarray, capa: str | None = None,
             clase: int | None = None) -> tuple[np.ndarray, int, np.ndarray]:
    """
    Mapa de activación por gradientes.

    Parameters
    ----------
    img : array (H, W, 3) uint8 o float; se le añade la dimensión de lote.

    Returns
    -------
    (mapa, clase_predicha, probabilidades)
    """
    import keras
    import tensorflow as tf

    capa = capa or "ultima_conv"
    x = np.asarray(img, dtype=np.float32)[None, ...]

    grad_model = keras.Model(
        modelo.inputs, [modelo.get_layer(capa).output, modelo.output]
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(x, training=False)
        preds = tf.cast(preds, tf.float32)
        if clase is None:
            clase = int(tf.argmax(preds[0]))
        objetivo = preds[:, clase]

    grads = tape.gradient(objetivo, conv_out)
    pesos = tf.reduce_mean(grads, axis=(0, 1, 2))
    mapa = tf.reduce_sum(tf.cast(conv_out[0], tf.float32) * tf.cast(pesos, tf.float32),
                         axis=-1)
    mapa = tf.maximum(mapa, 0)
    mapa = mapa / (tf.reduce_max(mapa) + 1e-8)
    return mapa.numpy(), int(clase), np.asarray(preds[0], dtype=float)


def superponer(img_uint8: np.ndarray, mapa: np.ndarray,
               alpha: float = 0.45) -> np.ndarray:
    """Devuelve la imagen con el mapa de calor encima, lista para mostrar."""
    import cv2
    import matplotlib.cm as cm

    h, w = img_uint8.shape[:2]
    m = cv2.resize(mapa, (w, h), interpolation=cv2.INTER_CUBIC)
    heat = (cm.jet(m)[..., :3] * 255).astype(np.uint8)
    return cv2.addWeighted(img_uint8.astype(np.uint8), 1 - alpha, heat, alpha, 0)


def rejilla_gradcam(modelo, imgs: np.ndarray, etiquetas, clases,
                    n_por_clase: int = 3, ruta_salida: str | None = None):
    """Rejilla de ejemplos con Grad-CAM, una fila por clase."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(clases), n_por_clase * 2,
                             figsize=(3 * n_por_clase, 2.4 * len(clases)))
    for fila, c in enumerate(clases):
        idx = np.where(np.asarray(etiquetas) == fila)[0][:n_por_clase]
        for j, i in enumerate(idx):
            mapa, pred, proba = grad_cam(modelo, imgs[i])
            axes[fila, 2 * j].imshow(imgs[i])
            axes[fila, 2 * j].set_title(c if j == 0 else "", fontsize=9,
                                        loc="left", fontweight="bold")
            axes[fila, 2 * j + 1].imshow(superponer(imgs[i], mapa))
            axes[fila, 2 * j + 1].set_title(f"{clases[pred]} {proba[pred]:.2f}",
                                            fontsize=8)
        for a in axes[fila]:
            a.axis("off")
    fig.tight_layout()
    if ruta_salida:
        fig.savefig(ruta_salida, dpi=140)
    return fig
