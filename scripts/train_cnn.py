#!/usr/bin/env python3
"""
Entrena la CNN desde terminal, para no bloquear el notebook.

    python scripts/train_cnn.py --size 128 --epocas 40 --batch 128

Lee las etiquetas y los splits del proyecto, entrena, evalúa sobre el test
congelado y escribe en reports/metrics.json igual que los demás modelos.

Con la RTX 4060 Ti y mixed_float16: ~25-40 s por época a 128x128.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

import numpy as np  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--epocas", type=int, default=40)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--umbral", type=float, default=None,
                    help="Confianza mínima. Por defecto lee umbral.json.")
    ap.add_argument("--sin-aumentacion", action="store_true")
    ap.add_argument("--nombre", default=None)
    ap.add_argument("--mixed", action="store_true", default=True)
    args = ap.parse_args()

    import tensorflow as tf
    import keras

    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
    print(f"GPU: {gpus}")
    if not gpus:
        print("ATENCIÓN: sin GPU esto tarda horas. Abortando.")
        return 1

    if args.mixed:
        keras.mixed_precision.set_global_policy("mixed_float16")
        print(f"Precisión: {keras.mixed_precision.global_policy().name}")

    from galaxia import data, labels, models, evaluate

    data.fijar_semillas(42)

    # ---- Datos --------------------------------------------------------
    df = labels.construir_etiquetas(data.cargar_csv())
    umbral = args.umbral
    if umbral is None:
        p = data.DIR_PROC / "umbral.json"
        umbral = json.loads(p.read_text())["umbral"] if p.exists() else 0.6
    df_clean = labels.filtrar_por_confianza(df, umbral=umbral)
    print(f"Umbral {umbral} -> {len(df_clean)} galaxias")

    d = data.preparar_conjunto(df_clean, size=args.size)
    clases = d["clases"]
    n_clases = len(clases)
    print(f"Clases: {clases}")

    # El caché es un mmap: se copia a RAM la parte que se usa.
    # ~30k x 128x128x3 = 1.5 GB, cabe de sobra.
    Xtr = np.array(d["train"]["X"]); ytr = d["train"]["y"]
    Xva = np.array(d["val"]["X"]);   yva = d["val"]["y"]
    Xte = np.array(d["test"]["X"]);  yte = d["test"]["y"]
    print(f"train {Xtr.shape} | val {Xva.shape} | test {Xte.shape}")

    ds_tr = (tf.data.Dataset.from_tensor_slices((Xtr, ytr))
             .shuffle(8192, seed=42).batch(args.batch)
             .prefetch(tf.data.AUTOTUNE))
    ds_va = (tf.data.Dataset.from_tensor_slices((Xva, yva))
             .batch(args.batch).prefetch(tf.data.AUTOTUNE))

    # ---- Modelo -------------------------------------------------------
    nombre = args.nombre or f"CNN {args.size}px"
    modelo = models.build_cnn(n_clases, img=args.size, lr=args.lr,
                              aumentar=not args.sin_aumentacion)
    modelo.summary()

    ruta = str(RAIZ / "models" / f"cnn_{args.size}.keras")
    (RAIZ / "models").mkdir(exist_ok=True)

    t0 = time.time()
    hist = modelo.fit(
        ds_tr, validation_data=ds_va, epochs=args.epocas,
        class_weight=models.pesos_clase(ytr),
        callbacks=models.callbacks_estandar(ruta),
        verbose=1,
    )
    t_fit = time.time() - t0
    print(f"\nEntrenamiento: {t_fit/60:.1f} min ({len(hist.history['loss'])} épocas)")

    # ---- Evaluación ---------------------------------------------------
    t0 = time.time()
    proba = modelo.predict(Xte, batch_size=args.batch, verbose=0)
    t_inf = time.time() - t0
    proba = np.asarray(proba, dtype=np.float64)

    r = evaluate.evaluar(nombre, yte, proba.argmax(1), proba, clases, t_fit, t_inf)
    print(f"\n{nombre}: acc={r['accuracy']:.4f}  F1-macro={r['macro_f1']:.4f}  "
          f"AUC={r['roc_auc_macro']:.4f}  AP={r['avg_precision_macro']:.4f}")

    # Predicciones guardadas: las necesita McNemar el día 3.
    np.savez_compressed(
        RAIZ / "reports" / f"pred_{evaluate._slug(nombre)}.npz",
        y_true=yte, y_pred=proba.argmax(1), proba=proba, ids=d["test"]["ids"],
    )
    json.dump({k: [float(x) for x in v] for k, v in hist.history.items()},
              open(RAIZ / "reports" / f"hist_{evaluate._slug(nombre)}.json", "w"))

    # Curvas de entrenamiento
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for i, (k, tit) in enumerate([("loss", "Pérdida"), ("accuracy", "Exactitud")]):
        ax[i].plot(hist.history[k], label="train")
        ax[i].plot(hist.history[f"val_{k}"], label="val")
        ax[i].set_title(tit); ax[i].set_xlabel("Época")
        ax[i].legend(); ax[i].grid(alpha=0.3)
    fig.suptitle(nombre); fig.tight_layout()
    fig.savefig(RAIZ / "reports" / "figures" / f"hist_{evaluate._slug(nombre)}.png", dpi=150)
    print("Listo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
