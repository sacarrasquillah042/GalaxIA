#!/usr/bin/env python3
"""
Prepara el paquete ligero para publicar la app en la web.

Problema que resuelve
---------------------
La app local depende de tres cosas que no existirán en el servidor:

  1. data/processed/images_128.npy  -> 3 GB, imposible de subir
  2. data/raw/training_solutions_rev1.csv -> no se publica
  3. TensorFlow + models/cnn_128.keras -> excede la memoria del plan gratuito

Solución: precalcular aquí (con GPU) las predicciones, los mapas Grad-CAM y las
miniaturas de vecinos, y dejar en app/assets/web/ un paquete pequeño que la app
solo tiene que leer. La versión publicada no necesita TensorFlow.

Uso:
    python scripts/export_web.py --n 150

Genera:
    app/assets/web/muestra.npz      imágenes de consulta + etiquetas + ids
    app/assets/web/gradcam.npz      superposiciones Grad-CAM precalculadas
    app/assets/web/vecinos.npz      miniaturas de los 5 vecinos por consulta
    app/assets/web/predicciones.json  probabilidades de cada modelo
    app/assets/web/resumen.json     conteos de clases y metadatos
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

import numpy as np  # noqa: E402

DIR_WEB = RAIZ / "app" / "assets" / "web"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150,
                    help="Galaxias de muestra. 150 ~ 35 MB en total.")
    ap.add_argument("--px-vecino", type=int, default=80)
    args = ap.parse_args()

    from galaxia import data, labels

    DIR_WEB.mkdir(parents=True, exist_ok=True)

    # ---- Datos y muestra --------------------------------------------------
    df = labels.construir_etiquetas(data.cargar_csv())
    p = data.DIR_PROC / "umbral.json"
    umbral = json.loads(p.read_text())["umbral"] if p.exists() else 0.6
    df_clean = labels.filtrar_por_confianza(df, umbral=umbral)

    d = data.preparar_conjunto(df_clean, size=128)
    clases = d["clases"]
    ids_t, y_t = d["test"]["ids"], d["test"]["y"]

    rng = np.random.default_rng(42)
    # Muestra estratificada: todas las clases representadas
    sel = []
    por_clase = max(args.n // len(clases), 1)
    for k in range(len(clases)):
        idx = np.where(y_t == k)[0]
        sel += list(rng.choice(idx, size=min(por_clase, len(idx)),
                               replace=False))
    sel = np.sort(np.array(sel))
    X = np.array(d["test"]["X"][sel])
    y = y_t[sel]
    ids = ids_t[sel]
    print(f"Muestra: {len(sel)} galaxias  {dict(zip(clases, np.bincount(y)))}")

    np.savez_compressed(DIR_WEB / "muestra.npz", X=X, y=y, ids=ids,
                        clases=np.array(clases))

    # ---- Predicciones de cada modelo -------------------------------------
    predicciones = {}

    ruta_cnn = next((RAIZ / "models" / n for n in ["cnn_128.keras", "cnn_224.keras"]
                     if (RAIZ / "models" / n).exists()), None)
    cnn = None
    if ruta_cnn:
        import keras
        cnn = keras.models.load_model(ruta_cnn)
        proba = np.asarray(cnn.predict(X.astype(np.float32), verbose=0),
                           dtype=float)
        predicciones["CNN"] = proba.tolist()
        print(f"CNN: {(proba.argmax(1) == y).mean():.3f} de acierto en la muestra")
    else:
        print("Sin CNN en models/")

    # ---- Grad-CAM precalculado -------------------------------------------
    if cnn is not None:
        from galaxia.explain import grad_cam, superponer
        capas = []
        for i in range(len(X)):
            try:
                mapa, pred, _ = grad_cam(cnn, X[i])
                capas.append(superponer(X[i].astype(np.uint8), mapa))
            except Exception:
                capas.append(X[i].astype(np.uint8))
            if (i + 1) % 25 == 0:
                print(f"  Grad-CAM {i+1}/{len(X)}")
        np.savez_compressed(DIR_WEB / "gradcam.npz",
                            G=np.stack(capas).astype(np.uint8))
        print("Grad-CAM guardado")

    # ---- Vecinos del KNN --------------------------------------------------
    ruta_knn = RAIZ / "models" / "knn.pkl"
    if ruta_knn.exists():
        import cv2
        import joblib
        from galaxia import features

        K = joblib.load(ruta_knn)
        z = K["prep"].transform(features.a_vector(X, size=64))[:, : K["K"]]
        _, vec = K["nn"].kneighbors(z, n_neighbors=5)

        imgs, ids_cache = data.cargar_cache(128)
        pos = {int(g): i for i, g in enumerate(ids_cache)}
        px = args.px_vecino
        V = np.zeros((len(X), 5, px, px, 3), dtype=np.uint8)
        etiquetas = np.zeros((len(X), 5), dtype=np.int16)
        ids_vec = np.zeros((len(X), 5), dtype=np.int64)
        for i in range(len(X)):
            for j, k in enumerate(vec[i]):
                g = int(K["ids_train"][k])
                V[i, j] = cv2.resize(np.array(imgs[pos[g]]), (px, px),
                                     interpolation=cv2.INTER_AREA)
                etiquetas[i, j] = int(K["y_train"][k])
                ids_vec[i, j] = g
        np.savez_compressed(DIR_WEB / "vecinos.npz", V=V, y=etiquetas,
                            ids=ids_vec, clases=np.array(K["clases"]))
        print("Vecinos guardados")

        # KNN también predice sobre la muestra
        predicciones["KNN"] = K["knn"].predict_proba(z).tolist()

    # ---- Otros modelos sobre la muestra ----------------------------------
    for nombre, archivo in [("MLP-PCA", "mlp_pca.keras")]:
        ruta = RAIZ / "models" / archivo
        if not ruta.exists() or not ruta_knn.exists():
            continue
        try:
            import joblib
            import keras
            from galaxia import features

            K = joblib.load(ruta_knn)
            z = K["prep"].transform(features.a_vector(X, size=64))[:, : K["K"]]
            m = keras.models.load_model(ruta)
            predicciones[nombre] = np.asarray(
                m.predict(z, verbose=0), dtype=float).tolist()
            print(f"{nombre} añadido")
        except Exception as e:
            print(f"{nombre} omitido: {e}")

    (DIR_WEB / "predicciones.json").write_text(
        json.dumps({"clases": clases, "modelos": predicciones}))

    # ---- Resumen (evita leer el CSV en el servidor) ----------------------
    (DIR_WEB / "resumen.json").write_text(json.dumps({
        "conteos": {k: int(v) for k, v in
                    df_clean["label_grouped"].value_counts().items()},
        "n_total": int(len(df)),
        "n_limpio": int(len(df_clean)),
        "umbral": float(umbral),
        "clases": clases,
    }, ensure_ascii=False, indent=2))

    # ---- Tamaño final -----------------------------------------------------
    total = sum(f.stat().st_size for f in DIR_WEB.glob("*"))
    print(f"\nPaquete web: {total/1e6:.1f} MB en {DIR_WEB}")
    for f in sorted(DIR_WEB.glob("*")):
        print(f"  {f.name:22s} {f.stat().st_size/1e6:6.2f} MB")
    if total > 90e6:
        print("\nAVISO: supera 90 MB. Baje --n para reducirlo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
