#!/usr/bin/env python3
"""
Exporta todo lo que la interfaz necesita. La app NO entrena ni recalcula nada:
solo carga estos artefactos. Si la app tarda más de 3 s en arrancar, algo se
está calculando que debería estar aquí.

    python scripts/export_artefactos.py

Genera:
    app/assets/galeria/<clase>/*.jpg   ejemplos curados por clase
    app/assets/galeria.json            metadatos de la galería
    app/assets/muestra_test.npz        imágenes de test para el clasificador
    reports/metrics.json               (ya existente, se valida)
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

N_GALERIA = 12
N_MUESTRA_TEST = 600


def main() -> int:
    from galaxia import data, labels

    dir_assets = RAIZ / "app" / "assets"
    dir_gal = dir_assets / "galeria"
    if dir_gal.exists():
        shutil.rmtree(dir_gal)
    dir_gal.mkdir(parents=True)

    # ---- Datos --------------------------------------------------------
    df = labels.construir_etiquetas(data.cargar_csv())
    p = data.DIR_PROC / "umbral.json"
    umbral = json.loads(p.read_text())["umbral"] if p.exists() else 0.6
    df_clean = labels.filtrar_por_confianza(df, umbral=umbral)

    imgs, ids_cache = data.cargar_cache(128)
    pos = {int(g): i for i, g in enumerate(ids_cache)}
    splits = data.cargar_splits()
    ids_test = set(int(g) for g in splits["test"])

    # ---- Galería: ejemplos de ENTRENAMIENTO, alta confianza ------------
    # Deliberadamente no se usan galaxias de test: la galería es material
    # educativo, y mezclarla con el conjunto de evaluación sería confuso.
    meta = {}
    for clase in ["Smooth", "Disk", "Spiral"]:
        sub = df_clean[(df_clean.label_grouped == clase)
                       & (~df_clean.GalaxyID.isin(ids_test))]
        sub = sub.nlargest(N_GALERIA * 4, "confianza").sample(
            N_GALERIA, random_state=42)

        carpeta = dir_gal / clase.replace("/", "_")
        carpeta.mkdir(parents=True, exist_ok=True)
        entradas = []
        for gid in sub.GalaxyID:
            gid = int(gid)
            arr = np.array(imgs[pos[gid]])
            Image.fromarray(arr).save(carpeta / f"{gid}.jpg", quality=92)
            fila = df_clean[df_clean.GalaxyID == gid].iloc[0]
            entradas.append({
                "galaxy_id": gid,
                "archivo": f"{clase.replace('/', '_')}/{gid}.jpg",
                "confianza": round(float(fila.confianza), 3),
                "edge_on": bool(fila.edge_on),
            })
        meta[clase] = entradas
        print(f"{clase:8s} {len(entradas)} imágenes")

    (dir_assets / "galeria.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False))

    # ---- Muestra de test para el clasificador interactivo --------------
    d = data.preparar_conjunto(df_clean, size=128)
    ids_t = d["test"]["ids"]
    y_t = d["test"]["y"]
    rng = np.random.default_rng(42)
    sel = rng.choice(len(ids_t), size=min(N_MUESTRA_TEST, len(ids_t)),
                     replace=False)
    np.savez_compressed(
        dir_assets / "muestra_test.npz",
        X=np.array(d["test"]["X"][np.sort(sel)]),
        y=y_t[np.sort(sel)],
        ids=ids_t[np.sort(sel)],
        clases=np.array(d["clases"]),
    )
    print(f"\nMuestra de test: {len(sel)} galaxias "
          f"({(dir_assets / 'muestra_test.npz').stat().st_size/1e6:.1f} MB)")

    # ---- Validación de modelos ----------------------------------------
    print("\nModelos disponibles para la app:")
    for f in sorted((RAIZ / "models").glob("*")):
        print(f"  {f.name}  ({f.stat().st_size/1e6:.1f} MB)")

    mj = RAIZ / "reports" / "metrics.json"
    if mj.exists():
        print(f"\nmetrics.json: {len(json.loads(mj.read_text()))} modelos")
    else:
        print("\nFALTA reports/metrics.json")
        return 1

    print("\nListo. Arrancar la app con:  streamlit run app/Inicio.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
