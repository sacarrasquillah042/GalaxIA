#!/usr/bin/env python3
"""
Precomputa el caché de imágenes recortadas y redimensionadas.

Motivo: decodificar 61 578 JPEG de 424x424 en cada época deja la RTX 4060 Ti
esperando al CPU. Haciéndolo una sola vez, cada época pasa a ser cómputo puro
de GPU. Con los 16 hilos del 5700X esto toma unos 8-12 minutos.

Uso:
    python scripts/build_cache.py --size 128
    python scripts/build_cache.py --size 224      # si sobra tiempo

Salida:
    data/processed/images_<size>.npy       (N, size, size, 3) uint8
    data/processed/images_<size>_ids.npy   (N,) int64  -- orden de las filas
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from galaxia.features import cargar_imagen  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
DIR_RAW = RAIZ / "data" / "raw"
DIR_PROC = RAIZ / "data" / "processed"

_PATH_IMGS: str = ""
_SIZE: int = 128


def _init(path_imgs: str, size: int) -> None:
    global _PATH_IMGS, _SIZE
    _PATH_IMGS, _SIZE = path_imgs, size


def _procesar(gid: int):
    try:
        img = cargar_imagen(gid, _PATH_IMGS, size=_SIZE, normalizar=False)
        return int(gid), img.astype(np.uint8)
    except Exception as e:  # imagen faltante o corrupta
        return int(gid), None if not isinstance(e, KeyboardInterrupt) else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--csv", default="training_solutions_rev1.csv")
    ap.add_argument("--images", default="images_training_rev1")
    ap.add_argument("--workers", type=int, default=14)  # 16 hilos, dejar 2 libres
    args = ap.parse_args()

    path_imgs = str(DIR_RAW / args.images)
    df = pd.read_csv(DIR_RAW / args.csv)
    ids = df["GalaxyID"].to_numpy()
    n = len(ids)

    gb = n * args.size * args.size * 3 / 1e9
    print(f"{n} imágenes -> {args.size}x{args.size}  (~{gb:.2f} GB en disco)")

    salida = np.zeros((n, args.size, args.size, 3), dtype=np.uint8)
    ids_ok, fallos = np.zeros(n, dtype=np.int64), []

    with ProcessPoolExecutor(
        max_workers=args.workers, initializer=_init, initargs=(path_imgs, args.size)
    ) as ex:
        for i, (gid, img) in enumerate(
            tqdm(ex.map(_procesar, ids, chunksize=64), total=n, desc="Caché")
        ):
            ids_ok[i] = gid
            if img is None:
                fallos.append(gid)
            else:
                salida[i] = img

    if fallos:
        print(f"\nATENCIÓN: {len(fallos)} imágenes no se pudieron leer.")
        print(f"  Primeras: {fallos[:10]}")
        print("  Quedaron como ceros. Fíltralas del DataFrame antes de entrenar.")
        np.save(DIR_PROC / f"fallos_{args.size}.npy", np.array(fallos))

    DIR_PROC.mkdir(parents=True, exist_ok=True)
    np.save(DIR_PROC / f"images_{args.size}.npy", salida)
    np.save(DIR_PROC / f"images_{args.size}_ids.npy", ids_ok)
    print(f"\nGuardado en {DIR_PROC}/images_{args.size}.npy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
