"""
Carga de datos, caché de imágenes y splits congelados.

La regla que sostiene toda la comparación final: TODOS los modelos (clásicos,
MLP y CNN) deben entrenarse y evaluarse sobre exactamente los mismos índices,
guardados en ``data/processed/splits.json``. Sin esto, la tabla comparativa de
la ponencia no compara nada.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RAIZ = Path(__file__).resolve().parents[2]
DIR_RAW = RAIZ / "data" / "raw"
DIR_PROC = RAIZ / "data" / "processed"
SEED = 42


# --------------------------------------------------------------------------- #
def cargar_csv(nombre: str = "training_solutions_rev1.csv") -> pd.DataFrame:
    return pd.read_csv(DIR_RAW / nombre)


def cargar_cache(size: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """
    Devuelve (imagenes_uint8, galaxy_ids) del caché generado por
    ``scripts/build_cache.py``.

    Se usa ``mmap_mode='r'`` para que los 3 GB no se carguen enteros en RAM;
    NumPy lee bajo demanda y el sistema operativo cachea lo que se usa.
    """
    imgs = np.load(DIR_PROC / f"images_{size}.npy", mmap_mode="r")
    ids = np.load(DIR_PROC / f"images_{size}_ids.npy")
    return imgs, ids


# --------------------------------------------------------------------------- #
def crear_splits(
    df: pd.DataFrame,
    col_label: str = "label_grouped",
    test_size: float = 0.20,
    val_size: float = 0.15,
    seed: int = SEED,
    ruta: Path | None = None,
) -> dict[str, list[int]]:
    """
    Split estratificado train/val/test sobre los GalaxyID (no sobre posiciones),
    para que siga siendo válido aunque después se reordene el DataFrame.

    ``val_size`` se toma como fracción del conjunto de entrenamiento restante.
    """
    ruta = ruta or (DIR_PROC / "splits.json")
    ids = df["GalaxyID"].to_numpy()
    y = df[col_label].to_numpy()

    id_tr, id_te = train_test_split(
        ids, test_size=test_size, stratify=y, random_state=seed
    )
    y_tr = df.set_index("GalaxyID").loc[id_tr, col_label].to_numpy()
    id_tr, id_va = train_test_split(
        id_tr, test_size=val_size, stratify=y_tr, random_state=seed
    )

    splits = {
        "train": id_tr.tolist(),
        "val": id_va.tolist(),
        "test": id_te.tolist(),
        "meta": {"seed": seed, "col_label": col_label,
                 "test_size": test_size, "val_size": val_size,
                 "n_total": int(len(ids))},
    }
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, "w") as f:
        json.dump(splits, f)
    return splits


def cargar_splits(ruta: Path | None = None) -> dict:
    ruta = ruta or (DIR_PROC / "splits.json")
    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe {ruta}. Ejecuta crear_splits() una sola vez y "
            "no lo regeneres: todos los modelos deben compartir estos índices."
        )
    with open(ruta) as f:
        return json.load(f)


def indices_de_ids(ids_split: list[int], ids_cache: np.ndarray) -> np.ndarray:
    """Traduce GalaxyID -> posiciones dentro del array del caché."""
    pos = {int(g): i for i, g in enumerate(ids_cache)}
    faltan = [g for g in ids_split if int(g) not in pos]
    if faltan:
        raise KeyError(f"{len(faltan)} GalaxyID del split no están en el caché "
                       f"(primeros: {faltan[:5]})")
    return np.array([pos[int(g)] for g in ids_split], dtype=np.int64)


# --------------------------------------------------------------------------- #
def preparar_conjunto(
    df: pd.DataFrame,
    size: int = 128,
    col_label: str = "label_grouped",
):
    """
    Devuelve un dict con imágenes y etiquetas por partición, alineadas y listas.

    Las imágenes salen en uint8; la normalización a [0,1] se hace en el pipeline
    (modelos clásicos) o en la capa Rescaling (CNN), nunca aquí.
    """
    from sklearn.preprocessing import LabelEncoder

    imgs, ids_cache = cargar_cache(size)
    splits = cargar_splits()
    df_idx = df.set_index("GalaxyID")

    le = LabelEncoder().fit(df[col_label])
    out = {"label_encoder": le, "clases": list(le.classes_)}

    for parte in ("train", "val", "test"):
        gid = [g for g in splits[parte] if g in df_idx.index]
        pos = indices_de_ids(gid, ids_cache)
        out[parte] = {
            "ids": np.asarray(gid),
            "X": imgs[pos],  # vista mmap; copiar con np.array() si hace falta
            "y": le.transform(df_idx.loc[gid, col_label].to_numpy()),
            "confianza": df_idx.loc[gid, "confianza"].to_numpy()
            if "confianza" in df_idx.columns else None,
        }
    return out


def fijar_semillas(seed: int = SEED) -> None:
    """Reproducibilidad en numpy, random y (si está disponible) TensorFlow."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        pass
