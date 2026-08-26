"""
Construcción de etiquetas morfológicas a partir del árbol de decisión Galaxy Zoo 2.

Reemplaza la lógica de umbrales ad-hoc del notebook original por comparaciones
entre respuestas de la misma pregunta y por una medida explícita de confianza,
calculada como el producto de probabilidades condicionadas a lo largo de la rama
recorrida (ver la sección teórica del notebook 01).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Clases finales del problema principal (3 clases) y de la tarea auxiliar.
CLASES_PRINCIPALES = ["Smooth", "Disk", "Spiral"]
CLASE_PUNTUAL = "Star/Artifact"


def construir_etiquetas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade al DataFrame las columnas:

    - ``rama``          : resultado de Q1 (Smooth / Features-Disk / Star-Artifact)
    - ``label_grouped`` : etiqueta final (Smooth, Disk, Spiral, Star/Artifact)
    - ``conf_q1``       : max de las probabilidades de Q1
    - ``confianza``     : producto de probabilidades condicionadas de la rama

    A diferencia de la versión original, ``Spiral`` se separa de ``Disk``
    comparando Class4.1 vs Class4.2 (¿se ven brazos espirales?) en lugar de
    aplicar un umbral fijo de 0.3.
    """
    df = df.copy()

    # ---- Q1: ¿lisa, con estructura/disco, o estrella/artefacto? -----------
    q1 = ["Class1.1", "Class1.2", "Class1.3"]
    p1 = df[q1].to_numpy(dtype=float)
    idx1 = p1.argmax(axis=1)
    df["conf_q1"] = p1.max(axis=1)
    df["rama"] = np.array(["Smooth", "Features/Disk", CLASE_PUNTUAL])[idx1]

    # ---- Q4 (solo dentro de Features/Disk): ¿brazos espirales? ------------
    p4 = df[["Class4.1", "Class4.2"]].to_numpy(dtype=float)
    es_espiral = p4[:, 0] > p4[:, 1]
    conf_q4 = p4.max(axis=1)

    df["label_grouped"] = df["rama"]
    m_fd = df["rama"].to_numpy() == "Features/Disk"
    df.loc[m_fd & es_espiral, "label_grouped"] = "Spiral"
    df.loc[m_fd & ~es_espiral, "label_grouped"] = "Disk"

    # ---- Confianza: producto de condicionadas a lo largo de la rama -------
    df["confianza"] = df["conf_q1"]
    df.loc[m_fd, "confianza"] = df.loc[m_fd, "conf_q1"].to_numpy() * conf_q4[m_fd]

    return df


def subclase_smooth(df: pd.DataFrame) -> pd.Series:
    """Q7 dentro de Smooth: redonda / intermedia / cigarro. Útil para la interfaz."""
    q7 = ["Class7.1", "Class7.2", "Class7.3"]
    nombres = np.array(["Redonda", "Intermedia", "Cigarro"])
    idx = df[q7].to_numpy(dtype=float).argmax(axis=1)
    s = pd.Series(nombres[idx], index=df.index, name="subclase")
    s[df["label_grouped"] != "Smooth"] = pd.NA
    return s


def filtrar_por_confianza(
    df: pd.DataFrame,
    umbral: float = 0.7,
    excluir_puntuales: bool = True,
) -> pd.DataFrame:
    """
    Conjunto 'clean': galaxias donde el consenso de los voluntarios fue claro.

    Reportar el desempeño sobre 'clean' y sobre el conjunto completo cuantifica
    cuánto del error del modelo es ruido de etiqueta humana. La diferencia entre
    ambos es un resultado, no un truco para inflar métricas: hay que presentar
    los dos números.
    """
    out = df[df["confianza"] >= umbral]
    if excluir_puntuales:
        out = out[out["label_grouped"] != CLASE_PUNTUAL]
    return out.copy()


def etiqueta_binaria_puntual(df: pd.DataFrame) -> np.ndarray:
    """Tarea auxiliar: objeto puntual/artefacto vs. galaxia resuelta."""
    return (df["label_grouped"] == CLASE_PUNTUAL).to_numpy(dtype=int)


def resumen(df: pd.DataFrame, umbral: float = 0.7) -> pd.DataFrame:
    """Tabla comparativa de la distribución de clases: completo vs. filtrado."""
    full = df["label_grouped"].value_counts()
    clean = filtrar_por_confianza(df, umbral, excluir_puntuales=False)
    clean = clean["label_grouped"].value_counts()
    tab = pd.DataFrame({"Completo": full, f"Confianza>={umbral}": clean}).fillna(0)
    tab = tab.astype(int)
    tab["Retenido %"] = (tab.iloc[:, 1] / tab.iloc[:, 0] * 100).round(1)
    return tab
