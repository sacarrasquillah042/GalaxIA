"""
Construcción de etiquetas morfológicas a partir del árbol de decisión Galaxy Zoo 2.

CORRECCIÓN v2 (importante)
--------------------------
La versión anterior calculaba la confianza como conf(Q1) * conf(Q4), saltándose
Q2. Eso está mal: en el árbol GZ2, Q4 ("¿se ve un patrón espiral?") SOLO se
formula a discos que NO están de canto. Si Q2 responde "sí, de canto", el flujo
salta a Q9 (forma del bulbo) y Q4 nunca se pregunta, de modo que Class4.1 y
Class4.2 quedan ambas cerca de cero.

Consecuencia del error: todas las galaxias de canto quedaban con confianza ~0 y
el filtro las eliminaba. La clase Disk se quedaba con ~145 objetos (discos de
cara sin brazos, que son raros) en lugar de la población real de discos.

Recorrido correcto:

    Q1 --+-- Class1.1 lisa ........................ Smooth
         +-- Class1.3 estrella/artefacto .......... Star/Artifact
         +-- Class1.2 con estructura/disco
              +-- Q2 --+-- Class2.1 de canto ...... Disk (edge-on)
                       +-- Class2.2 no de canto
                            +-- Q4 --+-- Class4.1 . Spiral
                                     +-- Class4.2 . Disk (sin brazos)

Confianza = producto de las probabilidades condicionadas de la rama recorrida.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CLASES_PRINCIPALES = ["Smooth", "Disk", "Spiral"]
CLASE_PUNTUAL = "Star/Artifact"


def construir_etiquetas(
    df: pd.DataFrame,
    separar_edge_on: bool = False,
) -> pd.DataFrame:
    """
    Anade las columnas ``rama``, ``label_grouped``, ``conf_q1`` y ``confianza``.

    Parameters
    ----------
    separar_edge_on : bool
        False (por defecto) -> 3 clases + puntuales: Smooth / Disk / Spiral.
        True -> separa 'Disk (canto)' de 'Disk (cara)'. Son poblaciones
        visualmente muy distintas, asi que puede convenir para la CNN; para el
        resultado principal alineado con el resumen ("tres clases principales")
        conviene dejarlo en False.

    Columnas auxiliares: ``edge_on`` (bool) y ``etapas`` (numero de preguntas
    recorridas), utiles para el analisis de errores y para el filtro.
    """
    df = df.copy()
    n = len(df)

    # ---- Q1: lisa / con estructura / estrella-artefacto -------------------
    p1 = df[["Class1.1", "Class1.2", "Class1.3"]].to_numpy(dtype=float)
    idx1 = p1.argmax(axis=1)
    df["conf_q1"] = p1.max(axis=1)
    df["rama"] = np.array(["Smooth", "Features/Disk", CLASE_PUNTUAL])[idx1]
    m_fd = idx1 == 1

    # ---- Q2 (solo si Features/Disk): esta de canto? -----------------------
    p2 = df[["Class2.1", "Class2.2"]].to_numpy(dtype=float)
    edge_on = p2[:, 0] > p2[:, 1]
    conf_q2 = p2.max(axis=1)

    # ---- Q4 (solo si Features/Disk y NO de canto): brazos espirales? ------
    p4 = df[["Class4.1", "Class4.2"]].to_numpy(dtype=float)
    espiral = p4[:, 0] > p4[:, 1]
    conf_q4 = p4.max(axis=1)

    # ---- Etiqueta final ---------------------------------------------------
    etiqueta = np.array(df["rama"], dtype=object)
    confianza = df["conf_q1"].to_numpy(dtype=float).copy()
    etapas = np.ones(n, dtype=int)

    # Rama A: disco de canto -> Q1 x Q2
    m_canto = m_fd & edge_on
    etiqueta[m_canto] = "Disk (canto)" if separar_edge_on else "Disk"
    confianza[m_canto] = df["conf_q1"].to_numpy()[m_canto] * conf_q2[m_canto]
    etapas[m_canto] = 2

    # Rama B: disco de cara -> Q1 x Q2 x Q4
    m_cara = m_fd & ~edge_on
    etiqueta[m_cara & espiral] = "Spiral"
    etiqueta[m_cara & ~espiral] = "Disk (cara)" if separar_edge_on else "Disk"
    confianza[m_cara] = (
        df["conf_q1"].to_numpy()[m_cara] * conf_q2[m_cara] * conf_q4[m_cara]
    )
    etapas[m_cara] = 3

    df["label_grouped"] = etiqueta
    df["confianza"] = confianza
    df["edge_on"] = m_fd & edge_on
    df["etapas"] = etapas
    return df


def confianza_normalizada(df: pd.DataFrame) -> np.ndarray:
    """
    Media geometrica de las condicionadas: confianza ** (1 / etapas).

    El producto crudo penaliza injustamente a las ramas profundas: una espiral
    atraviesa 3 preguntas y una lisa solo 1, asi que un umbral plano sobre el
    producto elimina casi todas las espirales. La media geometrica pone las
    ramas en pie de igualdad.
    """
    return df["confianza"].to_numpy() ** (1.0 / df["etapas"].to_numpy())


def filtrar_por_confianza(
    df: pd.DataFrame,
    umbral: float = 0.6,
    excluir_puntuales: bool = True,
    normalizar_por_etapas: bool = True,
) -> pd.DataFrame:
    """Conjunto 'clean': galaxias donde el consenso de los voluntarios fue claro."""
    conf = (
        confianza_normalizada(df)
        if normalizar_por_etapas and "etapas" in df.columns
        else df["confianza"].to_numpy()
    )
    out = df[conf >= umbral]
    if excluir_puntuales:
        out = out[~out["label_grouped"].astype(str).str.startswith("Star")]
    return out.copy()


def etiqueta_binaria_puntual(df: pd.DataFrame) -> np.ndarray:
    """Tarea auxiliar: objeto puntual/artefacto vs. galaxia resuelta."""
    return (df["label_grouped"] == CLASE_PUNTUAL).to_numpy(dtype=int)


def subclase_smooth(df: pd.DataFrame) -> pd.Series:
    """Q7 dentro de Smooth: redonda / intermedia / cigarro. Util en la interfaz."""
    nombres = np.array(["Redonda", "Intermedia", "Cigarro"])
    idx = df[["Class7.1", "Class7.2", "Class7.3"]].to_numpy(dtype=float).argmax(axis=1)
    s = pd.Series(nombres[idx], index=df.index, name="subclase")
    s[df["label_grouped"] != "Smooth"] = pd.NA
    return s


def resumen(df: pd.DataFrame, umbral: float = 0.6, **kw) -> pd.DataFrame:
    """Distribucion de clases: conjunto completo vs. filtrado por confianza."""
    full = df["label_grouped"].value_counts()
    clean = filtrar_por_confianza(df, umbral, excluir_puntuales=False, **kw)
    clean = clean["label_grouped"].value_counts()
    tab = pd.DataFrame({"Completo": full, f"Conf>={umbral}": clean}).fillna(0).astype(int)
    tab["Retenido %"] = (tab.iloc[:, 1] / tab.iloc[:, 0].clip(lower=1) * 100).round(1)
    tab["% del limpio"] = (tab.iloc[:, 1] / max(tab.iloc[:, 1].sum(), 1) * 100).round(1)
    return tab


def diagnostico_arbol(df: pd.DataFrame) -> None:
    """Comprobacion de que el recorrido del arbol es coherente."""
    print(f"Total: {len(df)}")
    print("\nRama Q1:")
    print(df["rama"].value_counts().to_string())
    m = df["rama"] == "Features/Disk"
    print(f"\nDentro de Features/Disk ({int(m.sum())}):")
    print(f"  de canto (Q2.1 > Q2.2): {int(df.loc[m, 'edge_on'].sum())}")
    print(f"  de cara               : {int((~df.loc[m, 'edge_on']).sum())}")
    print("\nEtiqueta final:")
    print(df["label_grouped"].value_counts().to_string())
    print("\nConfianza media (producto crudo) por clase:")
    print(df.groupby("label_grouped")["confianza"].mean().round(3).to_string())
    print("\nConfianza media (normalizada por etapas):")
    tmp = df.assign(_c=confianza_normalizada(df))
    print(tmp.groupby("label_grouped")["_c"].mean().round(3).to_string())
    print("\nPreguntas recorridas por clase:")
    print(df.groupby("label_grouped")["etapas"].mean().round(2).to_string())
