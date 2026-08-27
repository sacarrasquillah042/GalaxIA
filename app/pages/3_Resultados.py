"""Página 3: resultados comparados y metodología. Todo sale de metrics.json."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))

st.set_page_config(page_title="Resultados — GalaxIA", page_icon="📊", layout="wide")


@st.cache_data
def metricas():
    p = RAIZ / "reports" / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else {}


M = metricas()
st.title("Resultados")

if not M:
    st.error("Falta `reports/metrics.json`.")
    st.stop()

# --------------------------------------------------------------------------- #
tab1, tab2, tab3 = st.tabs(["Comparación", "Por modelo", "Metodología"])

with tab1:
    st.markdown("#### Todos los modelos sobre el mismo conjunto de prueba")
    st.caption("Mismos índices congelados para todos los modelos: sin eso, la "
               "comparación no significaría nada.")

    tabla = pd.DataFrame([{
        "Modelo": r["modelo"],
        "Exactitud": r["accuracy"],
        "F1-macro": r["macro_f1"],
        "AUC-ROC": r["roc_auc_macro"],
        "AP-macro": r["avg_precision_macro"],
        "Entren. (s)": r["t_entrenamiento_s"],
        "Inferencia (s)": r["t_inferencia_s"],
    } for r in M.values()]).sort_values("F1-macro", ascending=False)

    st.dataframe(
        tabla.style.format({
            "Exactitud": "{:.4f}", "F1-macro": "{:.4f}", "AUC-ROC": "{:.4f}",
            "AP-macro": "{:.4f}", "Entren. (s)": "{:.1f}", "Inferencia (s)": "{:.2f}",
        }).background_gradient(subset=["F1-macro"], cmap="Blues"),
        use_container_width=True, hide_index=True,
    )

    st.info(
        "**Por qué F1-macro y no exactitud.** Las clases están desbalanceadas, "
        "así que un modelo que ignore la clase minoritaria puede alcanzar una "
        "exactitud alta sin ser útil. El F1-macro pesa las tres clases por igual "
        "y delata ese comportamiento."
    )

    st.markdown("#### Exactitud frente a coste de inferencia")
    st.caption("KNN no entrena pero predice lento; la CNN es lo contrario. "
               "Para un sondeo de millones de objetos, esa diferencia importa "
               "tanto como la exactitud.")
    disp = tabla.dropna(subset=["Inferencia (s)"])
    if len(disp):
        st.scatter_chart(disp, x="Inferencia (s)", y="F1-macro", size=None)

with tab2:
    nombre = st.selectbox("Modelo", sorted(M, key=lambda k: -M[k]["macro_f1"]))
    r = M[nombre]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Exactitud", f"{r['accuracy']:.4f}")
    c2.metric("F1-macro", f"{r['macro_f1']:.4f}")
    c3.metric("AUC-ROC", f"{r['roc_auc_macro']:.4f}")
    c4.metric("AP-macro", f"{r['avg_precision_macro']:.4f}")

    st.markdown("#### Métricas por clase")
    st.dataframe(
        pd.DataFrame(r["por_clase"]).T.style.format("{:.4f}", na_rep="—"),
        use_container_width=True,
    )
    st.caption("Especificidad, valor predictivo negativo y media geométrica "
               "complementan a precisión y recall cuando las clases están "
               "desbalanceadas.")

    st.markdown("#### Matriz de confusión")
    cm = np.array(r["matriz_confusion"], dtype=float)
    cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    st.dataframe(
        pd.DataFrame(cmn, index=[f"real: {c}" for c in r["clases"]],
                     columns=[f"pred: {c}" for c in r["clases"]])
        .style.format("{:.3f}").background_gradient(cmap="Blues"),
        use_container_width=True,
    )

    slug = "".join(ch if ch.isalnum() else "_" for ch in nombre.lower()).strip("_")
    figs = [("roc", "Curvas ROC"), ("pr", "Curvas precisión-recall"),
            ("cm", "Matriz de confusión"), ("hist", "Curvas de entrenamiento")]
    disponibles = [(p, t) for p, t in figs
                   if (RAIZ / "reports" / "figures" / f"{p}_{slug}.png").exists()]
    if disponibles:
        st.markdown("#### Figuras")
        cols = st.columns(min(len(disponibles), 2))
        for k, (p, t) in enumerate(disponibles):
            cols[k % len(cols)].image(
                str(RAIZ / "reports" / "figures" / f"{p}_{slug}.png"), caption=t)

with tab3:
    st.markdown("""
#### Datos

Galaxy Zoo Challenge: **61 578 galaxias** del SDSS con clasificaciones ponderadas
de voluntarios distribuidas en 37 respuestas morfológicas.

#### Etiquetado

Las 37 respuestas no son independientes: forman un árbol donde cada pregunta se
formula solo si la anterior lo justifica. La etiqueta final se obtiene siguiendo
la rama dominante y la confianza es el producto de las probabilidades
condicionadas recorridas.

Como las ramas tienen profundidades distintas —una elíptica atraviesa una
pregunta y una espiral tres—, el producto crudo penalizaría sistemáticamente a
las ramas profundas. Se usa por eso la media geométrica, que pone todas las
ramas en pie de igualdad.

#### Preprocesamiento

- Recorte de la región central: la galaxia ocupa aproximadamente el cuadrado
  central de 207×207 dentro de la imagen de 424×424. El resto es cielo y objetos
  vecinos.
- Color conservado. El índice de color es el segundo discriminante morfológico
  más fuerte después de la concentración.

#### Control de fugas

Todos los transformadores (escalado, PCA) se ajustan exclusivamente con el
conjunto de entrenamiento. Las particiones están congeladas en un archivo y son
idénticas para todos los modelos.

#### Aumentación

Rotaciones completas de 360° y reflexiones: las galaxias no tienen orientación
privilegiada en el cielo, así que estas transformaciones son físicamente
válidas. El color **no** se altera, porque es señal física real.

#### Advertencia sobre los parámetros fotométricos

Las imágenes son composiciones RGB de las bandas g, r, i con estiramiento asinh
no lineal. Los colores e índices de concentración calculados aquí son **proxies
derivados de la imagen**, no fotometría SDSS calibrada.
""")
