"""
GalaxIA — interfaz interactiva.

    streamlit run app/Inicio.py

La app solo CARGA artefactos precomputados (ver scripts/export_artefactos.py).
No entrena ni recalcula nada.
"""
import json
import sys
from pathlib import Path

import streamlit as st

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

st.set_page_config(page_title="GalaxIA", page_icon="🌌", layout="wide")


# --------------------------------------------------------------------------- #
# Cargadores compartidos. Cache obligatorio: sin esto Streamlit recarga los
# modelos en cada interacción y la demo se vuelve inusable.
# --------------------------------------------------------------------------- #
@st.cache_data
def cargar_contenido():
    return json.loads((RAIZ / "app" / "contenido" / "tipologias.json").read_text())


@st.cache_data
def cargar_metricas():
    p = RAIZ / "reports" / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else {}


@st.cache_data
def cargar_galeria():
    p = RAIZ / "app" / "assets" / "galeria.json"
    return json.loads(p.read_text()) if p.exists() else {}


@st.cache_data
def cargar_muestra():
    import numpy as np
    p = RAIZ / "app" / "assets" / "muestra_test.npz"
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=True)
    return {"X": z["X"], "y": z["y"], "ids": z["ids"],
            "clases": [str(c) for c in z["clases"]]}


@st.cache_resource
def cargar_cnn():
    import os
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import keras
    for nombre in ["cnn_224.keras", "cnn_128.keras"]:
        p = RAIZ / "models" / nombre
        if p.exists():
            return keras.models.load_model(p), nombre
    return None, None


@st.cache_resource
def cargar_knn():
    import joblib
    p = RAIZ / "models" / "knn.pkl"
    return joblib.load(p) if p.exists() else None


# --------------------------------------------------------------------------- #
st.title("🌌 GalaxIA")
st.markdown("### Clasificación galáctica mediante inteligencia artificial")

st.markdown("""
Universidad Distrital Francisco José de Caldas
**Camila Pérez Angulo** · **Sergio Andrés Carrasquilla Hernández**
IX Congreso Colombiano de Astronomía y Astrofísica — COCOA 2026
""")

st.divider()

col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("""
#### El problema

El Sloan Digital Sky Survey registró cientos de miles de galaxias, muchas más de
las que un equipo de astrónomos puede clasificar a mano. El proyecto Galaxy Zoo
demostró que la clasificación visual por voluntarios produce resultados
comparables a los de profesionales, pero ese enfoque tampoco escala: cada nuevo
sondeo multiplica el volumen de datos.

Este trabajo entrena clasificadores automáticos sobre las clasificaciones de
Galaxy Zoo 2 y compara modelos clásicos de aprendizaje automático con redes
neuronales, siguiendo el árbol de decisión del cuestionario original.

#### Qué hay en esta interfaz

- **Tipologías morfológicas** — los tipos de galaxia, sus características
  físicas y ejemplos reales del SDSS.
- **Clasificador** — probar los modelos sobre galaxias reales, con mapas de
  atención y búsqueda de objetos similares.
- **Resultados** — el desempeño comparado de todos los modelos entrenados.
""")

with col2:
    metricas = cargar_metricas()
    if metricas:
        mejor = max(metricas.values(), key=lambda r: r["macro_f1"])
        st.metric("Mejor modelo", mejor["modelo"])
        a, b = st.columns(2)
        a.metric("F1-macro", f"{mejor['macro_f1']:.3f}")
        b.metric("Exactitud", f"{mejor['accuracy']:.3f}")
        st.caption(f"Evaluado sobre {mejor['n_test']:,} galaxias del conjunto "
                   "de prueba, nunca vistas durante el entrenamiento.")
        st.metric("Modelos comparados", len(metricas))
    else:
        st.info("Faltan las métricas. Ejecutar `scripts/export_artefactos.py`.")

st.divider()
st.caption("Datos: Galaxy Zoo 2 / Galaxy Zoo Challenge (61 578 galaxias del SDSS). "
           "Las imágenes son composiciones de las bandas g, r, i con estiramiento "
           "asinh, no fotometría calibrada.")
