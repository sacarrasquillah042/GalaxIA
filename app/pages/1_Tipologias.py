"""Página 1: tipologías morfológicas, características físicas y ejemplos SDSS."""
import json
import sys
from pathlib import Path

import streamlit as st

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))

st.set_page_config(page_title="Tipologías — GalaxIA", page_icon="🌀", layout="wide")


@st.cache_data
def contenido():
    return json.loads((RAIZ / "app" / "contenido" / "tipologias.json").read_text())


@st.cache_data
def galeria():
    p = RAIZ / "app" / "assets" / "galeria.json"
    return json.loads(p.read_text()) if p.exists() else {}


C = contenido()
G = galeria()

st.title("🌀 Tipologías morfológicas")
st.markdown(
    "La morfología de una galaxia no es un rasgo estético: refleja su historia "
    "de formación, su contenido de gas y su dinámica interna. La secuencia de "
    "Hubble ordena esa variedad, aunque hoy se entiende como un continuo y no "
    "como una serie de cajas discretas."
)

# --------------------------------------------------------------------------- #
tab1, tab2, tab3 = st.tabs(
    ["Las tres clases del modelo", "Secuencia de Hubble", "El árbol de Galaxy Zoo"]
)

with tab1:
    clases = ["Smooth", "Disk", "Spiral", "Star/Artifact"]
    elegida = st.radio(
        "Clase", clases, horizontal=True,
        format_func=lambda c: C["clases_modelo"][c]["nombre"],
    )
    info = C["clases_modelo"][elegida]

    st.subheader(info["nombre"])
    if info.get("hubble") and info["hubble"] != "—":
        st.caption(f"Secuencia de Hubble: {info['hubble']}")
    st.write(info["resumen"])

    col_a, col_b = st.columns([3, 2])

    with col_a:
        st.markdown("#### Características físicas")
        for k, v in info["caracteristicas"].items():
            st.markdown(f"**{k}.** {v}")

    with col_b:
        if info.get("clave_visual"):
            st.info(f"**Cómo reconocerla.** {info['clave_visual']}")
        for campo, titulo in [
            ("por_que_se_clasifica_bien", "Por qué se clasifica bien"),
            ("por_que_es_dificil", "Por qué es difícil"),
            ("barras", "Sobre las barras"),
            ("nota_metodologica", "Nota metodológica"),
        ]:
            if info.get(campo):
                with st.expander(titulo):
                    st.write(info[campo])

    # ---- Galería de ejemplos reales -----------------------------------
    st.markdown("#### Ejemplos reales del SDSS")
    entradas = G.get(elegida, [])
    if not entradas:
        st.warning(
            "No hay galería para esta clase. Ejecutar "
            "`python scripts/export_artefactos.py`."
        )
    else:
        st.caption(
            f"{len(entradas)} galaxias donde el consenso de los voluntarios de "
            "Galaxy Zoo fue más claro. Imágenes recortadas a la región central."
        )
        base = RAIZ / "app" / "assets" / "galeria"
        for inicio in range(0, len(entradas), 6):
            cols = st.columns(6)
            for col, e in zip(cols, entradas[inicio:inicio + 6]):
                ruta = base / e["archivo"]
                if ruta.exists():
                    col.image(str(ruta), use_container_width=True)
                    etiqueta = f"conf. {e['confianza']:.2f}"
                    if e.get("edge_on"):
                        etiqueta += " · de canto"
                    col.caption(etiqueta)

with tab2:
    st.markdown(
        "Hubble ordenó las galaxias en 1936 según su apariencia. El diagrama se "
        "lee de izquierda a derecha, pero **no representa una secuencia evolutiva**: "
        "los nombres «tempranas» y «tardías» son un accidente histórico, no una "
        "descripción de edad."
    )
    for fam in ["Elíptica", "Lenticular", "Espiral", "Espiral barrada", "Irregular"]:
        items = [h for h in C["secuencia_hubble"] if h["familia"] == fam]
        if not items:
            continue
        st.markdown(f"**{fam}**")
        cols = st.columns(len(items))
        for col, h in zip(cols, items):
            col.markdown(f"`{h['tipo']}`")
            col.caption(h["desc"])

with tab3:
    a = C["arbol_gz2"]
    st.write(a["descripcion"])
    st.markdown("#### Preguntas utilizadas en este trabajo")
    for q in a["preguntas_usadas"]:
        with st.container(border=True):
            st.markdown(f"**{q['id']}** — {q['texto']}")
            st.caption(" · ".join(q["respuestas"]))
            if q.get("condicion"):
                st.caption(f"↳ {q['condicion']}")
    st.warning(a["advertencia"])
