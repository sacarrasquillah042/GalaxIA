"""
GalaxIA — interfaz interactiva.

    streamlit run app/Inicio.py

La app solo CARGA artefactos precomputados. No entrena ni recalcula nada.
"""
import json
import sys
from pathlib import Path

import numpy as np
import streamlit as st

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "app"))

from componentes import graficos as G  # noqa: E402
from componentes import visual as V  # noqa: E402

st.set_page_config(page_title="GalaxIA", page_icon="🌌", layout="wide")


# --------------------------------------------------------------------------- #
@st.cache_data
def cargar_metricas():
    p = RAIZ / "reports" / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else {}


@st.cache_data
def cargar_galeria():
    p = RAIZ / "app" / "assets" / "galeria.json"
    return json.loads(p.read_text()) if p.exists() else {}


# --------------------------------------------------------------------------- #
st.markdown(
    """<div style="text-align:center;padding:18px 0 6px 0">
      <div style="font-size:52px;font-weight:800;letter-spacing:-1px;
                  background:linear-gradient(90deg,#D4A032,#4A90D9);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent">
        GalaxIA</div>
      <div style="font-size:19px;color:#9AA5B5;margin-top:-4px">
        Clasificación galáctica mediante inteligencia artificial</div>
    </div>""",
    unsafe_allow_html=True,
)

M = cargar_metricas()
GAL = cargar_galeria()

# ---- Métricas destacadas --------------------------------------------------
if M:
    mejor = max(M.values(), key=lambda r: r["macro_f1"])
    c = st.columns(4)
    c[0].metric("Modelos comparados", len(M))
    c[1].metric("Mejor F1-macro", f"{mejor['macro_f1']:.4f}", mejor["modelo"])
    c[2].metric("Exactitud", f"{mejor['accuracy']:.4f}")
    c[3].metric("Galaxias en prueba", f"{mejor['n_test']:,}")

st.divider()

# ---- Tres tipos, animados -------------------------------------------------
st.markdown("#### Tres formas de ser una galaxia")
c1, c2, c3 = st.columns(3)
for col, (svg, titulo, pie) in zip(
    (c1, c2, c3),
    [(V.svg_eliptica(), "Elíptica",
      "Estrellas viejas en órbitas desordenadas. Roja, sin gas, muy concentrada."),
     (V.svg_espiral(), "Espiral",
      "Disco en rotación con brazos azules donde nacen estrellas nuevas."),
     (V.svg_disco_canto(), "Disco de canto",
      "El mismo disco visto de perfil: la banda de polvo oculta los brazos.")],
):
    with col:
        st.html(svg)
        st.markdown(f"**{titulo}**")
        st.caption(pie)

st.divider()

# ---- Resultados y datos ---------------------------------------------------
col_a, col_b = st.columns([3, 2])

with col_a:
    if M:
        st.plotly_chart(
            G.barras_modelos(M, "macro_f1",
                             "Desempeño de los nueve modelos entrenados"),
            use_container_width=True,
        )
        st.caption("Pase el cursor por cada barra para ver la exactitud. "
                   "Se usa F1-macro y no exactitud porque las clases están "
                   "desbalanceadas.")
    else:
        st.info("Ejecute `python scripts/export_artefactos.py` para ver los "
                "resultados.")

with col_b:
    if GAL:
        conteos = {}
        try:
            import sys as _s
            from galaxia import data, labels
            df = labels.construir_etiquetas(data.cargar_csv())
            u = data.DIR_PROC / "umbral.json"
            umbral = json.loads(u.read_text())["umbral"] if u.exists() else 0.6
            dfc = labels.filtrar_por_confianza(df, umbral=umbral)
            conteos = dfc["label_grouped"].value_counts().to_dict()
        except Exception:
            conteos = {k: len(v) for k, v in GAL.items()}
        st.plotly_chart(G.dona_clases(conteos), use_container_width=True)
        st.caption("Distribución tras filtrar por consenso de los voluntarios.")

st.divider()

# ---- Contexto -------------------------------------------------------------
izq, der = st.columns([3, 2])

with izq:
    st.markdown("""
#### El problema

El Sloan Digital Sky Survey registró cientos de miles de galaxias resueltas,
muchas más de las que un equipo de astrónomos puede clasificar a mano. El
proyecto **Galaxy Zoo** demostró que la clasificación visual por voluntarios
produce resultados comparables a los de profesionales, pero ese enfoque tampoco
escala: cada nuevo sondeo multiplica el volumen por órdenes de magnitud.

Este trabajo entrena clasificadores automáticos sobre las clasificaciones de
Galaxy Zoo 2 y compara aprendizaje automático clásico con redes neuronales,
siguiendo el árbol de decisión del cuestionario original.
""")

with der:
    st.markdown("""
#### Recorrido sugerido

**Tipologías morfológicas** — qué tipos de galaxia existen, cómo se
reconocen y cómo los organizó Hubble.

**Clasificador** — pruebe los modelos sobre galaxias reales del SDSS,
con mapas de atención y búsqueda de objetos similares.

**Resultados** — desempeño comparado, cómo leer cada métrica y qué
decisiones técnicas se tomaron.
""")
    st.page_link("pages/1_Tipologias.py", label="Ir a Tipologías", icon="🌀")
    st.page_link("pages/2_Clasificador.py", label="Ir al Clasificador", icon="🔭")
    st.page_link("pages/3_Resultados.py", label="Ir a Resultados", icon="📊")

st.divider()
st.markdown(
    """<div style="color:#7A8496;font-size:13px;line-height:1.7">
    <b>Camila Pérez Angulo</b> · <b>Sergio Andrés Carrasquilla Hernández</b><br>
    Universidad Distrital Francisco José de Caldas<br>
    IX Congreso Colombiano de Astronomía y Astrofísica — COCOA 2026<br><br>
    Datos: Galaxy Zoo 2 / Galaxy Zoo Challenge, 61 578 galaxias del SDSS.
    Las imágenes son composiciones de las bandas g, r, i con estiramiento
    asinh: no son fotometría calibrada.
    </div>""",
    unsafe_allow_html=True,
)
