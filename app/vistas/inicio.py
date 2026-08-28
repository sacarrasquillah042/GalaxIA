"""Inicio: portada con galaxia interactiva, y el resto al hacer scroll."""
import json
from pathlib import Path

import streamlit as st

from componentes import graficos as G
from componentes import tema as T
from componentes import visual as V

RAIZ = Path(__file__).resolve().parents[2]
T.aplicar_tema(st)


@st.cache_data
def metricas():
    p = RAIZ / "reports" / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else {}


@st.cache_data
def contenido():
    return json.loads((RAIZ / "app" / "contenido" / "tipologias.json").read_text())


M = metricas()
C = contenido()

# =========================== HERO ========================================== #
st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
V.render_svg(V.svg_hero(), alto=400)

st.markdown(
    """<div class="gx-centro" style="margin-top:-14px">
      <div style="font-size:64px;font-weight:800;letter-spacing:-2px;line-height:1.05;
                  background:linear-gradient(100deg,#B18CF0,#7FB2F0 55%,#F07B8C);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent">
        GalaxIA</div>
      <div style="font-size:24px;color:#E9E6F5;margin-top:6px;font-weight:500">
        Clasificación galáctica mediante inteligencia artificial</div>
      <div style="font-size:18px;color:#A9A2C4;margin-top:22px;line-height:1.9">
        <b style="color:#B18CF0">Camila Pérez Angulo</b>
        &nbsp;·&nbsp;
        <b style="color:#B18CF0">Sergio Andrés Carrasquilla Hernández</b><br>
        Universidad Distrital Francisco José de Caldas<br>
        <span style="font-size:16px">IX Congreso Colombiano de Astronomía
        y Astrofísica — COCOA 2026</span>
      </div>
      <div style="margin-top:34px;color:#7A7396;font-size:14px;
                  letter-spacing:.18em;text-transform:uppercase">
        desplácese para continuar</div>
    </div>""",
    unsafe_allow_html=True,
)

st.markdown("<div style='height:70px'></div>", unsafe_allow_html=True)
st.divider()

# =========================== MÉTRICAS ====================================== #
if M:
    mejor = max(M.values(), key=lambda r: r["macro_f1"])
    st.markdown("<div class='gx-centro'><h3>Resultados en un vistazo</h3></div>",
                unsafe_allow_html=True)
    c = st.columns(4)
    c[0].metric("Modelos comparados", len(M))
    c[1].metric("Mejor F1-macro", f"{mejor['macro_f1']:.4f}", mejor["modelo"])
    c[2].metric("Exactitud", f"{mejor['accuracy']:.4f}")
    c[3].metric("Galaxias en prueba", f"{mejor['n_test']:,}")

st.markdown("<div style='height:34px'></div>", unsafe_allow_html=True)

# =========================== TRES TIPOS ==================================== #
st.markdown("<div class='gx-centro'><h3>Tres formas de ser una galaxia</h3></div>",
            unsafe_allow_html=True)
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
        T.render_svg(svg, alto=250)
        st.markdown(
            f"<div style='text-align:center'><b style='font-size:19px;"
            f"color:#B18CF0'>{titulo}</b><br>"
            f"<span style='font-size:15px;color:#A9A2C4'>{pie}</span></div>",
            unsafe_allow_html=True)

st.markdown("<div style='height:34px'></div>", unsafe_allow_html=True)
st.divider()

# =========================== GALAXY ZOO ==================================== #
GZ = C["galaxy_zoo"]
st.markdown(f"<div class='gx-centro'><h3>{GZ['titulo']}</h3></div>",
            unsafe_allow_html=True)
gz1, gz2 = st.columns([3, 2])
with gz1:
    st.write(GZ["que_es"])
    st.markdown(T.cita(GZ["nasa"], GZ["fuente_nombre"], GZ["fuente_url"]),
                unsafe_allow_html=True)
    st.write(GZ["nasa_ia"])
with gz2:
    st.markdown(T.tarjeta("Por qué funciona", GZ["por_que_funciona"]),
                unsafe_allow_html=True)
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    b1, b2 = st.columns(2)
    b1.link_button("Clasificar galaxias", GZ["plataforma"],
                   use_container_width=True, type="primary")
    b2.link_button("Datos públicos", GZ["datos"], use_container_width=True)

st.markdown("<div style='height:34px'></div>", unsafe_allow_html=True)
st.divider()

# =========================== GRÁFICAS ====================================== #
ga, gb = st.columns([3, 2])
with ga:
    if M:
        st.plotly_chart(G.barras_modelos(M, "macro_f1",
                                         "Desempeño de los modelos entrenados"),
                        use_container_width=True)
        st.caption("Pase el cursor por cada barra para ver la exactitud. Se usa "
                   "F1-macro porque las clases están desbalanceadas.")
    else:
        st.info("Ejecute `python scripts/export_artefactos.py` para ver los "
                "resultados.")
with gb:
    try:
        from galaxia import data, labels
        df = labels.construir_etiquetas(data.cargar_csv())
        u = data.DIR_PROC / "umbral.json"
        umbral = json.loads(u.read_text())["umbral"] if u.exists() else 0.6
        conteos = labels.filtrar_por_confianza(
            df, umbral=umbral)["label_grouped"].value_counts().to_dict()
        st.plotly_chart(G.dona_clases(conteos), use_container_width=True)
        st.caption("Distribución tras filtrar por consenso de los voluntarios.")
    except Exception:
        st.empty()

st.divider()

# =========================== CONTEXTO ====================================== #
st.markdown(
    """<div class="gx-centro">
    <h3>El problema de escala</h3>
    <p style="font-size:18px;line-height:1.85;color:#C9C3E0">
    El Sloan Digital Sky Survey registró cientos de miles de galaxias resueltas,
    muchas más de las que un equipo de astrónomos puede clasificar a mano.
    Galaxy Zoo demostró que la clasificación visual por voluntarios produce
    resultados comparables a los de profesionales, pero ese enfoque tampoco
    escala: cada nuevo sondeo multiplica el volumen por órdenes de magnitud.
    </p>
    <p style="font-size:18px;line-height:1.85;color:#C9C3E0">
    Este trabajo entrena clasificadores automáticos sobre las clasificaciones de
    Galaxy Zoo 2 y compara aprendizaje automático clásico con redes neuronales,
    siguiendo el árbol de decisión del cuestionario original.
    </p></div>""",
    unsafe_allow_html=True,
)

st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)
st.markdown(
    """<div style="text-align:center;color:#6E6890;font-size:14px;line-height:1.8">
    Datos: Galaxy Zoo 2 / Galaxy Zoo Challenge, 61 578 galaxias del SDSS.<br>
    Las imágenes son composiciones de las bandas g, r, i con estiramiento asinh:
    no son fotometría calibrada.
    </div>""",
    unsafe_allow_html=True,
)
