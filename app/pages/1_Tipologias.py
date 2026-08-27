"""Página 1: tipologías morfológicas, secuencia de Hubble y árbol GZ2."""
import json
import sys
from pathlib import Path

import streamlit as st

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "app"))

from componentes import visual as V  # noqa: E402

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
SVGS = {"eliptica": V.svg_eliptica, "espiral": V.svg_espiral,
        "disco_canto": V.svg_disco_canto, "puntual": V.svg_puntual}

st.title("🌀 Tipologías morfológicas")
st.markdown(
    "La morfología de una galaxia no es un rasgo estético: refleja su historia "
    "de formación, su contenido de gas y su dinámica interna."
)

t1, t2, t3, t4 = st.tabs([
    "Las clases del modelo", "Otros tipos de galaxia",
    "Secuencia de Hubble", "El árbol de Galaxy Zoo",
])

# =========================================================================== #
with t1:
    clases = list(C["clases_modelo"])
    elegida = st.radio(
        "Clase", clases, horizontal=True, label_visibility="collapsed",
        format_func=lambda c: C["clases_modelo"][c]["nombre"],
    )
    info = C["clases_modelo"][elegida]

    ca, cb = st.columns([2, 3])
    with ca:
        st.html(SVGS[info["svg"]]())
    with cb:
        st.subheader(info["nombre"])
        if info.get("hubble") and info["hubble"] != "—":
            st.caption(f"Secuencia de Hubble: {info['hubble']}")
        st.write(info["resumen"])
        if info.get("nota_modelo"):
            st.info(info["nota_modelo"])

    # ---- Características físicas interactivas -----------------------------
    st.markdown("#### Características físicas")
    st.caption("Seleccione una propiedad para ver el dato y su justificación física.")

    props = list(info["caracteristicas"])
    sel = st.segmented_control("Propiedad", props, default=props[0],
                               label_visibility="collapsed") or props[0]
    d = info["caracteristicas"][sel]
    with st.container(border=True):
        st.markdown(f"**{sel}**")
        st.write(d["texto"] if isinstance(d, dict) else d)
        if isinstance(d, dict) and d.get("porque"):
            st.markdown(
                f"<div style='color:#9AA5B5;border-left:3px solid #D4A032;"
                f"padding-left:12px;margin-top:8px'><b>¿Por qué?</b> "
                f"{d['porque']}</div>", unsafe_allow_html=True)

    # ---- Cómo reconocerla -------------------------------------------------
    st.markdown("#### Cómo reconocerla")
    cr1, cr2 = st.columns([2, 3])
    with cr1:
        st.html(SVGS[info["svg"]]())
    with cr2:
        for i, punto in enumerate(info.get("reconocer", []), 1):
            st.markdown(
                f"<div style='display:flex;gap:10px;margin-bottom:9px'>"
                f"<div style='background:#D4A032;color:#0B0E17;border-radius:50%;"
                f"width:22px;height:22px;min-width:22px;text-align:center;"
                f"font-weight:700;font-size:13px;line-height:22px'>{i}</div>"
                f"<div>{punto}</div></div>", unsafe_allow_html=True)

    if info.get("extra_barras"):
        with st.expander("Sobre las barras centrales"):
            st.write(info["extra_barras"])

    # ---- Galería ----------------------------------------------------------
    st.markdown("#### Ejemplos reales del SDSS")
    entradas = G.get(elegida, [])
    if not entradas:
        st.warning("Ejecute `python scripts/export_artefactos.py` para generar "
                   "la galería.")
    else:
        st.caption(f"{len(entradas)} galaxias donde el consenso de los "
                   "voluntarios de Galaxy Zoo fue más claro. "
                   "Imágenes recortadas a la región central.")
        base = RAIZ / "app" / "assets" / "galeria"
        for ini in range(0, len(entradas), 6):
            cols = st.columns(6)
            for col, e in zip(cols, entradas[ini:ini + 6]):
                r = base / e["archivo"]
                if r.exists():
                    col.image(str(r), use_container_width=True)
                    et = f"conf. {e['confianza']:.2f}"
                    if e.get("edge_on"):
                        et += " · de canto"
                    col.caption(et)
    st.caption("Crédito: Sloan Digital Sky Survey (SDSS), bandas g, r, i. "
               "Subconjunto público del Galaxy Zoo Challenge.")

# =========================================================================== #
with t2:
    st.markdown(
        "El clasificador trabaja con tres clases, pero el universo tiene muchas "
        "más formas. Estas son las que quedan fuera del esquema y por qué."
    )
    for tipo in C["otros_tipos"]:
        with st.expander(tipo["nombre"]):
            st.markdown(f"**{tipo['resumen']}**")
            st.write(tipo["detalle"])
            st.info(f"**En Galaxy Zoo 2:** {tipo['en_gz2']}")

# =========================================================================== #
with t3:
    H = C["hubble"]

    st.markdown("#### Cómo llegamos hasta aquí")
    hitos = [f"{h['anio']} — {h['titulo']}" for h in H["historia"]]
    idx = st.select_slider("Hito", options=list(range(len(hitos))),
                           format_func=lambda i: H["historia"][i]["anio"],
                           label_visibility="collapsed")
    h = H["historia"][idx]
    with st.container(border=True):
        st.markdown(f"### {h['anio']} · {h['titulo']}")
        st.write(h["texto"])

    st.divider()

    st.markdown("#### El diagrama diapasón")
    tipos = ["E0", "E7", "S0", "Sa", "Sb", "Sc", "SBa", "SBb", "SBc"]
    res = st.selectbox("Resaltar un tipo", ["(todos)"] + tipos)
    st.html(V.svg_diapason(None if res == "(todos)" else res))
    st.warning(f"**Un malentendido persistente.** {H['malentendido']}")

    st.divider()

    # ---- Elipticidad ------------------------------------------------------
    st.markdown("#### ¿Qué significa el número en E0, E1… E7?")
    ce1, ce2 = st.columns([2, 3])
    with ce1:
        n = st.slider("Elipticidad n", 0, 7, 3)
        st.html(V.svg_elipticidad(n))
    with ce2:
        e = H["elipticidad"]
        st.latex(r"n = 10 \times \left(1 - \frac{b}{a}\right)")
        st.write(e["explicacion"])
        st.error(f"**¿Y una E8 o E9?** {e['por_que_no_e8']}")
        st.caption(e["advertencia"])

    st.divider()

    # ---- Criterios --------------------------------------------------------
    st.markdown("#### Qué debe cumplir una galaxia para entrar en cada clase")
    import pandas as pd
    tabla = pd.DataFrame(H["criterios"]).rename(columns={
        "tipo": "Tipo", "bulbo": "Bulbo", "brazos": "Brazos",
        "gas": "Gas y polvo", "color": "Color", "concentracion": "Concentración"})
    st.dataframe(tabla, use_container_width=True, hide_index=True)

    st.markdown("#### Nomenclatura")
    for n_ in H["nomenclatura"]:
        st.markdown(
            f"<div style='display:flex;gap:14px;margin-bottom:7px'>"
            f"<code style='background:#1E3A6E;color:#FFD98A;padding:2px 9px;"
            f"border-radius:4px;min-width:74px;text-align:center'>"
            f"{n_['simbolo']}</code><div>{n_['significado']}</div></div>",
            unsafe_allow_html=True)

# =========================================================================== #
with t4:
    A = C["arbol_gz2"]
    st.write(A["descripcion"])
    st.latex(r"P(\text{hoja}) = \prod_{q \in \text{rama}} "
             r"P(\text{respuesta}_q \mid \text{padre})")

    usadas = [q["id"] for q in A["preguntas"] if q["usada"]]
    st.markdown("#### Recorra el árbol")
    st.caption("Seleccione una pregunta para resaltarla en el esquema.")
    qsel = st.radio("Pregunta", usadas, horizontal=True,
                    label_visibility="collapsed")
    st.html(V.svg_arbol_gz2(qsel))

    q = next(x for x in A["preguntas"] if x["id"] == qsel)
    with st.container(border=True):
        st.markdown(f"**{q['id']} — {q['texto']}**")
        cols = st.columns(len(q["respuestas"]))
        for col, r in zip(cols, q["respuestas"]):
            col.markdown(
                f"<div style='background:#1E3A6E;border:1px solid #4A5A78;"
                f"border-radius:6px;padding:8px;text-align:center;font-size:13px'>"
                f"{r}</div>", unsafe_allow_html=True)
        st.caption(f"Condición: {q['condicion']}")

    st.error(f"**Por qué el orden importa.** {A['advertencia']}")

    st.divider()
    st.markdown("#### Las once preguntas del cuestionario completo")
    st.caption("Este trabajo usa Q1, Q2 y Q4. El resto permitiría un desglose "
               "mucho más fino: barras, número de brazos, prominencia del bulbo.")
    for q in A["preguntas"]:
        marca = "✅" if q["usada"] else "○"
        with st.expander(f"{marca}  {q['id']} — {q['texto']}"):
            st.write("**Respuestas:** " + " · ".join(q["respuestas"]))
            st.caption(f"Condición: {q['condicion']}")
