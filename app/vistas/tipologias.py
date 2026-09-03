"""Tipologías: qué es una galaxia, las clases, otros tipos, Hubble y el árbol GZ2."""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from componentes import tema as T
from componentes import visual as V

RAIZ = T.raiz_proyecto(__file__)
T.aplicar_tema(st)


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

st.markdown("<div class='gx-centro'><h1>Tipologías morfológicas</h1></div>",
            unsafe_allow_html=True)

t1, t2, t3, t4, t5 = st.tabs([
    "Qué es una galaxia", "Clases del modelo", "Otros tipos",
    "Secuencia de Hubble", "Árbol de Galaxy Zoo",
])

# =========================================================================== #
with t1:
    Q = C["que_es_galaxia"]
    st.markdown(
        f"<div class='gx-centro'><p style='font-size:19px;line-height:1.85'>"
        f"{Q['definicion']}</p>"
        f"<p style='font-size:17px;color:#A9A2C4'>{Q['escala']}</p></div>",
        unsafe_allow_html=True)

    st.markdown("#### Anatomía de una galaxia espiral")
    T.render_svg(V.svg_partes_galaxia(), alto=660)

    partes = Q["partes"]
    for fila in range(0, len(partes), 3):
        cols = st.columns(3)
        for col, p in zip(cols, partes[fila:fila + 3]):
            col.markdown(T.tarjeta(p["nombre"], p["texto"]),
                         unsafe_allow_html=True)
        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    st.markdown(
        f"<div class='gx-centro' style='margin-top:26px'>"
        f"<div class='gx-card'><h5>Por qué importa la morfología</h5>"
        f"<p>{Q['por_que_morfologia']}</p></div></div>",
        unsafe_allow_html=True)

# =========================================================================== #
with t2:
    clases = list(C["clases_modelo"])
    st.markdown("<div class='gx-centro'></div>", unsafe_allow_html=True)
    st.markdown("<div class='gx-cuadros'>", unsafe_allow_html=True)
    elegida = st.radio(
        "Clase", clases, horizontal=True, label_visibility="collapsed",
        format_func=lambda c: C["clases_modelo"][c]["nombre"])
    st.markdown("</div>", unsafe_allow_html=True)

    info = C["clases_modelo"][elegida]

    ca, cb = st.columns([2, 3])
    with ca:
        T.render_svg(SVGS[info["svg"]](), alto=280)
    with cb:
        st.markdown(f"### {info['nombre']}")
        if info.get("hubble") and info["hubble"] != "—":
            st.markdown(f"<span class='gx-chip'>{info['hubble']}</span>",
                        unsafe_allow_html=True)
        st.write(info["resumen"])
        if info.get("nota_modelo"):
            st.info(info["nota_modelo"])

    # ---- Características físicas: todas visibles, en dos columnas ---------
    st.markdown("#### Características físicas")
    props = list(info["caracteristicas"].items())
    for fila in range(0, len(props), 2):
        cols = st.columns(2)
        for col, (k, d) in zip(cols, props[fila:fila + 2]):
            txt = d["texto"] if isinstance(d, dict) else d
            por = d.get("porque", "") if isinstance(d, dict) else ""
            col.markdown(T.tarjeta(k, txt, porque=por), unsafe_allow_html=True)
        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    # ---- Cómo reconocerla -------------------------------------------------
    st.markdown("#### Cómo reconocerla")
    cr1, cr2 = st.columns([2, 3])
    with cr1:
        T.render_svg(SVGS[info["svg"]](), alto=260)
    with cr2:
        for i, punto in enumerate(info.get("reconocer", []), 1):
            st.markdown(
                f"<div style='display:flex;gap:13px;margin-bottom:13px;"
                f"align-items:flex-start'>"
                f"<div style='background:linear-gradient(120deg,#7B4FBF,#4A7FD9);"
                f"color:#fff;border-radius:8px;width:28px;height:28px;"
                f"min-width:28px;text-align:center;font-weight:700;"
                f"font-size:15px;line-height:28px'>{i}</div>"
                f"<div style='font-size:17px'>{punto}</div></div>",
                unsafe_allow_html=True)

    if info.get("extra_barras"):
        st.markdown(T.tarjeta("Sobre las barras centrales", info["extra_barras"]),
                    unsafe_allow_html=True)

  

# =========================================================================== #
with t3:
    st.markdown(
        "<div class='gx-centro'><p style='font-size:18px'>El clasificador "
        "trabaja con tres clases, pero el universo tiene muchas más formas. "
        "Estas son las que quedan fuera del esquema y por qué.</p></div>",
        unsafe_allow_html=True)

    for tipo in C["otros_tipos"]:
        with st.container(border=True):
            st.markdown(f"### {tipo['nombre']}")
            st.markdown(
                f"<p style='font-size:18px;color:#B18CF0;font-weight:600'>"
                f"{tipo['resumen']}</p>", unsafe_allow_html=True)
            oa, ob = st.columns([3, 2])
            oa.write(tipo["detalle"])
            ob.markdown(
                f"<div class='gx-porque'><b>En Galaxy Zoo 2</b><br>"
                f"{tipo['en_gz2']}</div>", unsafe_allow_html=True)
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

# =========================================================================== #
with t4:
    H = C["hubble"]

    # ---- Línea de tiempo compacta y centrada ------------------------------
    st.markdown("<div class='gx-centro'><h4>Cómo llegamos hasta aquí</h4></div>",
                unsafe_allow_html=True)
    anios = [h["anio"] for h in H["historia"]]
    cw = st.columns([1, 6, 1])
    with cw[1]:
        st.markdown("<div class='gx-cuadros'>", unsafe_allow_html=True)
        sel_anio = st.select_slider("Hito", options=anios, value=anios[0],
                                    label_visibility="collapsed")
        st.markdown("</div>", unsafe_allow_html=True)

        # Relieve en cada parada
        puntos = "".join(
            f"<div style='flex:1;text-align:center'>"
            f"<div style='width:{18 if a == sel_anio else 13}px;"
            f"height:{18 if a == sel_anio else 13}px;margin:0 auto;"
            f"border-radius:50%;"
            f"background:{'linear-gradient(140deg,#B18CF0,#7FB2F0)' if a == sel_anio else '#2C2545'};"
            f"border:2px solid {'#E0B050' if a == sel_anio else '#4A4270'};"
            f"box-shadow:{'0 0 18px rgba(177,140,240,.85), inset 0 1px 2px rgba(255,255,255,.6)' if a == sel_anio else 'inset 0 1px 2px rgba(255,255,255,.18)'};"
            f"transition:all .35s ease'></div>"
            f"<div style='font-size:13px;margin-top:7px;"
            f"color:{'#E0B050' if a == sel_anio else '#6E6890'};"
            f"font-weight:{700 if a == sel_anio else 400}'>{a}</div></div>"
            for a in anios)
        st.markdown(
            f"<div style='display:flex;align-items:flex-start;"
            f"border-top:1px solid #3A3260;padding-top:12px;margin-top:-6px'>"
            f"{puntos}</div>", unsafe_allow_html=True)

        h = next(x for x in H["historia"] if x["anio"] == sel_anio)
        st.markdown(
            f"<div class='gx-card' style='margin-top:20px;text-align:left'>"
            f"<h5>{h['anio']} · {h['titulo']}</h5><p>{h['texto']}</p></div>",
            unsafe_allow_html=True)

    st.markdown("<div style='height:26px'></div>", unsafe_allow_html=True)

    # ---- Diapasón ---------------------------------------------------------
    st.markdown("<div class='gx-centro'><h4>El diagrama diapasón</h4></div>",
                unsafe_allow_html=True)
    tipos = ["E0", "E7", "S0", "Sa", "Sb", "Sc", "SBa", "SBb", "SBc"]
    cs = st.columns([1, 3, 1])
    with cs[1]:
        st.markdown("<div class='gx-cuadros'>", unsafe_allow_html=True)
        res = st.radio("Resaltar", ["(todos)"] + tipos, horizontal=True,
                       label_visibility="collapsed")
        st.markdown("</div>", unsafe_allow_html=True)
    T.render_svg(V.svg_diapason(None if res == "(todos)" else res), alto=520)
    st.warning(f"**Un malentendido persistente.** {H['malentendido']}")

    st.markdown("<div style='height:26px'></div>", unsafe_allow_html=True)

    # ---- Elipticidad ------------------------------------------------------
    e = H["elipticidad"]
    st.markdown(
        "<div class='gx-centro'><h4>El número de E0 a E7 mide el aplanamiento"
        "</h4></div>", unsafe_allow_html=True)
    T.render_svg(V.svg_fila_elipticas(), alto=400)

    ce = st.columns(3)
    ce[0].markdown(T.tarjeta(
        "La fórmula",
        "n = 10 × (1 − b/a), donde <b>a</b> es el semieje mayor y <b>b</b> el "
        "menor. Una E0 se ve circular (b/a = 1); una E7 es la más alargada "
        "del esquema (b/a = 0.3)."), unsafe_allow_html=True)
    ce[1].markdown(T.tarjeta(
        "¿Por qué no hay E8, E9 ni E10?", e["por_que_no_e8"]),
        unsafe_allow_html=True)
    ce[2].markdown(T.tarjeta(
        "Cuidado con la orientación", e["advertencia"]), unsafe_allow_html=True)

    st.markdown("<div style='height:26px'></div>", unsafe_allow_html=True)

    # ---- Criterios como tarjetas -----------------------------------------
    st.markdown(
        "<div class='gx-centro'><h4>Qué debe cumplir una galaxia para entrar "
        "en cada clase</h4></div>", unsafe_allow_html=True)
    crit = H["criterios"]
    for fila in range(0, len(crit), 3):
        cols = st.columns(3)
        for col, c_ in zip(cols, crit[fila:fila + 3]):
            filas = "".join(
                f"<div style='display:flex;justify-content:space-between;"
                f"padding:5px 0;border-bottom:1px solid rgba(177,140,240,.14);"
                f"font-size:15px'>"
                f"<span style='color:#A9A2C4'>{k}</span>"
                f"<span style='font-weight:600;text-align:right'>{v}</span></div>"
                for k, v in [("Bulbo", c_["bulbo"]), ("Brazos", c_["brazos"]),
                             ("Gas y polvo", c_["gas"]), ("Color", c_["color"]),
                             ("Concentración", c_["concentracion"])])
            col.markdown(
                f"<div class='gx-card'><h5>{c_['tipo']}</h5>{filas}</div>",
                unsafe_allow_html=True)
        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    # ---- Nomenclatura -----------------------------------------------------
    st.markdown("#### Nomenclatura")
    nom = H["nomenclatura"]
    for fila in range(0, len(nom), 2):
        cols = st.columns(2)
        for col, n_ in zip(cols, nom[fila:fila + 2]):
            col.markdown(
                f"<div style='display:flex;gap:14px;align-items:center;"
                f"background:rgba(123,79,191,.09);border:1px solid "
                f"rgba(177,140,240,.18);border-radius:12px;padding:12px 16px;"
                f"margin-bottom:10px'>"
                f"<code style='background:#2A1F52;color:#E0B050;padding:4px 12px;"
                f"border-radius:6px;min-width:82px;text-align:center;"
                f"font-size:15px'>{n_['simbolo']}</code>"
                f"<span style='font-size:16px'>{n_['significado']}</span></div>",
                unsafe_allow_html=True)

# =========================================================================== #
with t5:
    A = C["arbol_gz2"]
    st.markdown(f"<div class='gx-centro'><p style='font-size:18px'>"
                f"{A['descripcion']}</p></div>", unsafe_allow_html=True)

    # ---- Imagen del artículo, si está disponible --------------------------
    img_arbol = None
    for nombre in ["arbol_gz2.png", "arbol_gz2.jpg", "gz2_tree.png"]:
        p = RAIZ / "app" / "assets" / nombre
        if p.exists():
            img_arbol = p
            break

    if img_arbol:
        ci = st.columns([1, 12, 1])
        ci[1].image(str(img_arbol), use_container_width=True)
        st.caption(
            "Árbol de decisión del cuestionario de Galaxy Zoo 2. "
            "Fuente: Willett, K. W. et al. (2013), *Galaxy Zoo 2: detailed "
            "morphological classifications for 304,122 galaxies from the Sloan "
            "Digital Sky Survey*, MNRAS 435(4), 2835–2860. "
            "https://doi.org/10.1093/mnras/stt1458")
    else:
        st.warning(
            "**Falta la figura del árbol de decisión.**\n\n"
            "Guarde la imagen en la carpeta `app/assets/` con uno de estos "
            "nombres y aparecerá aquí automáticamente, ya con la cita de "
            "Willett et al. (2013):\n\n"
            "- `app/assets/arbol_gz2.png`  ← recomendado\n"
            "- `app/assets/arbol_gz2.jpg`\n"
            "- `app/assets/gz2_tree.png`\n\n"
            "Mientras tanto se muestra el esquema simplificado de las tres "
            "preguntas que usa este trabajo.")
        T.render_svg(V.svg_arbol_gz2(None), alto=600)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # ---- Recorrido interactivo -------------------------------------------
    st.markdown("<div class='gx-centro'><h4>Las tres preguntas que usa este "
                "trabajo</h4></div>", unsafe_allow_html=True)
    usadas = [q for q in A["preguntas"] if q["usada"]]
    cols = st.columns(3)
    for col, q in zip(cols, usadas):
        resp = "".join(f"<span class='gx-chip'>{r}</span>"
                       for r in q["respuestas"])
        col.markdown(
            f"<div class='gx-card'><h5>{q['id']}</h5>"
            f"<p style='font-size:17px'>{q['texto']}</p>{resp}"
            f"<div class='gx-porque'>{q['condicion']}</div></div>",
            unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.error(f"**Por qué el orden importa.** {A['advertencia']}")

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    st.markdown("#### Las once preguntas del cuestionario completo")
    st.caption("Este trabajo usa Q1, Q2 y Q4. El resto permitiría un desglose "
               "mucho más fino: barras, número de brazos, prominencia del bulbo.")
    tabla = pd.DataFrame([{
        "": "●" if q["usada"] else "○",
        "Pregunta": q["id"],
        "Texto": q["texto"],
        "Respuestas": " · ".join(q["respuestas"]),
        "Se formula": q["condicion"],
    } for q in A["preguntas"]])
    st.dataframe(tabla, use_container_width=True, hide_index=True)
