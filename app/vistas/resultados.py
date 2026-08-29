"""Resultados: comparación, métricas explicadas y metodología."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from componentes import graficos as G
from componentes import tema as T
from componentes import visual as V

RAIZ = T.raiz_proyecto(__file__)
T.aplicar_tema(st)


@st.cache_data
def metricas():
    p = RAIZ / "reports" / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else {}


@st.cache_data
def contenido():
    return json.loads((RAIZ / "app" / "contenido" / "tipologias.json").read_text())


@st.cache_data
def mcnemar():
    p = RAIZ / "reports" / "mcnemar.json"
    return json.loads(p.read_text()) if p.exists() else []


def cientifica(v, umbral=1e-3):
    """Notación científica cuando el valor es muy pequeño o muy grande."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return v
    if x != 0 and (abs(x) < umbral or abs(x) >= 1e5):
        m, e = f"{x:.2e}".split("e")
        return f"{m} × 10^{int(e)}"
    return f"{x:.4f}" if abs(x) < 1 else f"{x:,.2f}"


M = metricas()
C = contenido()
MC = mcnemar()

st.markdown("<div class='gx-centro'><h1>Resultados</h1></div>",
            unsafe_allow_html=True)
if not M:
    st.error("Falta `reports/metrics.json`.")
    st.stop()

t1, t2, t3, t4 = st.tabs(
    ["Comparación", "Por modelo", "Guía de métricas", "Metodología"])

# =========================================================================== #
with t1:
    st.markdown("#### Los modelos sobre el mismo conjunto de prueba")
    cm = st.columns([1, 3, 1])
    with cm[1]:
        st.markdown("<div class='gx-cuadros'>", unsafe_allow_html=True)
        met = st.radio(
            "Métrica",
            ["macro_f1", "accuracy", "roc_auc_macro", "avg_precision_macro"],
            horizontal=True, label_visibility="collapsed",
            format_func=lambda k: {"macro_f1": "F1-macro", "accuracy": "Exactitud",
                                   "roc_auc_macro": "AUC-ROC",
                                   "avg_precision_macro": "AP-macro"}[k])
        st.markdown("</div>", unsafe_allow_html=True)

    st.plotly_chart(G.barras_modelos(M, met), use_container_width=True)

    tabla = pd.DataFrame([{
        "Modelo": r["modelo"], "Exactitud": r["accuracy"],
        "F1-macro": r["macro_f1"], "AUC-ROC": r["roc_auc_macro"],
        "AP-macro": r["avg_precision_macro"],
        "Entren. (s)": r["t_entrenamiento_s"],
        "Inferencia (s)": r["t_inferencia_s"],
    } for r in M.values()]).sort_values("F1-macro", ascending=False)
    st.dataframe(
        tabla.style.format({
            "Exactitud": "{:.4f}", "F1-macro": "{:.4f}", "AUC-ROC": "{:.4f}",
            "AP-macro": "{:.4f}", "Entren. (s)": "{:.1f}",
            "Inferencia (s)": "{:.2f}"}),
        use_container_width=True, hide_index=True)

    st.info("**Por qué F1-macro y no exactitud.** Las clases están "
            "desbalanceadas, así que un modelo que ignore la clase minoritaria "
            "puede alcanzar exactitud alta sin ser útil. El F1-macro pesa las "
            "tres clases por igual y delata ese comportamiento.")

    st.divider()

    # ---- Perfil comparado: gráfica izquierda, selectores derecha ----------
    st.markdown("#### Perfil comparado")
    pg, ps = st.columns([3, 1], gap="large")
    with ps:
        st.markdown("<div style='height:52px'></div>", unsafe_allow_html=True)
        sel = st.multiselect(
            "Modelos a superponer", sorted(M),
            default=sorted(M, key=lambda k: -M[k]["macro_f1"])[:3])
        st.caption("Cada eje es una métrica distinta. Superponer modelos "
                   "muestra si uno domina en todo o solo en algunos aspectos.")
    with pg:
        if sel:
            st.plotly_chart(G.radar_modelos(M, sel), use_container_width=True)
        else:
            st.info("Seleccione al menos un modelo.")

    st.divider()
    st.markdown("#### Exactitud frente a coste de inferencia")
    st.plotly_chart(G.coste_vs_exactitud(M), use_container_width=True)
    st.caption("Eje horizontal en escala logarítmica. El **tamaño** de cada "
               "punto representa el coste de entrenamiento. KNN no entrena pero "
               "predice lento; la CNN es lo contrario. Para clasificar millones "
               "de objetos, esa diferencia importa tanto como la exactitud.")

    if MC:
        st.divider()
        st.markdown("#### ¿Las diferencias son estadísticamente reales?")
        st.caption("Prueba de McNemar sobre las mismas galaxias: cuenta en "
                   "cuántos casos un modelo acierta y el otro falla.")
        dmc = pd.DataFrame(MC)
        for col in dmc.columns:
            if "p" in col.lower() and "valor" in col.lower():
                dmc[col] = dmc[col].map(lambda v: cientifica(v, 1e-3))
            elif "chi" in col.lower():
                dmc[col] = dmc[col].map(lambda v: cientifica(v))
        st.dataframe(dmc, use_container_width=True, hide_index=True)
        st.caption("Los p-valores se muestran en notación científica: "
                   "5.12 × 10⁻⁵³ significa que la probabilidad de observar esa "
                   "diferencia por azar es prácticamente nula.")

# =========================================================================== #
with t2:
    cs = st.columns([1, 2, 1])
    with cs[1]:
        nombre = st.selectbox("Modelo",
                              sorted(M, key=lambda k: -M[k]["macro_f1"]))
    r = M[nombre]

    c = st.columns(4)
    c[0].metric("Exactitud", f"{r['accuracy']:.4f}")
    c[1].metric("F1-macro", f"{r['macro_f1']:.4f}")
    c[2].metric("AUC-ROC", f"{r['roc_auc_macro']:.4f}")
    c[3].metric("AP-macro", f"{r['avg_precision_macro']:.4f}")

    st.markdown("#### Matriz de confusión")
    cm1, cm2 = st.columns([3, 2], gap="large")
    with cm1:
        st.plotly_chart(
            G.matriz_confusion(r["matriz_confusion"], r["clases"], True),
            use_container_width=True)
    with cm2:
        g = C["como_leer"]["matriz_confusion"]
        st.markdown(f"**{g['titulo']}**")
        for i, p in enumerate(g["pasos"], 1):
            st.markdown(f"{i}. {p}")
        st.info(g["interpretar"])

    st.markdown("#### Métricas por clase")
    st.dataframe(
        pd.DataFrame(r["por_clase"]).T.style.format("{:.4f}", na_rep="—"),
        use_container_width=True)

    slug = "".join(ch if ch.isalnum() else "_" for ch in nombre.lower()).strip("_")
    figs = [("roc", "Curvas ROC"), ("pr", "Curvas precisión-recall"),
            ("hist", "Curvas de entrenamiento")]
    disp = [(p, t) for p, t in figs
            if (RAIZ / "reports" / "figures" / f"{p}_{slug}.png").exists()]
    if disp:
        st.markdown("#### Curvas de este modelo")
        cols = st.columns(min(len(disp), 2))
        for k, (p, t) in enumerate(disp):
            cols[k % len(cols)].image(
                str(RAIZ / "reports" / "figures" / f"{p}_{slug}.png"), caption=t)

# =========================================================================== #
with t3:
    st.markdown("#### Qué mide cada métrica")
    st.caption("VP = verdaderos positivos · VN = verdaderos negativos · "
               "FP = falsos positivos · FN = falsos negativos")

    items = list(C["metricas"].items())
    for fila in range(0, len(items), 2):
        cols = st.columns(2)
        for col, (k, m) in zip(cols, items[fila:fila + 2]):
            aviso = ""
            if m.get("cuando_importa"):
                aviso = (f"<div class='gx-porque'><b>Cuándo importa.</b> "
                         f"{m['cuando_importa']}</div>")
            if m.get("cuando_engana"):
                aviso += (f"<div class='gx-porque' style='border-color:#D9455F'>"
                          f"<b>Cuándo engaña.</b> {m['cuando_engana']}</div>")
            col.markdown(
                f"<div class='gx-card'><h5>{m['nombre']}</h5>"
                f"<div class='gx-sub'><code>{m['formula']}</code></div>"
                f"<p>{m['que_mide']}</p>{aviso}</div>", unsafe_allow_html=True)
        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    st.divider()
    st.markdown("#### Cómo leer las curvas")
    for clave, tipo in [("roc", "roc"), ("pr", "pr")]:
        g = C["como_leer"][clave]
        a, b = st.columns([3, 2], gap="large")
        with a:
            st.plotly_chart(G.curva_pedagogica(tipo), use_container_width=True)
            st.caption("Curvas ilustrativas, no datos del proyecto.")
        with b:
            st.markdown(f"**{g['titulo']}**")
            for i, p in enumerate(g["pasos"], 1):
                st.markdown(f"{i}. {p}")
            st.warning(g["cuidado"])
        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

# =========================================================================== #
with t4:
    MT = C["metodologia"]
    st.markdown("<div class='gx-centro'><h3>Del telescopio al modelo</h3>"
                "<p style='color:#A9A2C4'>Cada decisión del preprocesamiento "
                "tiene una justificación física y una algorítmica. "
                "Aquí están ambas.</p></div>", unsafe_allow_html=True)

    # ---- Bandas -----------------------------------------------------------
    B = MT["bandas"]
    st.markdown("#### ① Las bandas fotométricas g, r, i")
    st.write(B["intro"])
    T.render_svg(V.svg_bandas(), alto=250)
    st.dataframe(
        pd.DataFrame(B["detalle"]).rename(columns={
            "banda": "Banda", "rango": "Longitud de onda", "region": "Región",
            "uso": "Qué traza"}), use_container_width=True, hide_index=True)
    bc = st.columns(2)
    bc[0].markdown(T.tarjeta("Cómo se compone la imagen", B["composicion"]),
                   unsafe_allow_html=True)
    bc[1].markdown(T.tarjeta("Consecuencia para el modelo", B["consecuencia"]),
                   unsafe_allow_html=True)

    st.divider()

    # ---- asinh ------------------------------------------------------------
    A = MT["asinh"]
    st.markdown("#### ② El estiramiento asinh")
    ac = st.columns(2)
    ac[0].markdown(T.tarjeta("El problema", A["problema"]),
                   unsafe_allow_html=True)
    ac[1].markdown(T.tarjeta("La solución", A["solucion"]),
                   unsafe_allow_html=True)
    st.latex(r"v = \frac{\operatorname{asinh}(f/\beta)}"
             r"{\operatorname{asinh}(1/\beta)}")
    T.render_svg(V.html_asinh(), alto=210)
    st.write(A["ventaja"])
    st.warning(A["consecuencia"])

    st.divider()

    # ---- Recorte ----------------------------------------------------------
    R = MT["recorte"]
    st.markdown("#### ③ Por qué se recorta la imagen")
    rc1, rc2 = st.columns([2, 3], gap="large")
    with rc1:
        T.render_svg(V.svg_recorte(), alto=280)
    with rc2:
        st.markdown(T.tarjeta("El hecho", R["hecho"]), unsafe_allow_html=True)
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        st.markdown(T.tarjeta("La cuenta", R["cuenta"]), unsafe_allow_html=True)
    rc3 = st.columns(2)
    rc3[0].markdown(T.tarjeta("Razón algorítmica", R["algoritmico"]),
                    unsafe_allow_html=True)
    rc3[1].markdown(T.tarjeta("Razón física", R["fisico"]),
                    unsafe_allow_html=True)
    st.success(R["resultado"])

    st.divider()

    # ---- Color ------------------------------------------------------------
    CO = MT["color"]
    st.markdown("#### ④ Por qué se conserva el color")
    cc = st.columns(2)
    cc[0].markdown(T.tarjeta("Razón física", CO["fisico"]),
                   unsafe_allow_html=True)
    cc[1].markdown(T.tarjeta("Razón algorítmica", CO["algoritmico"]),
                   unsafe_allow_html=True)
    st.info(CO["consecuencia_aumentacion"])

    st.divider()

    # ---- Fugas ------------------------------------------------------------
    F = MT["fugas"]
    st.markdown("#### ⑤ Control de fugas de información")
    fc = st.columns(2)
    fc[0].markdown(T.tarjeta("El problema", F["problema"]),
                   unsafe_allow_html=True)
    fc[1].markdown(T.tarjeta("La solución", F["solucion"]),
                   unsafe_allow_html=True)
    st.success(F["verificacion"])

    st.divider()

    # ---- Etiquetado -------------------------------------------------------
    st.markdown("#### ⑥ El etiquetado desde el árbol de decisión")
    st.write(C["arbol_gz2"]["descripcion"])
    st.markdown(
        "Como las ramas tienen profundidades distintas —una elíptica atraviesa "
        "una pregunta y una espiral tres—, el producto de probabilidades "
        "condicionadas penalizaría sistemáticamente a las ramas profundas. Se "
        "usa por eso la **media geométrica**, que pone todas las ramas en pie "
        "de igualdad.")
    st.error(C["arbol_gz2"]["advertencia"])
