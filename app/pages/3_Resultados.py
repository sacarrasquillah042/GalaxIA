"""Página 3: resultados interactivos, guía de métricas y metodología."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "app"))

from componentes import graficos as G  # noqa: E402
from componentes import visual as V  # noqa: E402

st.set_page_config(page_title="Resultados — GalaxIA", page_icon="📊",
                   layout="wide")


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


M = metricas()
C = contenido()
MC = mcnemar()

st.title("📊 Resultados")
if not M:
    st.error("Falta `reports/metrics.json`.")
    st.stop()

t1, t2, t3, t4 = st.tabs(
    ["Comparación", "Por modelo", "Guía de métricas", "Metodología"])

# =========================================================================== #
with t1:
    st.markdown("#### Los nueve modelos sobre el mismo conjunto de prueba")

    met = st.radio("Métrica", ["macro_f1", "accuracy", "roc_auc_macro",
                               "avg_precision_macro"],
                   horizontal=True, label_visibility="collapsed",
                   format_func=lambda k: {"macro_f1": "F1-macro",
                                          "accuracy": "Exactitud",
                                          "roc_auc_macro": "AUC-ROC",
                                          "avg_precision_macro": "AP-macro"}[k])
    st.plotly_chart(G.barras_modelos(M, met), use_container_width=True)

    with st.expander("Ver la tabla con todos los valores", expanded=False):
        tabla = pd.DataFrame([{
            "Modelo": r["modelo"], "Exactitud": r["accuracy"],
            "F1-macro": r["macro_f1"], "AUC-ROC": r["roc_auc_macro"],
            "AP-macro": r["avg_precision_macro"],
            "Entren. (s)": r["t_entrenamiento_s"],
            "Inferencia (s)": r["t_inferencia_s"],
        } for r in M.values()]).sort_values("F1-macro", ascending=False)
        st.dataframe(
            tabla.style.format({
                "Exactitud": "{:.4f}", "F1-macro": "{:.4f}",
                "AUC-ROC": "{:.4f}", "AP-macro": "{:.4f}",
                "Entren. (s)": "{:.1f}", "Inferencia (s)": "{:.2f}"})
            .background_gradient(subset=["F1-macro"], cmap="Blues"),
            use_container_width=True, hide_index=True)

    st.info("**Por qué F1-macro y no exactitud.** Las clases están "
            "desbalanceadas, así que un modelo que ignore la clase minoritaria "
            "puede alcanzar exactitud alta sin ser útil. El F1-macro pesa las "
            "tres clases por igual y delata ese comportamiento.")

    st.divider()
    st.markdown("#### Perfil comparado")
    sel = st.multiselect("Modelos", sorted(M),
                         default=sorted(M, key=lambda k: -M[k]["macro_f1"])[:3])
    if sel:
        st.plotly_chart(G.radar_modelos(M, sel), use_container_width=True)

    st.divider()
    st.markdown("#### Exactitud frente a coste de inferencia")
    st.plotly_chart(G.coste_vs_exactitud(M), use_container_width=True)
    st.caption("El eje horizontal está en escala logarítmica. El **tamaño** de "
               "cada punto representa el coste de entrenamiento. KNN no entrena "
               "pero predice lento; la CNN es lo contrario. Para clasificar "
               "millones de objetos en un sondeo, esa diferencia importa tanto "
               "como la exactitud.")

    # ---- McNemar ----------------------------------------------------------
    if MC:
        st.divider()
        st.markdown("#### ¿Las diferencias son estadísticamente reales?")
        st.caption("Prueba de McNemar sobre las mismas galaxias: cuenta en "
                   "cuántos casos un modelo acierta y el otro falla.")
        dmc = pd.DataFrame(MC)
        st.dataframe(dmc, use_container_width=True, hide_index=True)

# =========================================================================== #
with t2:
    nombre = st.selectbox("Modelo", sorted(M, key=lambda k: -M[k]["macro_f1"]))
    r = M[nombre]

    c = st.columns(4)
    c[0].metric("Exactitud", f"{r['accuracy']:.4f}")
    c[1].metric("F1-macro", f"{r['macro_f1']:.4f}")
    c[2].metric("AUC-ROC", f"{r['roc_auc_macro']:.4f}")
    c[3].metric("AP-macro", f"{r['avg_precision_macro']:.4f}")

    st.markdown("#### Matriz de confusión")
    norm = st.toggle("Normalizar por fila", value=True)
    cm1, cm2 = st.columns([3, 2])
    with cm1:
        st.plotly_chart(G.matriz_confusion(r["matriz_confusion"], r["clases"],
                                           norm), use_container_width=True)
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
    st.caption("Consulte la pestaña «Guía de métricas» para saber qué mide "
               "cada columna y cuándo importa.")

    # ---- Figuras ----------------------------------------------------------
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

    for k, m in C["metricas"].items():
        with st.expander(m["nombre"]):
            st.code(m["formula"], language=None)
            st.write(m["que_mide"])
            if m.get("cuando_importa"):
                st.success(f"**Cuándo importa.** {m['cuando_importa']}")
            if m.get("cuando_engana"):
                st.warning(f"**Cuándo engaña.** {m['cuando_engana']}")

    st.divider()
    st.markdown("#### Cómo leer las curvas")
    cc1, cc2 = st.tabs(["Curva ROC", "Curva precisión-recall"])
    for tab, clave, tipo in [(cc1, "roc", "roc"), (cc2, "pr", "pr")]:
        with tab:
            g = C["como_leer"][clave]
            a, b = st.columns([3, 2])
            with a:
                st.plotly_chart(G.curva_pedagogica(tipo),
                                use_container_width=True)
                st.caption("Curvas ilustrativas, no datos del proyecto.")
            with b:
                st.markdown(f"**{g['titulo']}**")
                for i, p in enumerate(g["pasos"], 1):
                    st.markdown(f"{i}. {p}")
                st.warning(g["cuidado"])

# =========================================================================== #
with t4:
    MT = C["metodologia"]

    st.markdown("### Del telescopio al modelo")
    st.caption("Cada decisión del preprocesamiento tiene una justificación "
               "física y una algorítmica. Aquí están ambas.")

    # ---- Bandas -----------------------------------------------------------
    with st.expander("① Las bandas fotométricas g, r, i", expanded=True):
        B = MT["bandas"]
        st.write(B["intro"])
        st.html(V.svg_bandas())
        st.dataframe(
            pd.DataFrame(B["detalle"]).rename(columns={
                "banda": "Banda", "rango": "Longitud de onda",
                "region": "Región", "uso": "Qué traza"}),
            use_container_width=True, hide_index=True)
        st.info(B["composicion"])
        st.success(B["consecuencia"])

    # ---- asinh ------------------------------------------------------------
    with st.expander("② El estiramiento asinh"):
        A = MT["asinh"]
        st.markdown(f"**El problema.** {A['problema']}")
        st.markdown(f"**La solución.** {A['solucion']}")
        st.latex(r"v = \frac{\operatorname{asinh}(f/\beta)}"
                 r"{\operatorname{asinh}(1/\beta)}")
        st.html(V.html_asinh())
        st.write(A["ventaja"])
        st.warning(A["consecuencia"])

    # ---- Recorte ----------------------------------------------------------
    with st.expander("③ Por qué se recorta la imagen"):
        R = MT["recorte"]
        rc1, rc2 = st.columns([2, 3])
        with rc1:
            st.html(V.svg_recorte())
        with rc2:
            st.markdown(f"**El hecho.** {R['hecho']}")
            st.markdown(f"**La cuenta.** {R['cuenta']}")
            st.markdown(f"**Razón algorítmica.** {R['algoritmico']}")
            st.markdown(f"**Razón física.** {R['fisico']}")
        st.success(R["resultado"])

    # ---- Color ------------------------------------------------------------
    with st.expander("④ Por qué se conserva el color"):
        CO = MT["color"]
        st.markdown(f"**Razón física.** {CO['fisico']}")
        st.markdown(f"**Razón algorítmica.** {CO['algoritmico']}")
        st.info(CO["consecuencia_aumentacion"])

    # ---- Fugas ------------------------------------------------------------
    with st.expander("⑤ Control de fugas de información"):
        F = MT["fugas"]
        st.markdown(f"**El problema.** {F['problema']}")
        st.markdown(f"**La solución.** {F['solucion']}")
        st.success(F["verificacion"])

    # ---- Árbol ------------------------------------------------------------
    with st.expander("⑥ El etiquetado desde el árbol de decisión"):
        st.write(C["arbol_gz2"]["descripcion"])
        st.latex(r"P(\text{hoja}) = \prod_{q \in \text{rama}} "
                 r"P(\text{respuesta}_q \mid \text{padre})")
        st.markdown(
            "Como las ramas tienen profundidades distintas —una elíptica "
            "atraviesa una pregunta y una espiral tres—, el producto crudo "
            "penalizaría sistemáticamente a las ramas profundas. Se usa por "
            "eso la **media geométrica** "
            r"$\text{conf}^{1/\text{etapas}}$, que pone todas las ramas en "
            "pie de igualdad.")
        st.error(C["arbol_gz2"]["advertencia"])
