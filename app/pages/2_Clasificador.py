"""Página 2: clasificador interactivo con transición de baraja y Grad-CAM."""
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import streamlit as st

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "app"))

from componentes import graficos as G  # noqa: E402
from componentes import visual as V  # noqa: E402

st.set_page_config(page_title="Clasificador — GalaxIA", page_icon="🔭",
                   layout="wide")


# --------------------------------------------------------------------------- #
@st.cache_data
def muestra():
    p = RAIZ / "app" / "assets" / "muestra_test.npz"
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=True)
    return {"X": z["X"], "y": z["y"], "ids": z["ids"],
            "clases": [str(c) for c in z["clases"]]}


@st.cache_resource
def cnn():
    import keras
    for n in ["cnn_224.keras", "cnn_128.keras"]:
        p = RAIZ / "models" / n
        if p.exists():
            return keras.models.load_model(p), n
    return None, None


@st.cache_resource
def knn():
    import joblib
    p = RAIZ / "models" / "knn.pkl"
    return joblib.load(p) if p.exists() else None


@st.cache_resource
def cache_imgs():
    from galaxia import data
    imgs, ids = data.cargar_cache(128)
    return imgs, {int(g): i for i, g in enumerate(ids)}


# --------------------------------------------------------------------------- #
st.title("🔭 Clasificador")

M = muestra()
modelo, nombre_modelo = cnn()
if M is None:
    st.error("Falta `app/assets/muestra_test.npz`. Ejecute "
             "`python scripts/export_artefactos.py`.")
    st.stop()
if modelo is None:
    st.error("No se encontró ninguna CNN en `models/`.")
    st.stop()

clases = M["clases"]

# Estado
if "idx" not in st.session_state:
    st.session_state.idx = int(np.random.randint(len(M["ids"])))
if "giro" not in st.session_state:
    st.session_state.giro = 0
if "propia" not in st.session_state:
    st.session_state.propia = None

usando_propia = st.session_state.propia is not None

# --------------------------------------------------------------------------- #
izq, der = st.columns([1, 2])

with izq:
    if usando_propia:
        img = st.session_state.propia
        verdad, gid = None, None
        st.image(img, use_container_width=True, caption="Su imagen (procesada)")
        if st.button("← Volver al catálogo", use_container_width=True):
            st.session_state.propia = None
            st.rerun()
    else:
        i = st.session_state.idx
        img = np.asarray(M["X"][i])
        verdad = clases[int(M["y"][i])]
        gid = int(M["ids"][i])

        # Animación de baraja: la clave fuerza a Streamlit a re-renderizar
        st.html(V.carta_baraja(V.a_base64(img), st.session_state.giro))

        if st.button("🎲  Otra galaxia al azar", use_container_width=True,
                     type="primary"):
            nuevo = st.session_state.idx
            while nuevo == st.session_state.idx and len(M["ids"]) > 1:
                nuevo = int(np.random.randint(len(M["ids"])))
            st.session_state.idx = nuevo
            st.session_state.giro += 1
            st.rerun()

        st.markdown(
            f"<div style='display:flex;justify-content:space-between;"
            f"font-size:14px;margin-top:6px'>"
            f"<span style='color:#9AA5B5'>GalaxyID</span>"
            f"<code>{gid}</code></div>", unsafe_allow_html=True)
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;"
            f"font-size:14px'><span style='color:#9AA5B5'>Galaxy Zoo</span>"
            f"<b style='color:#D4A032'>{verdad}</b></div>",
            unsafe_allow_html=True)

st.caption(f"Modelo: {nombre_modelo}. Las galaxias del catálogo pertenecen al "
           "conjunto de prueba, nunca vistas durante el entrenamiento.")

# --------------------------------------------------------------------------- #
with der:
    proba = np.asarray(
        modelo.predict(np.asarray(img, dtype=np.float32)[None, ...], verbose=0),
        dtype=float)[0]
    pred = int(proba.argmax())

    st.markdown(
        f"<div style='font-size:15px;color:#9AA5B5'>Predicción del modelo</div>"
        f"<div style='font-size:38px;font-weight:800;color:#D4A032;"
        f"margin-top:-6px'>{clases[pred]}</div>", unsafe_allow_html=True)

    if verdad is not None:
        if clases[pred] == verdad:
            st.success(f"Coincide con la clasificación de los voluntarios.")
        else:
            st.warning(f"No coincide: Galaxy Zoo la clasificó como **{verdad}**.")

    st.plotly_chart(G.barras_probabilidad(clases, proba),
                    use_container_width=True, config={"displayModeBar": False})
    st.caption("Una predicción repartida entre dos clases no es un fallo: suele "
               "señalar galaxias donde los voluntarios humanos también "
               "discreparon.")

    # ---- Grad-CAM ---------------------------------------------------------
    with st.expander("¿Dónde está mirando la red? (Grad-CAM)", expanded=True):
        try:
            from galaxia.explain import grad_cam, superponer
            mapa, _, _ = grad_cam(modelo, img, clase=pred)
            a, b = st.columns(2)
            a.image(img, caption="Original", use_container_width=True)
            b.image(superponer(np.asarray(img, dtype=np.uint8), mapa),
                    caption="Atención de la red", use_container_width=True)
            st.caption(
                "Las zonas cálidas son las que más influyen en la decisión. "
                "Si la red se fijara en el fondo o en los bordes del recorte, "
                "el resultado no sería confiable por buena que fuera la "
                "exactitud.")
        except Exception as e:
            st.info(f"Grad-CAM no disponible: {e}")

# --------------------------------------------------------------------------- #
K = knn()
if K is not None:
    st.divider()
    st.markdown("#### Galaxias similares")
    st.caption("Vecinos más cercanos en el espacio de componentes principales. "
               "Es lo que aporta el KNN y ningún otro modelo del estudio: "
               "objetos reales que se parecen al consultado.")
    try:
        from galaxia import features
        z = K["prep"].transform(
            features.a_vector(np.asarray(img)[None, ...], size=64))[:, : K["K"]]
        _, vecinos = K["nn"].kneighbors(z, n_neighbors=5)
        imgs_c, pos = cache_imgs()
        for col, j in zip(st.columns(5), vecinos[0]):
            g = int(K["ids_train"][j])
            col.image(np.array(imgs_c[pos[g]]), use_container_width=True)
            col.caption(f"{K['clases'][int(K['y_train'][j])]} · `{g}`")
    except Exception as e:
        st.info(f"Vecinos no disponibles: {e}")

# --------------------------------------------------------------------------- #
st.divider()
with st.expander("📤  ¿Quiere clasificar una imagen propia?", expanded=False):
    st.markdown(
        "Suba la imagen de una galaxia y el modelo la clasificará. Para que el "
        "resultado sea comparable, la imagen se recorta al centro y se "
        "redimensiona igual que las del catálogo.")
    subida = st.file_uploader("Imagen", type=["jpg", "jpeg", "png"],
                              label_visibility="collapsed")
    if subida is not None:
        import cv2
        from PIL import Image
        from galaxia.features import recortar_centro

        arr = np.array(Image.open(subida).convert("RGB"))
        if min(arr.shape[:2]) > 250:
            arr = recortar_centro(arr, crop=min(arr.shape[:2]) // 2)
        st.session_state.propia = cv2.resize(arr, (128, 128),
                                             interpolation=cv2.INTER_AREA)
        st.rerun()
    st.caption("El modelo fue entrenado con imágenes del SDSS en las bandas "
               "g, r, i. Con imágenes de otros instrumentos o procesadas de "
               "otra forma, la predicción es menos fiable.")

# --------------------------------------------------------------------------- #
st.divider()
with st.expander("¿Por qué no hay enlace a la ficha SDSS de esta galaxia?"):
    st.markdown("""
Las herramientas del SkyServer localizan un objeto por sus **coordenadas
(RA, Dec)** o por su **objID de SDSS**, un identificador de 18 dígitos.

El conjunto del *Galaxy Zoo Challenge* usa identificadores propios de seis
dígitos —el `GalaxyID` que aparece arriba— que fueron **anonimizados para la
competición**. No hay coordenadas en el archivo ni una tabla pública de
correspondencia con los objID del SDSS, así que no es posible construir un
enlace directo a la ficha de un objeto concreto.

Recuperar esa correspondencia exigiría cruzar el catálogo completo de Galaxy
Zoo 2 (`gz2_hart16`, que sí incluye `dr7objid` y coordenadas) con las imágenes,
lo que queda como trabajo futuro de este proyecto.
""")
    c1, c2 = st.columns(2)
    c1.link_button("Herramienta Explore del SkyServer",
                   "https://skyserver.sdss.org/dr18/VisualTools/explore/summary",
                   use_container_width=True)
    c2.link_button("Catálogo de Galaxy Zoo",
                   "https://data.galaxyzoo.org/",
                   use_container_width=True)
