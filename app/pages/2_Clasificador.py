"""Página 2: clasificador interactivo con Grad-CAM y galaxias similares."""
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import streamlit as st

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))

st.set_page_config(page_title="Clasificador — GalaxIA", page_icon="🔭", layout="wide")


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
def cache_imagenes():
    from galaxia import data
    imgs, ids = data.cargar_cache(128)
    return imgs, {int(g): i for i, g in enumerate(ids)}


st.title("🔭 Clasificador")

M = muestra()
modelo, nombre_modelo = cnn()

if M is None:
    st.error("Falta `app/assets/muestra_test.npz`. "
             "Ejecutar `python scripts/export_artefactos.py`.")
    st.stop()
if modelo is None:
    st.error("No se encontró ninguna CNN en `models/`.")
    st.stop()

clases = M["clases"]
st.caption(f"Modelo cargado: {nombre_modelo}. Las galaxias mostradas pertenecen "
           "al conjunto de prueba, nunca vistas durante el entrenamiento.")

# --------------------------------------------------------------------------- #
col_izq, col_der = st.columns([1, 2])

with col_izq:
    origen = st.radio("Origen de la imagen",
                      ["Galaxia del conjunto de prueba", "Subir una imagen"])

    if origen == "Galaxia del conjunto de prueba":
        if "idx" not in st.session_state:
            st.session_state.idx = int(np.random.randint(len(M["ids"])))
        if st.button("🎲 Otra galaxia al azar", use_container_width=True):
            st.session_state.idx = int(np.random.randint(len(M["ids"])))
        i = st.session_state.idx
        img = M["X"][i]
        verdad = clases[int(M["y"][i])]
        gid = int(M["ids"][i])
        st.image(img, caption=f"GalaxyID {gid}", use_container_width=True)
        st.markdown(f"**Etiqueta de Galaxy Zoo:** {verdad}")
        st.link_button(
            "Ficha en SDSS SkyServer",
            "https://skyserver.sdss.org/dr18/VisualTools/explore",
            use_container_width=True,
        )
    else:
        subida = st.file_uploader("Imagen de una galaxia", type=["jpg", "jpeg", "png"])
        if subida is None:
            st.info("Sube una imagen para clasificarla.")
            st.stop()
        from PIL import Image
        import cv2
        from galaxia.features import recortar_centro

        arr = np.array(Image.open(subida).convert("RGB"))
        if min(arr.shape[:2]) > 250:
            arr = recortar_centro(arr, crop=min(arr.shape[:2]) // 2)
        img = cv2.resize(arr, (128, 128), interpolation=cv2.INTER_AREA)
        verdad, gid = None, None
        st.image(img, caption="Imagen procesada (recorte central)",
                 use_container_width=True)

# --------------------------------------------------------------------------- #
with col_der:
    entrada = np.asarray(img, dtype=np.float32)[None, ...]
    proba = np.asarray(modelo.predict(entrada, verbose=0), dtype=float)[0]
    pred = int(proba.argmax())

    st.subheader(f"Predicción: **{clases[pred]}**")
    if verdad is not None:
        if clases[pred] == verdad:
            st.success(f"Coincide con la clasificación de los voluntarios ({verdad}).")
        else:
            st.warning(f"No coincide. Galaxy Zoo la clasificó como **{verdad}**.")

    st.markdown("#### Probabilidad por clase")
    for c, p in sorted(zip(clases, proba), key=lambda t: -t[1]):
        st.progress(float(p), text=f"{c} — {p*100:.1f} %")

    st.caption(
        "Una predicción repartida entre dos clases no es un fallo del modelo: "
        "suele señalar galaxias donde los voluntarios humanos también discreparon."
    )

    # ---- Grad-CAM -----------------------------------------------------
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
                "el resultado no sería confiable por buena que fuera la exactitud."
            )
        except Exception as e:
            st.info(f"Grad-CAM no disponible: {e}")

# --------------------------------------------------------------------------- #
K = knn()
if K is not None and origen == "Galaxia del conjunto de prueba":
    st.divider()
    st.markdown("#### Galaxias similares")
    st.caption("Vecinos más cercanos en el espacio de componentes principales. "
               "Es lo que el KNN aporta y ningún otro modelo del estudio: "
               "objetos reales que se parecen al consultado.")
    try:
        from galaxia import features

        vec_px = features.a_vector(np.asarray(img)[None, ...], size=64)
        z = K["prep"].transform(vec_px)[:, : K["K"]]
        _, vecinos = K["nn"].kneighbors(z, n_neighbors=5)

        imgs_cache, pos = cache_imagenes()
        cols = st.columns(5)
        for col, j in zip(cols, vecinos[0]):
            g = int(K["ids_train"][j])
            col.image(np.array(imgs_cache[pos[g]]), use_container_width=True)
            col.caption(f"{K['clases'][int(K['y_train'][j])]}\n`{g}`")
    except Exception as e:
        st.info(f"Vecinos no disponibles: {e}")
