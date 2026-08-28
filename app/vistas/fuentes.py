"""Fuentes: todas las referencias con su enlace explícito."""
import json
from pathlib import Path

import streamlit as st

from componentes import tema as T

RAIZ = Path(__file__).resolve().parents[2]
T.aplicar_tema(st)


@st.cache_data
def contenido():
    return json.loads((RAIZ / "app" / "contenido" / "tipologias.json").read_text())


C = contenido()

st.markdown("<div class='gx-centro'><h1>Fuentes</h1>"
            "<p style='color:#A9A2C4;font-size:17px'>Toda la información "
            "presentada en esta interfaz procede de las referencias siguientes."
            "</p></div>", unsafe_allow_html=True)

fuentes = C.get("fuentes", [])
grupos = {}
for f in fuentes:
    grupos.setdefault(f["tipo"], []).append(f)

orden = ["Artículo", "Libro", "Divulgación", "Plataforma", "Datos"]
for tipo in [t for t in orden if t in grupos] + \
            [t for t in grupos if t not in orden]:
    st.markdown(f"#### {tipo}")
    for f in grupos[tipo]:
        enlace = (f"<br><a href='{f['url']}' target='_blank' "
                  f"style='color:#7FB2F0;font-size:15px'>{f['url']}</a>"
                  if f.get("url") else "")
        st.markdown(
            f"<div style='background:rgba(123,79,191,.08);border:1px solid "
            f"rgba(177,140,240,.18);border-left:3px solid #B18CF0;"
            f"border-radius:10px;padding:14px 18px;margin-bottom:12px;"
            f"font-size:16px;line-height:1.7'>{f['cita']}{enlace}</div>",
            unsafe_allow_html=True)

st.divider()
st.markdown("#### Sobre las ilustraciones")
st.markdown(
    "Los esquemas y animaciones de esta interfaz son gráficos vectoriales "
    "generados por el propio código del proyecto (`app/componentes/visual.py`), "
    "no imágenes descargadas. Se tomó esta decisión para que la aplicación "
    "funcione sin conexión, para no depender de licencias de terceros y porque "
    "un esquema hecho a medida explica mejor un concepto concreto que una "
    "fotografía genérica.")
st.markdown(
    "Las únicas fotografías son las del **Sloan Digital Sky Survey**, "
    "procedentes del subconjunto público del Galaxy Zoo Challenge, en las "
    "bandas g, r, i.")

st.divider()
GZ = C["galaxy_zoo"]
cf = st.columns(3)
cf[0].link_button("NASA — Hubble & Citizen Science", GZ["fuente_url"],
                  use_container_width=True)
cf[1].link_button("Clasificar en Galaxy Zoo", GZ["plataforma"],
                  use_container_width=True, type="primary")
cf[2].link_button("SDSS SkyServer", "https://skyserver.sdss.org/",
                  use_container_width=True)
