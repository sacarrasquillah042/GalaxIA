"""
GalaxIA — punto de entrada.

    streamlit run app/Inicio.py

Usa st.navigation(position="top") para el menú fijo superior. Las vistas viven
en app/vistas/ (ya no en app/pages/, que Streamlit ignora cuando hay
navegación programática).
"""
import sys
from pathlib import Path

import streamlit as st

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "app"))

# Salvaguarda: este archivo es el ENRUTADOR, no la portada. La portada es
# app/vistas/inicio.py. Si se copia una sobre otra el fallo aparece lejos,
# como un JSON que no existe, así que conviene detectarlo aquí.
if not (RAIZ / "app" / "vistas" / "inicio.py").exists():
    st.error(
        "**Falta `app/vistas/inicio.py`.**\n\n"
        "Este archivo (`app/Inicio.py`) es solo el enrutador del menú. "
        "La portada vive en `app/vistas/inicio.py`. Si copió un archivo "
        "sobre el otro, vuelva a descomprimir el paquete completo de `app/`."
    )
    st.stop()

st.set_page_config(page_title="GalaxIA", page_icon="🌌", layout="wide",
                   initial_sidebar_state="collapsed")

# Logo de la cabecera: enlaza siempre al inicio.
LOGO = RAIZ / "app" / "assets" / "logo.svg"
if not LOGO.exists():
    LOGO.parent.mkdir(parents=True, exist_ok=True)
    LOGO.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 210 46" width="210" '
        'height="46"><defs><linearGradient id="g" x1="0" x2="1">'
        '<stop offset="0%" stop-color="#B18CF0"/>'
        '<stop offset="55%" stop-color="#7FB2F0"/>'
        '<stop offset="100%" stop-color="#F07B8C"/></linearGradient></defs>'
        '<circle cx="23" cy="23" r="13" fill="none" stroke="url(#g)" '
        'stroke-width="2.4" opacity="0.9"/>'
        '<ellipse cx="23" cy="23" rx="18" ry="7" fill="none" stroke="url(#g)" '
        'stroke-width="1.6" opacity="0.7" transform="rotate(-24 23 23)"/>'
        '<circle cx="23" cy="23" r="4.5" fill="#E0B050"/>'
        '<text x="47" y="31" font-family="Inter,Helvetica,sans-serif" '
        'font-size="25" font-weight="800" fill="url(#g)">galaxIA</text></svg>'
    )

# st.logo no admite enlaces internos (solo http/https), así que el logo se
# hace clicable desde componentes/tema.py con un pequeño script.
st.logo(str(LOGO), size="large")

paginas = [
    st.Page("vistas/inicio.py", title="Inicio", icon=":material/home:",
            default=True, url_path="inicio"),
    st.Page("vistas/tipologias.py", title="Tipologías",
            icon=":material/blur_circular:", url_path="tipologias"),
    st.Page("vistas/clasificador.py", title="Clasificador",
            icon=":material/travel_explore:", url_path="clasificador"),
    st.Page("vistas/resultados.py", title="Resultados",
            icon=":material/bar_chart:", url_path="resultados"),
    st.Page("vistas/fuentes.py", title="Fuentes",
            icon=":material/menu_book:", url_path="fuentes"),
]

st.navigation(paginas, position="top").run()
