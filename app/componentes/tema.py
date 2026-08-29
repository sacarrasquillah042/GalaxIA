"""
Tema visual de GalaxIA: paleta galáctica, fondo estrellado, tipografía y
transiciones de aparición al hacer scroll.

Nota técnica importante
-----------------------
``st.html`` sanitiza el contenido con DOMPurify, que elimina los elementos
``<animateTransform>`` y ``<animate>`` de SVG. Por eso las animaciones no se
veían. La solución es ``st.components.v1.html``, que renderiza en un iframe
aislado sin sanitización. Use siempre ``render_svg()`` de este módulo.
"""
from __future__ import annotations

from pathlib import Path

import streamlit.components.v1 as components


def raiz_proyecto(desde: str | Path | None = None) -> Path:
    """
    Localiza la raíz del proyecto subiendo hasta encontrar un marcador.

    No se cuentan niveles de carpeta a propósito: contar es frágil, porque un
    mismo archivo colocado en otra profundidad devuelve una raíz equivocada y
    el fallo aparece lejos, como un fichero que no existe.
    """
    p = Path(desde or __file__).resolve()
    for candidato in [p, *p.parents]:
        if (candidato / "app" / "contenido").is_dir() and \
           (candidato / "src" / "galaxia").is_dir():
            return candidato
    # Sin marcadores: dos niveles por encima de componentes/
    return Path(__file__).resolve().parents[2]

# --------------------------------------------------------------------------- #
# Paleta galáctica
# --------------------------------------------------------------------------- #
P = {
    "fondo": "#07060F",
    "fondo2": "#0E0A1E",
    "morado": "#7B4FBF",
    "morado_claro": "#B18CF0",
    "azul": "#4A7FD9",
    "azul_claro": "#7FB2F0",
    "rojo": "#D9455F",
    "rojo_claro": "#F07B8C",
    "oro": "#E0B050",
    "texto": "#E9E6F5",
    "texto2": "#A9A2C4",
    "borde": "rgba(177,140,240,.22)",
}


# --------------------------------------------------------------------------- #
def _campo_estrellas(n: int = 160, semilla: int = 7) -> str:
    """Genera box-shadows para el campo de estrellas (sin imágenes externas)."""
    import random

    rng = random.Random(semilla)
    capas = []
    for tam, cant, op in [(1, n, 0.75), (2, n // 3, 0.55), (3, n // 9, 0.40)]:
        sombras = ", ".join(
            f"{rng.randint(0, 2000)}px {rng.randint(0, 2000)}px "
            f"rgba(255,255,255,{op * rng.uniform(.4, 1):.2f})"
            for _ in range(cant)
        )
        capas.append((tam, sombras))
    return capas


def css_global() -> str:
    """Hoja de estilos completa de la aplicación."""
    capas = _campo_estrellas()
    estrellas = "\n".join(
        f"""
        .gx-stars{i} {{
          position: fixed; top: 0; left: 0;
          width: {t}px; height: {t}px; border-radius: 50%;
          background: transparent; box-shadow: {s};
          animation: gxDrift {90 + i * 55}s linear infinite;
          z-index: 0; pointer-events: none;
        }}"""
        for i, (t, s) in enumerate(capas)
    )

    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Fraunces:opsz,wght@9..144,600;9..144,700&display=swap');

/* ---------------- Fondo galáctico ---------------- */
.stApp {{
  background:
    radial-gradient(1200px 700px at 12% -8%, rgba(123,79,191,.32), transparent 62%),
    radial-gradient(1000px 620px at 88% 4%, rgba(74,127,217,.24), transparent 60%),
    radial-gradient(900px 560px at 62% 108%, rgba(217,69,95,.20), transparent 62%),
    linear-gradient(180deg, {P['fondo']} 0%, {P['fondo2']} 55%, {P['fondo']} 100%);
  background-attachment: fixed;
  color: {P['texto']};
}}

@keyframes gxDrift {{
  from {{ transform: translateY(0); }}
  to   {{ transform: translateY(-2000px); }}
}}
{estrellas}

/* ---------------- Tipografía ----------------
   IMPORTANTE: no ampliar la fuente con selectores amplios como
   [class*="st-"] ni sobre .stApp. Los widgets de Streamlit tienen alturas
   fijas calculadas para 14-15px; al agrandar el texto dentro de ellos, este
   se desborda y se superpone. La ampliación se aplica SOLO al contenido. */
html, body, .stApp {{
  font-family: 'Inter', system-ui, sans-serif;
}}
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
.stMarkdown p, .stMarkdown li {{
  font-size: 17.5px; line-height: 1.75;
}}
/* Los widgets conservan su escala nativa */
[data-testid="stFileUploader"], [data-testid="stFileUploader"] *,
[data-baseweb="select"], [data-baseweb="select"] *,
[data-testid="stTopNav"], [data-testid="stTopNav"] *,
[data-testid="stSelectbox"] *, [data-testid="stMultiSelect"] *,
[data-testid="stSlider"] *, [data-testid="stNumberInput"] * {{
  line-height: normal !important;
}}
h1 {{ font-family:'Fraunces',Georgia,serif !important; font-size: 3.0rem !important;
      letter-spacing:-.02em; font-weight:700 !important; }}
h2 {{ font-family:'Fraunces',Georgia,serif !important; font-size: 2.1rem !important; }}
h3 {{ font-size: 1.62rem !important; font-weight: 700 !important; }}
h4 {{ font-size: 1.32rem !important; font-weight: 700 !important;
      color: {P['morado_claro']} !important; margin-top: 1.4em !important; }}
.stCaption, [data-testid="stCaptionContainer"] p {{
  font-size: 14.5px !important; color: {P['texto2']} !important; }}

/* ---------------- Navegación superior ---------------- */
header[data-testid="stHeader"] {{
  background: rgba(7,6,15,.86) !important;
  backdrop-filter: blur(14px);
  border-bottom: 1px solid {P['borde']};
}}
/* OJO: no ocultar [data-testid="stSidebar"]. Con position="top" Streamlit
   monta ahí el logo y los enlaces de navegación, así que ocultarlo los borra. */

/* Menú superior: una sola línea, sin solapamientos */
[data-testid="stTopNav"] {{
  display: flex !important; align-items: center !important;
  gap: 4px; flex-wrap: nowrap; overflow-x: auto;
}}
[data-testid="stTopNavLink"], [data-testid="stTopNav"] a {{
  font-size: 15px !important; font-weight: 600 !important;
  white-space: nowrap !important; line-height: 1.2 !important;
  padding: 7px 13px !important; border-radius: 9px;
  display: inline-flex !important; align-items: center !important; gap: 7px;
}}
[data-testid="stTopNavLink"] p, [data-testid="stTopNav"] a p {{
  margin: 0 !important; font-size: 15px !important; line-height: 1.2 !important;
  white-space: nowrap !important;
}}
[data-testid="stTopNavLink"]:hover, [data-testid="stTopNav"] a:hover {{
  background: rgba(123,79,191,.20) !important;
}}
[data-testid="stTopNavLink"][aria-current="page"],
[data-testid="stTopNav"] a[aria-current="page"] {{
  color: {P['morado_claro']} !important;
}}

/* El logo, bien visible y con espacio */
[data-testid="stLogo"] {{
  height: 42px !important; margin: 4px 10px 4px 4px;
  transition: filter .3s ease, transform .3s ease;
}}
[data-testid="stLogo"]:hover {{
  filter: drop-shadow(0 0 10px rgba(177,140,240,.7));
  transform: scale(1.03);
}}

/* Botón de colapsar la barra lateral: innecesario con menú superior */
[data-testid="stSidebarCollapseButton"],
[data-testid="collapsedControl"] {{ display: none !important; }}

/* ---------------- Aparición al hacer scroll ---------------- */
.gx-reveal {{
  opacity: 0; transform: translateY(26px); filter: blur(7px);
  transition: opacity .85s cubic-bezier(.22,.9,.3,1),
              transform .85s cubic-bezier(.22,.9,.3,1),
              filter .85s cubic-bezier(.22,.9,.3,1);
}}
.gx-reveal.gx-in {{ opacity: 1; transform: none; filter: blur(0); }}

/* ---------------- Componentes ---------------- */
[data-testid="stMetric"] {{
  background: linear-gradient(160deg, rgba(123,79,191,.16), rgba(74,127,217,.07));
  border: 1px solid {P['borde']};
  border-radius: 16px; padding: 16px 18px;
}}
[data-testid="stMetricValue"] {{ font-size: 2.0rem !important;
  color: {P['morado_claro']} !important; }}
[data-testid="stMetricLabel"] p {{ font-size: 14px !important;
  color: {P['texto2']} !important; }}

[data-testid="stVerticalBlockBorderWrapper"] > div[style*="border"] {{
  border-color: {P['borde']} !important;
  border-radius: 16px !important;
  background: rgba(123,79,191,.06);
}}

/* Casillas cuadradas en lugar de círculos para los radios */
.gx-cuadros [data-baseweb="radio"] div:first-child {{
  border-radius: 6px !important; width: 22px !important; height: 22px !important;
  border-width: 2px !important; border-color: {P['morado']} !important;
}}
.gx-cuadros [data-baseweb="radio"] div:first-child div {{
  border-radius: 3px !important; background: {P['morado_claro']} !important; }}
.gx-cuadros [role="radiogroup"] {{
  justify-content: center !important; gap: 26px !important;
  flex-wrap: wrap !important; }}
.gx-cuadros label {{ font-size: 17px !important; }}

/* Tarjetas */
.gx-card {{
  background: linear-gradient(160deg, rgba(123,79,191,.13), rgba(74,127,217,.05));
  border: 1px solid {P['borde']};
  border-radius: 16px; padding: 20px 22px; height: 100%;
  transition: transform .3s ease, border-color .3s ease, box-shadow .3s ease;
}}
.gx-card:hover {{
  transform: translateY(-4px); border-color: {P['morado_claro']};
  box-shadow: 0 12px 32px rgba(123,79,191,.22);
}}
.gx-card h5 {{
  margin: 0 0 8px 0; font-size: 1.18rem; font-weight: 700;
  color: {P['morado_claro']};
}}
.gx-card .gx-sub {{ font-size: 14px; color: {P['texto2']}; margin-bottom: 10px; }}
.gx-card p {{ font-size: 16px; line-height: 1.68; margin: 0 0 10px 0; }}
.gx-porque {{
  border-left: 3px solid {P['oro']}; padding-left: 13px; margin-top: 10px;
  color: {P['texto2']}; font-size: 15px;
}}

.gx-chip {{
  display: inline-block; background: rgba(123,79,191,.25);
  border: 1px solid {P['borde']}; border-radius: 999px;
  padding: 3px 13px; font-size: 14px; margin: 0 6px 6px 0;
  color: {P['morado_claro']};
}}

.gx-cita {{
  border-left: 3px solid {P['azul']}; padding: 6px 0 6px 15px;
  font-size: 15px; color: {P['texto2']}; margin: 14px 0;
}}
.gx-cita a {{ color: {P['azul_claro']}; }}

/* Centrado de bloques */
.gx-centro {{ max-width: 980px; margin: 0 auto; text-align: center; }}

/* Tablas */
[data-testid="stDataFrame"] {{ border-radius: 12px; overflow: hidden; }}

/* Pestañas más grandes */
button[data-baseweb="tab"] {{ font-size: 17px !important; font-weight: 600 !important; }}

/* Cargador de archivos: alturas propias, sin texto desbordado */
[data-testid="stFileUploaderDropzone"] {{
  background: rgba(123,79,191,.08) !important;
  border: 1.5px dashed {P['borde']} !important;
  border-radius: 14px !important;
  padding: 20px 22px !important;
  min-height: 96px !important;
  align-items: center !important;
}}
[data-testid="stFileUploaderDropzoneInstructions"] {{
  display: flex !important; flex-direction: column !important;
  justify-content: center !important;
}}
[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzoneInstructions"] small {{
  font-size: 14px !important; line-height: 1.45 !important;
}}
[data-testid="stFileUploaderDropzone"] button {{
  white-space: nowrap !important; font-size: 14px !important;
  padding: 7px 16px !important; min-height: 38px !important;
}}
[data-testid="stFileUploaderFile"] {{ font-size: 14px !important; }}

/* Botones */
.stButton button, .stDownloadButton button, .stLinkButton a {{
  border-radius: 12px !important; font-size: 15.5px !important;
  font-weight: 600 !important; border: 1px solid {P['borde']} !important;
  line-height: 1.3 !important; min-height: 42px !important;
  padding: 9px 18px !important; white-space: nowrap;
  display: inline-flex !important; align-items: center !important;
  justify-content: center !important;
}}
.stButton button p, .stLinkButton a p {{
  margin: 0 !important; font-size: 15.5px !important; line-height: 1.3 !important;
}}
.stButton button[kind="primary"] {{
  background: linear-gradient(120deg, {P['morado']}, {P['azul']}) !important;
  border: none !important;
}}
</style>
<div class="gx-stars0"></div><div class="gx-stars1"></div><div class="gx-stars2"></div>
"""


# --------------------------------------------------------------------------- #
JS_REVEAL = """
<script>
(function () {
  const doc = window.parent.document;
  function activar() {
    const obs = new IntersectionObserver((entradas) => {
      entradas.forEach(e => { if (e.isIntersecting) {
        e.target.classList.add('gx-in'); obs.unobserve(e.target); } });
    }, { threshold: 0.08, rootMargin: '0px 0px -40px 0px' });

    const bloques = doc.querySelectorAll(
      '[data-testid="stVerticalBlock"] > [data-testid="stElementContainer"],' +
      '[data-testid="stVerticalBlock"] > [data-testid="stVerticalBlockBorderWrapper"],' +
      '[data-testid="stHorizontalBlock"]');
    bloques.forEach((b, i) => {
      if (b.dataset.gxRev) return;
      b.dataset.gxRev = '1';
      // Los primeros elementos visibles no se ocultan, para que la portada
      // no aparezca en blanco al cargar.
      if (b.getBoundingClientRect().top < window.parent.innerHeight * 0.9) {
        b.classList.add('gx-reveal', 'gx-in');
      } else {
        b.classList.add('gx-reveal');
        obs.observe(b);
      }
    });
  }
  // El logo de la cabecera vuelve al inicio. st.logo(link=...) solo admite
  // URLs externas, así que se conecta aquí.
  function logoClicable() {
    const logo = doc.querySelector('[data-testid="stLogo"]');
    if (!logo || logo.dataset.gxLink) return;
    logo.dataset.gxLink = '1';
    logo.style.cursor = 'pointer';
    logo.title = 'Volver al inicio';
    logo.addEventListener('click', () => {
      window.parent.location.href = window.parent.location.origin + '/inicio';
    });
  }

  activar(); logoClicable();
  new MutationObserver(() => { activar(); logoClicable(); })
    .observe(doc.body, { childList: true, subtree: true });
})();
</script>
"""


def aplicar_tema(st) -> None:
    """Inyecta CSS y el observador de scroll. Llamar una vez por página."""
    st.markdown(css_global(), unsafe_allow_html=True)
    components.html(JS_REVEAL, height=0)


_DIR_SVG = None


def render_svg(svg: str, alto: int = 280) -> None:
    """
    Renderiza SVG animado dentro de un iframe.

    Por qué un iframe: ``st.html`` sanitiza con DOMPurify y elimina los
    elementos ``<animateTransform>``/``<animate>``, así que las animaciones
    desaparecen. El iframe no pasa por el sanitizador.

    Se intenta primero ``st.iframe`` (API actual, recibe una ruta) y se recurre
    a ``st.components.v1.html`` si no está disponible. Esta última está marcada
    como obsoleta pero sigue funcionando: el aviso en consola es inofensivo.
    """
    global _DIR_SVG
    doc = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>
      html,body{{margin:0;padding:0;background:transparent;overflow:hidden;
                 display:flex;align-items:center;justify-content:center;
                 height:100%;font-family:Inter,system-ui,sans-serif}}
      svg{{max-width:100%;max-height:100%}}
    </style></head><body>{svg}</body></html>"""

    try:
        import hashlib
        import streamlit as st

        if not hasattr(st, "iframe"):
            raise AttributeError

        if _DIR_SVG is None:
            from pathlib import Path

            _DIR_SVG = Path(__file__).resolve().parents[1] / "assets" / "_svg"
            _DIR_SVG.mkdir(parents=True, exist_ok=True)

        clave = hashlib.md5(doc.encode()).hexdigest()[:16]
        archivo = _DIR_SVG / f"{clave}.html"
        if not archivo.exists():
            archivo.write_text(doc, encoding="utf-8")
        st.iframe(archivo, height=alto)
    except Exception:
        components.html(doc, height=alto, scrolling=False)


def tarjeta(titulo: str, cuerpo: str, sub: str = "", porque: str = "") -> str:
    """HTML de una tarjeta con estilo del tema."""
    s = f'<div class="gx-sub">{sub}</div>' if sub else ""
    p = f'<div class="gx-porque"><b>¿Por qué?</b> {porque}</div>' if porque else ""
    return f'<div class="gx-card"><h5>{titulo}</h5>{s}<p>{cuerpo}</p>{p}</div>'


def cita(texto: str, fuente: str, url: str) -> str:
    """Bloque de cita con enlace explícito a la fuente."""
    return (f'<div class="gx-cita">{texto}<br>'
            f'<a href="{url}" target="_blank">{fuente}</a></div>')
