"""
Ilustraciones SVG animadas para la interfaz.

Decisión deliberada: todo el material gráfico se genera aquí como SVG en lugar
de descargar imágenes externas. Motivos:

1. La app debe funcionar sin red. En un congreso el wifi falla.
2. Evita depender de la disponibilidad o las licencias de imágenes de terceros.
3. Un esquema hecho a medida explica mejor un concepto concreto que una
   fotografía genérica.

Las fotografías reales que sí se muestran son las del propio SDSS, que ya
están en el conjunto de datos y cuya procedencia se cita explícitamente.
"""
from __future__ import annotations

PALETA = {
    "azul": "#192D5A",
    "oro": "#D4A032",
    "rojo": "#C1440E",
    "azulc": "#4A90D9",
    "gris": "#5F6978",
    "fondo": "#0B0E17",
}


def _wrap(svg: str, alto: int = 260) -> str:
    return f'<div style="width:100%;display:flex;justify-content:center">{svg}</div>'


# --------------------------------------------------------------------------- #
# Tipos de galaxia — animaciones
# --------------------------------------------------------------------------- #
def svg_eliptica() -> str:
    """Elíptica: elipsoide de brillo suave, órbitas desordenadas."""
    orbitas = "".join(
        f'<ellipse cx="150" cy="110" rx="{40+i*9}" ry="{22+i*7}" fill="none" '
        f'stroke="#D4A032" stroke-width="0.7" opacity="0.30" '
        f'transform="rotate({i*37} 150 110)">'
        f'<animateTransform attributeName="transform" type="rotate" '
        f'from="{i*37} 150 110" to="{i*37+360} 150 110" '
        f'dur="{28+i*5}s" repeatCount="indefinite"/></ellipse>'
        for i in range(7)
    )
    return _wrap(f"""
<svg viewBox="0 0 300 220" width="100%" style="max-width:420px">
  <defs>
    <radialGradient id="ge" cx="50%" cy="50%">
      <stop offset="0%" stop-color="#FFF3D6"/>
      <stop offset="30%" stop-color="#F0C36B"/>
      <stop offset="70%" stop-color="#C1440E" stop-opacity="0.55"/>
      <stop offset="100%" stop-color="#C1440E" stop-opacity="0"/>
    </radialGradient>
  </defs>
  <rect width="300" height="220" fill="#0B0E17" rx="8"/>
  {orbitas}
  <ellipse cx="150" cy="110" rx="88" ry="60" fill="url(#ge)"/>
  <text x="150" y="200" fill="#9AA5B5" font-size="11" text-anchor="middle"
        font-family="sans-serif">Órbitas estelares desordenadas · sin plano preferente</text>
</svg>""")


def svg_espiral() -> str:
    """Espiral: disco en rotación con brazos como ondas de densidad."""
    def brazo(fase, color, op):
        pts = []
        for k in range(60):
            import math
            t = k / 59 * 3.4
            r = 8 + t * 24
            a = t * 1.5 + fase
            pts.append(f"{150 + r*math.cos(a):.1f},{110 + r*0.45*math.sin(a):.1f}")
        return (f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" '
                f'stroke-width="7" stroke-linecap="round" opacity="{op}"/>')

    return _wrap(f"""
<svg viewBox="0 0 300 220" width="100%" style="max-width:420px">
  <defs>
    <radialGradient id="gd" cx="50%" cy="50%">
      <stop offset="0%" stop-color="#FFE9B0"/>
      <stop offset="25%" stop-color="#E8B44A" stop-opacity="0.8"/>
      <stop offset="100%" stop-color="#2A4A7A" stop-opacity="0"/>
    </radialGradient>
  </defs>
  <rect width="300" height="220" fill="#0B0E17" rx="8"/>
  <g>
    <animateTransform attributeName="transform" type="rotate"
      from="0 150 110" to="360 150 110" dur="22s" repeatCount="indefinite"/>
    <ellipse cx="150" cy="110" rx="98" ry="46" fill="#2A4A7A" opacity="0.30"/>
    {brazo(0, "#6FA8DC", 0.85)}
    {brazo(3.1416, "#6FA8DC", 0.85)}
    {brazo(0.35, "#BFE0FF", 0.45)}
    {brazo(3.49, "#BFE0FF", 0.45)}
  </g>
  <ellipse cx="150" cy="110" rx="34" ry="24" fill="url(#gd)"/>
  <text x="150" y="200" fill="#9AA5B5" font-size="11" text-anchor="middle"
        font-family="sans-serif">Rotación ordenada · brazos azules por formación estelar</text>
</svg>""")


def svg_disco_canto() -> str:
    """Disco de canto: banda de polvo cruzando el plano medio."""
    return _wrap("""
<svg viewBox="0 0 300 220" width="100%" style="max-width:420px">
  <defs>
    <linearGradient id="gc" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#E8B44A" stop-opacity="0"/>
      <stop offset="30%" stop-color="#E8B44A" stop-opacity="0.75"/>
      <stop offset="50%" stop-color="#FFF3D6"/>
      <stop offset="70%" stop-color="#E8B44A" stop-opacity="0.75"/>
      <stop offset="100%" stop-color="#E8B44A" stop-opacity="0"/>
    </linearGradient>
  </defs>
  <rect width="300" height="220" fill="#0B0E17" rx="8"/>
  <ellipse cx="150" cy="110" rx="118" ry="17" fill="url(#gc)"/>
  <ellipse cx="150" cy="110" rx="34" ry="30" fill="#FFF0CC" opacity="0.9"/>
  <rect x="32" y="107" width="236" height="5" fill="#2A1508" opacity="0.85" rx="2">
    <animate attributeName="opacity" values="0.85;0.55;0.85" dur="4s"
             repeatCount="indefinite"/>
  </rect>
  <text x="150" y="165" fill="#C1440E" font-size="11" text-anchor="middle"
        font-family="sans-serif">banda de polvo</text>
  <text x="150" y="200" fill="#9AA5B5" font-size="11" text-anchor="middle"
        font-family="sans-serif">El mismo disco visto de perfil oculta los brazos</text>
</svg>""")


def svg_puntual() -> str:
    """Estrella/artefacto: PSF puntual con picos de difracción."""
    return _wrap("""
<svg viewBox="0 0 300 220" width="100%" style="max-width:420px">
  <defs>
    <radialGradient id="gp"><stop offset="0%" stop-color="#FFFFFF"/>
      <stop offset="45%" stop-color="#CFE4FF" stop-opacity="0.7"/>
      <stop offset="100%" stop-color="#CFE4FF" stop-opacity="0"/></radialGradient>
  </defs>
  <rect width="300" height="220" fill="#0B0E17" rx="8"/>
  <g opacity="0.85">
    <rect x="148" y="45" width="4" height="130" fill="#CFE4FF" opacity="0.55"/>
    <rect x="85" y="108" width="130" height="4" fill="#CFE4FF" opacity="0.55"/>
  </g>
  <circle cx="150" cy="110" r="34" fill="url(#gp)">
    <animate attributeName="r" values="32;36;32" dur="3s" repeatCount="indefinite"/>
  </circle>
  <text x="150" y="200" fill="#9AA5B5" font-size="11" text-anchor="middle"
        font-family="sans-serif">Perfil idéntico a la PSF del instrumento · no está resuelta</text>
</svg>""")


# --------------------------------------------------------------------------- #
# Elipticidad: qué significa el número en E0–E7
# --------------------------------------------------------------------------- #
def svg_elipticidad(n: int) -> str:
    """Muestra una elíptica En con su relación de ejes."""
    b_a = 1 - n / 10
    rx, ry = 78, 78 * b_a
    return _wrap(f"""
<svg viewBox="0 0 300 200" width="100%" style="max-width:400px">
  <rect width="300" height="200" fill="#0B0E17" rx="8"/>
  <defs><radialGradient id="ge{n}">
    <stop offset="0%" stop-color="#FFF3D6"/>
    <stop offset="45%" stop-color="#E8B44A" stop-opacity="0.8"/>
    <stop offset="100%" stop-color="#C1440E" stop-opacity="0"/>
  </radialGradient></defs>
  <ellipse cx="150" cy="100" rx="{rx}" ry="{ry:.1f}" fill="url(#ge{n})"/>
  <line x1="{150-rx}" y1="100" x2="{150+rx}" y2="100"
        stroke="#4A90D9" stroke-width="1.5" stroke-dasharray="4 3"/>
  <line x1="150" y1="{100-ry:.1f}" x2="150" y2="{100+ry:.1f}"
        stroke="#D4A032" stroke-width="1.5" stroke-dasharray="4 3"/>
  <text x="{150+rx+6}" y="104" fill="#4A90D9" font-size="13"
        font-family="serif" font-style="italic">a</text>
  <text x="156" y="{100-ry+14:.1f}" fill="#D4A032" font-size="13"
        font-family="serif" font-style="italic">b</text>
  <text x="150" y="188" fill="#E6EAF0" font-size="13" text-anchor="middle"
        font-family="sans-serif">E{n} &#8594; b/a = {b_a:.1f}</text>
</svg>""")


# --------------------------------------------------------------------------- #
# Secuencia de Hubble
# --------------------------------------------------------------------------- #
def svg_diapason(resaltar: str | None = None) -> str:
    """Diagrama diapasón de Hubble."""
    def gal(x, y, rx, ry, color, etiqueta, rot=0, activo=False):
        op = 1.0 if (activo or resaltar is None) else 0.32
        anillo = (f'<ellipse cx="{x}" cy="{y}" rx="{rx+7}" ry="{ry+7}" fill="none" '
                  f'stroke="#D4A032" stroke-width="2"/>') if activo else ""
        return f"""
        <g opacity="{op}">
          {anillo}
          <ellipse cx="{x}" cy="{y}" rx="{rx}" ry="{ry}" fill="{color}"
                   transform="rotate({rot} {x} {y})"/>
          <text x="{x}" y="{y+ry+16}" fill="#E6EAF0" font-size="11"
                text-anchor="middle" font-family="sans-serif">{etiqueta}</text>
        </g>"""

    return _wrap(f"""
<svg viewBox="0 0 620 260" width="100%" style="max-width:820px">
  <rect width="620" height="260" fill="#0B0E17" rx="8"/>
  <path d="M 150 130 L 250 130" stroke="#5F6978" stroke-width="2"/>
  <path d="M 250 130 Q 300 130 330 78" stroke="#5F6978" stroke-width="2" fill="none"/>
  <path d="M 250 130 Q 300 130 330 186" stroke="#5F6978" stroke-width="2" fill="none"/>

  {gal(55, 130, 26, 25, "#E8B44A", "E0", 0, resaltar == "E0")}
  {gal(115, 130, 30, 15, "#E8B44A", "E7", -20, resaltar == "E7")}
  {gal(212, 130, 30, 17, "#D8C39A", "S0", -15, resaltar == "S0")}

  {gal(360, 70, 26, 14, "#6FA8DC", "Sa", -18, resaltar == "Sa")}
  {gal(440, 62, 30, 15, "#6FA8DC", "Sb", -18, resaltar == "Sb")}
  {gal(525, 55, 34, 16, "#8FC4F0", "Sc", -18, resaltar == "Sc")}

  {gal(360, 192, 26, 14, "#6FA8DC", "SBa", -18, resaltar == "SBa")}
  {gal(440, 200, 30, 15, "#6FA8DC", "SBb", -18, resaltar == "SBb")}
  {gal(525, 208, 34, 16, "#8FC4F0", "SBc", -18, resaltar == "SBc")}

  <text x="95" y="34" fill="#9AA5B5" font-size="12" font-family="sans-serif">Elípticas</text>
  <text x="196" y="34" fill="#9AA5B5" font-size="12" font-family="sans-serif">Lenticulares</text>
  <text x="420" y="24" fill="#9AA5B5" font-size="12" font-family="sans-serif">Espirales normales</text>
  <text x="420" y="248" fill="#9AA5B5" font-size="12" font-family="sans-serif">Espirales barradas</text>
</svg>""")


# --------------------------------------------------------------------------- #
# Árbol de decisión de Galaxy Zoo 2
# --------------------------------------------------------------------------- #
def svg_arbol_gz2(activa: str | None = None) -> str:
    """Esquema del árbol GZ2 con la pregunta activa resaltada."""
    def nodo(x, y, w, h, texto, qid, hoja=False):
        act = (qid == activa)
        fill = "#D4A032" if act else ("#1E3A6E" if not hoja else "#3A2E1A")
        stroke = "#FFD98A" if act else "#4A5A78"
        tcol = "#0B0E17" if act else "#E6EAF0"
        return f"""
        <g>
          <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="7"
                fill="{fill}" stroke="{stroke}" stroke-width="{2.5 if act else 1.2}"/>
          <text x="{x+w/2}" y="{y+h/2+4}" fill="{tcol}" font-size="11"
                text-anchor="middle" font-family="sans-serif">{texto}</text>
        </g>"""

    return _wrap(f"""
<svg viewBox="0 0 640 330" width="100%" style="max-width:860px">
  <rect width="640" height="330" fill="#0B0E17" rx="8"/>
  <g stroke="#4A5A78" stroke-width="1.4" fill="none">
    <path d="M 130 60 L 175 40"/><path d="M 130 70 L 175 110"/><path d="M 130 80 L 175 285"/>
    <path d="M 285 110 L 330 78"/><path d="M 285 122 L 330 165"/>
    <path d="M 440 165 L 490 140"/><path d="M 440 178 L 490 218"/>
  </g>

  {nodo(15, 50, 115, 34, "Q1: ¿lisa o con", "Q1")}
  <text x="72" y="78" fill="{'#0B0E17' if activa=='Q1' else '#E6EAF0'}"
        font-size="11" text-anchor="middle" font-family="sans-serif">estructura?</text>

  {nodo(175, 25, 105, 30, "Smooth", "L1", True)}
  {nodo(175, 100, 110, 34, "Q2: ¿de canto?", "Q2")}
  {nodo(175, 270, 105, 30, "Star/Artifact", "L2", True)}

  {nodo(330, 62, 105, 30, "Disk (canto)", "L3", True)}
  {nodo(330, 150, 110, 34, "Q4: ¿brazos?", "Q4")}

  {nodo(490, 124, 105, 30, "Spiral", "L4", True)}
  {nodo(490, 202, 105, 30, "Disk (cara)", "L5", True)}

  <text x="320" y="318" fill="#9AA5B5" font-size="11" text-anchor="middle"
        font-family="sans-serif">Q4 nunca se formula a los discos de canto: esa rama termina en Q2</text>
</svg>""")


# --------------------------------------------------------------------------- #
# Recorte y estiramiento asinh
# --------------------------------------------------------------------------- #
def svg_recorte() -> str:
    """Por qué se recorta la región central."""
    return _wrap("""
<svg viewBox="0 0 320 240" width="100%" style="max-width:420px">
  <rect width="320" height="240" fill="#0B0E17" rx="8"/>
  <rect x="35" y="18" width="204" height="204" fill="#101828" stroke="#4A5A78"
        stroke-width="1" stroke-dasharray="4 3"/>
  <rect x="85" y="68" width="104" height="104" fill="none" stroke="#D4A032"
        stroke-width="2.5"/>
  <defs><radialGradient id="gr">
    <stop offset="0%" stop-color="#FFF3D6"/><stop offset="60%" stop-color="#E8B44A"
    stop-opacity="0.6"/><stop offset="100%" stop-color="#E8B44A" stop-opacity="0"/>
  </radialGradient></defs>
  <ellipse cx="137" cy="120" rx="40" ry="28" fill="url(#gr)"/>
  <circle cx="62" cy="45" r="5" fill="#8FC4F0" opacity="0.8"/>
  <circle cx="215" cy="196" r="4" fill="#F0C36B" opacity="0.8"/>
  <circle cx="205" cy="52" r="6" fill="#CFE4FF" opacity="0.7"/>
  <text x="137" y="15" fill="#9AA5B5" font-size="11" text-anchor="middle"
        font-family="sans-serif">424 × 424 px</text>
  <text x="137" y="236" fill="#D4A032" font-size="11" text-anchor="middle"
        font-family="sans-serif">207 × 207 px (recorte) &#8594; 76 % de píxeles descartados</text>
  <text x="268" y="49" fill="#8FC4F0" font-size="10" font-family="sans-serif">vecinos</text>
</svg>""")


def html_asinh() -> str:
    """Comparación visual lineal vs asinh."""
    return """
<div style="display:flex;gap:14px;flex-wrap:wrap;justify-content:center">
  <svg viewBox="0 0 250 190" width="240">
    <rect width="250" height="190" fill="#0B0E17" rx="8"/>
    <line x1="30" y1="160" x2="230" y2="160" stroke="#5F6978"/>
    <line x1="30" y1="160" x2="30" y2="20" stroke="#5F6978"/>
    <path d="M 30 160 L 230 30" stroke="#4A90D9" stroke-width="2.5" fill="none"/>
    <text x="130" y="182" fill="#9AA5B5" font-size="11" text-anchor="middle"
          font-family="sans-serif">flujo físico</text>
    <text x="130" y="16" fill="#4A90D9" font-size="12" text-anchor="middle"
          font-family="sans-serif">Escala lineal</text>
  </svg>
  <svg viewBox="0 0 250 190" width="240">
    <rect width="250" height="190" fill="#0B0E17" rx="8"/>
    <line x1="30" y1="160" x2="230" y2="160" stroke="#5F6978"/>
    <line x1="30" y1="160" x2="30" y2="20" stroke="#5F6978"/>
    <path d="M 30 160 Q 70 40 130 32 T 230 26" stroke="#D4A032" stroke-width="2.5"
          fill="none"/>
    <text x="130" y="182" fill="#9AA5B5" font-size="11" text-anchor="middle"
          font-family="sans-serif">flujo físico</text>
    <text x="130" y="16" fill="#D4A032" font-size="12" text-anchor="middle"
          font-family="sans-serif">Estiramiento asinh</text>
  </svg>
</div>"""


# --------------------------------------------------------------------------- #
# Bandas fotométricas
# --------------------------------------------------------------------------- #
def svg_bandas() -> str:
    """Curvas de transmisión aproximadas de u g r i z del SDSS."""
    bandas = [
        ("u", 355, "#8A4FCF", 0.35), ("g", 470, "#3FA34D", 0.75),
        ("r", 620, "#C43C2E", 0.80), ("i", 750, "#8C3B2E", 0.65),
        ("z", 890, "#5C3A34", 0.40),
    ]
    def x(lam):  # 300–1000 nm -> 40–580 px
        return 40 + (lam - 300) / 700 * 540

    curvas = "".join(
        f'<path d="M {x(l-70):.0f} 175 Q {x(l):.0f} {175-120*a:.0f} '
        f'{x(l+70):.0f} 175 Z" fill="{c}" opacity="{0.30 if b in "uz" else 0.65}"/>'
        f'<text x="{x(l):.0f}" y="{175-120*a-8:.0f}" fill="{c}" font-size="13" '
        f'text-anchor="middle" font-family="serif" font-style="italic">{b}</text>'
        for b, l, c, a in bandas
    )
    return _wrap(f"""
<svg viewBox="0 0 620 220" width="100%" style="max-width:820px">
  <rect width="620" height="220" fill="#0B0E17" rx="8"/>
  {curvas}
  <line x1="40" y1="175" x2="590" y2="175" stroke="#5F6978" stroke-width="1.2"/>
  <text x="315" y="205" fill="#9AA5B5" font-size="11" text-anchor="middle"
        font-family="sans-serif">longitud de onda (nm) — 300 a 1000</text>
  <text x="{x(470):.0f}" y="196" fill="#3FA34D" font-size="10" text-anchor="middle"
        font-family="sans-serif">&#8594; azul</text>
  <text x="{x(620):.0f}" y="196" fill="#C43C2E" font-size="10" text-anchor="middle"
        font-family="sans-serif">&#8594; verde</text>
  <text x="{x(750):.0f}" y="196" fill="#8C3B2E" font-size="10" text-anchor="middle"
        font-family="sans-serif">&#8594; rojo</text>
</svg>""")


# --------------------------------------------------------------------------- #
# Transición tipo baraja de cartas
# --------------------------------------------------------------------------- #
CSS_BARAJA = """
<style>
@keyframes gxSwipe {
  0%   { transform: translateX(65%) rotate(7deg) scale(0.90); opacity: 0; }
  55%  { transform: translateX(-4%) rotate(-1.5deg) scale(1.02); opacity: 1; }
  100% { transform: translateX(0) rotate(0deg) scale(1); opacity: 1; }
}
.gx-deck { position: relative; }
.gx-deck .gx-card {
  animation: gxSwipe 0.55s cubic-bezier(.22,.9,.3,1) both;
  border-radius: 12px; overflow: hidden;
  box-shadow: 0 10px 28px rgba(0,0,0,.55), 0 0 0 1px rgba(212,160,50,.35);
}
.gx-deck::before, .gx-deck::after {
  content: ""; position: absolute; inset: 0; border-radius: 12px;
  border: 1px solid rgba(212,160,50,.18); z-index: -1;
}
.gx-deck::before { transform: translate(7px, 7px) rotate(1.6deg); }
.gx-deck::after  { transform: translate(14px, 13px) rotate(3.2deg); }
</style>
"""


def carta_baraja(img_b64: str, clave: int) -> str:
    """Envuelve una imagen en la animación de baraja. `clave` fuerza el rerender."""
    return f"""{CSS_BARAJA}
<div class="gx-deck" data-k="{clave}">
  <div class="gx-card">
    <img src="data:image/png;base64,{img_b64}" style="width:100%;display:block"/>
  </div>
</div>"""


def a_base64(arr) -> str:
    """array uint8 (H,W,3) -> PNG en base64."""
    import base64
    import io

    import numpy as np
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.asarray(arr, dtype="uint8")).resize(
        (384, 384), Image.NEAREST).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()
