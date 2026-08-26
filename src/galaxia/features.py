"""
Extracción de características a partir de las imágenes SDSS del Galaxy Zoo Challenge.

Dos rutas:

1. ``cargar_imagen`` / ``a_vector``  -> píxeles para modelos clásicos y CNN.
2. ``parametros_morfologicos``       -> vector interpretable de ~24 dimensiones
   (concentración, color, asimetría, Gini, Hu, elipticidad, textura), pensado
   para el experimento tipo Banerji et al. 2010 sin necesidad del cruce SDSS.

ADVERTENCIA IMPORTANTE PARA LA PONENCIA
---------------------------------------
Los JPEG del Galaxy Zoo Challenge son composiciones RGB de las bandas g, r, i
con un estiramiento asinh no lineal aplicado. Por tanto los "colores" y el
"índice de concentración" que se calculan aquí son PROXIES, no fotometría
calibrada. Hay que decirlo explícitamente: son indicadores derivados de la
imagen, no magnitudes SDSS. Presentarlos como fotometría sería falso.
"""
from __future__ import annotations

import os

import cv2
import numpy as np

# La galaxia objetivo ocupa aproximadamente el cuadrado central de 207x207
# dentro de la imagen de 424x424. El resto es cielo y objetos vecinos.
CROP = 207
IMG_ORIG = 424


# --------------------------------------------------------------------------- #
# 1. Carga de píxeles
# --------------------------------------------------------------------------- #
def cargar_imagen(
    galaxy_id: int,
    path_images: str,
    size: int = 128,
    gris: bool = False,
    normalizar: bool = True,
) -> np.ndarray:
    """Lee un JPEG, recorta el centro y redimensiona. Devuelve RGB por defecto."""
    ruta = os.path.join(path_images, f"{galaxy_id}.jpg")
    img = cv2.imread(ruta, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(ruta)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = recortar_centro(img)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    if gris:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img.astype(np.float32) / 255.0 if normalizar else img


def recortar_centro(img: np.ndarray, crop: int = CROP) -> np.ndarray:
    h, w = img.shape[:2]
    y0, x0 = (h - crop) // 2, (w - crop) // 2
    return img[y0 : y0 + crop, x0 : x0 + crop]


def a_vector(imgs: np.ndarray, size: int = 64, gris: bool = False) -> np.ndarray:
    """
    Aplana un lote (N, H, W, 3) uint8 del caché a features para sklearn.

    Con size=64 y RGB salen 12 288 dimensiones; el PCA del pipeline las reduce.
    Mantener el color es deliberado: es el segundo discriminante morfológico
    más fuerte después de la concentración.
    """
    n = len(imgs)
    out = []
    for i in range(n):
        im = imgs[i]
        if im.shape[0] != size:
            im = cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)
        if gris:
            im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        out.append(im.astype(np.float32).ravel() / 255.0)
    return np.asarray(out, dtype=np.float32)


# --------------------------------------------------------------------------- #
# 2. Parámetros morfológicos interpretables
# --------------------------------------------------------------------------- #
NOMBRES_PARAMETROS = [
    "C_concentracion",   # 5*log10(r90/r50) - discriminante elíptica/espiral
    "r50", "r90",
    "color_R_G",         # proxy de (i - r)
    "color_G_B",         # proxy de (r - g)
    "brillo_medio",
    "A_asimetria",       # Conselice: rotación 180 grados
    "S_suavidad",        # clumpiness
    "gini",
    "elipticidad",
    "orientacion_sin", "orientacion_cos",
    "flujo_total",
    "radio_efectivo",
    "hu1", "hu2", "hu3", "hu4", "hu5", "hu6", "hu7",
    "textura_contraste",
    "textura_homogeneidad",
    "textura_energia",
]


def _fondo(img_gray: np.ndarray) -> float:
    """Estima el nivel de cielo con el borde de la imagen recortada."""
    b = 8
    borde = np.concatenate(
        [img_gray[:b].ravel(), img_gray[-b:].ravel(),
         img_gray[:, :b].ravel(), img_gray[:, -b:].ravel()]
    )
    return float(np.median(borde))


def _radios_flujo(flujo: np.ndarray, fracciones=(0.5, 0.9)) -> tuple[float, ...]:
    """Radios que encierran una fracción dada del flujo total, desde el centro."""
    h, w = flujo.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    orden = np.argsort(r, axis=None)
    r_ord = r.ravel()[orden]
    f_ord = flujo.ravel()[orden]
    acum = np.cumsum(f_ord)
    total = acum[-1]
    if total <= 0:
        return tuple(0.0 for _ in fracciones)
    return tuple(float(r_ord[np.searchsorted(acum, fr * total)]) for fr in fracciones)


def _gini(flujo: np.ndarray) -> float:
    x = np.sort(flujo.ravel())
    x = x[x > 0]
    n = x.size
    if n < 2:
        return 0.0
    mu = x.mean()
    i = np.arange(1, n + 1)
    return float(np.sum((2 * i - n - 1) * x) / (mu * n * (n - 1)))


def parametros_morfologicos(img_uint8: np.ndarray) -> np.ndarray:
    """
    Vector de parámetros interpretables a partir de una imagen RGB uint8
    (ya recortada al centro). Devuelve un array de longitud
    ``len(NOMBRES_PARAMETROS)``.
    """
    img = img_uint8.astype(np.float32)
    R, G, B = img[..., 0], img[..., 1], img[..., 2]

    # Canal r del SDSS ~ canal verde del compuesto. Se usa como banda de trabajo.
    gray = G
    cielo = _fondo(gray)
    flujo = np.clip(gray - cielo, 0, None)
    total = float(flujo.sum()) + 1e-8

    # --- Concentración (índice tipo Petrosian) ---
    r50, r90 = _radios_flujo(flujo)
    C = 5.0 * np.log10((r90 + 1e-6) / (r50 + 1e-6)) if r50 > 0 else 0.0

    # --- Colores proxy: se miden dentro de la apertura r90 ---
    h, w = gray.shape
    yy, xx = np.mgrid[0:h, 0:w]
    rr = np.sqrt((yy - (h - 1) / 2) ** 2 + (xx - (w - 1) / 2) ** 2)
    ap = rr <= max(r90, 3.0)
    fR = max(float(np.clip(R - _fondo(R), 0, None)[ap].mean()), 1e-3)
    fG = max(float(np.clip(G - cielo, 0, None)[ap].mean()), 1e-3)
    fB = max(float(np.clip(B - _fondo(B), 0, None)[ap].mean()), 1e-3)
    color_rg = -2.5 * np.log10(fR / fG)
    color_gb = -2.5 * np.log10(fG / fB)

    # --- Asimetría de Conselice: I vs. I rotada 180 grados ---
    rot = flujo[::-1, ::-1]
    A = float(np.abs(flujo - rot).sum() / (2 * total))

    # --- Suavidad / clumpiness: alta frecuencia residual ---
    suave = cv2.GaussianBlur(flujo, (0, 0), sigmaX=max(r50 / 5.0, 1.0))
    S = float(np.abs(flujo - suave).sum() / total)

    # --- Gini ---
    g = _gini(flujo)

    # --- Momentos: elipticidad, orientación, Hu ---
    m = cv2.moments(flujo)
    if m["m00"] > 0:
        mu20, mu02, mu11 = m["mu20"] / m["m00"], m["mu02"] / m["m00"], m["mu11"] / m["m00"]
        comun = np.sqrt(max((mu20 - mu02) ** 2 + 4 * mu11 ** 2, 0.0))
        l1 = (mu20 + mu02 + comun) / 2
        l2 = (mu20 + mu02 - comun) / 2
        elip = float(1 - np.sqrt(max(l2, 0) / max(l1, 1e-8)))
        theta = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
        r_ef = float(np.sqrt(max(l1, 0)))
    else:
        elip, theta, r_ef = 0.0, 0.0, 0.0

    hu = cv2.HuMoments(m).ravel()
    # Escala logarítmica con signo: los momentos de Hu abarcan varios órdenes.
    hu = np.sign(hu) * np.log10(np.abs(hu) + 1e-30)

    # --- Textura (GLCM) sobre la versión cuantizada a 8 bits ---
    try:
        from skimage.feature import graycomatrix, graycoprops

        q = np.clip(flujo / (flujo.max() + 1e-8) * 31, 0, 31).astype(np.uint8)
        glcm = graycomatrix(q, distances=[3], angles=[0, np.pi / 2],
                            levels=32, symmetric=True, normed=True)
        contraste = float(graycoprops(glcm, "contrast").mean())
        homog = float(graycoprops(glcm, "homogeneity").mean())
        energia = float(graycoprops(glcm, "energy").mean())
    except Exception:
        contraste = homog = energia = 0.0

    vec = np.array(
        [C, r50, r90, color_rg, color_gb, float(flujo.mean()),
         A, S, g, elip, float(np.sin(2 * theta)), float(np.cos(2 * theta)),
         np.log10(total), r_ef, *hu, contraste, homog, energia],
        dtype=np.float32,
    )
    return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)


def lote_parametros(imgs_uint8: np.ndarray, verbose: bool = True) -> np.ndarray:
    """Aplica ``parametros_morfologicos`` a un lote (N, H, W, 3)."""
    from tqdm import tqdm

    it = tqdm(range(len(imgs_uint8)), desc="Parámetros") if verbose else range(len(imgs_uint8))
    return np.stack([parametros_morfologicos(imgs_uint8[i]) for i in it])
