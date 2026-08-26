#!/usr/bin/env python3
"""Genera notebooks/01_fase0_reparacion.ipynb"""
import json
from pathlib import Path

md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s.strip().split("\n")}
code = lambda s: {"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": s.strip().split("\n")}

cells = [
md("""
# GalaxIA — Fase 0: reparación de la base

**Objetivo del día 1.** Dejar el pipeline en un estado defendible antes de añadir
ningún modelo nuevo. Se corrigen cuatro cosas del notebook original:

1. **Fuga de información**: el `StandardScaler` y el `PCA` se ajustaban sobre el
   dataset completo *antes* del split. Ahora van dentro de un `Pipeline`.
2. **Recorte central + color**: se pasaba la imagen completa de 424×424 a gris de
   64×64. El objeto ocupa solo el centro y el color es discriminante.
3. **Etiquetas con confianza**: se sustituyen los umbrales ad-hoc (0.3, 0.5) por
   comparaciones dentro de cada pregunta, y se guarda la confianza del voto.
4. **Splits congelados**: todos los modelos compartirán los mismos índices.

Al final del notebook tendremos **el número clave**: cuánto sube la exactitud
solo por arreglar el preprocesamiento, sin cambiar de modelo.
"""),

code("""
import sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from galaxia import data, labels, features, evaluate

data.fijar_semillas(42)
pd.set_option("display.width", 120)
print("Raíz del proyecto:", data.RAIZ)
"""),

md("""
## 1. Carga y etiquetado

Ejecuta **antes** desde la terminal, una sola vez (8–12 min con 14 procesos):

```bash
python scripts/build_cache.py --size 128
```
"""),

code("""
df = data.cargar_csv()
print(f"{len(df)} galaxias, {df.shape[1]} columnas")

df = labels.construir_etiquetas(df)
display(labels.resumen(df, umbral=0.7))
"""),

md("""
La columna **Retenido %** es interesante por sí sola: si `Disk` y `Spiral`
retienen mucho menos que `Smooth`, eso indica que la frontera disco/espiral es
donde los voluntarios humanos también dudaron. Es un resultado para la ponencia,
no un problema del código.
"""),

code("""
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
df["confianza"].hist(bins=50, ax=ax[0])
ax[0].axvline(0.7, color="r", ls="--", label="umbral 0.7")
ax[0].set_title("Distribución de la confianza del voto"); ax[0].legend()

for c in ["Smooth", "Disk", "Spiral"]:
    df.loc[df.label_grouped == c, "confianza"].hist(bins=40, alpha=0.5, ax=ax[1], label=c)
ax[1].set_title("Confianza por clase"); ax[1].legend()
plt.tight_layout(); plt.show()
"""),

md("""
## 2. Decisión sobre `Star/Artifact`

Con ~59 ejemplos (0.10 %) esta clase rompe el `macro avg` y sus curvas ROC no son
interpretables. Se saca del problema multiclase — quedan las **tres clases
principales** que menciona el resumen — y se deja como tarea binaria aparte
(`labels.etiqueta_binaria_puntual`) si sobra tiempo el día 3.
"""),

code("""
UMBRAL = 0.7
df_clean = labels.filtrar_por_confianza(df, umbral=UMBRAL, excluir_puntuales=True)
print(f"Conjunto limpio: {len(df_clean)} de {len(df)} ({len(df_clean)/len(df)*100:.1f} %)")
print(df_clean["label_grouped"].value_counts())
"""),

code("""
# Splits congelados. EJECUTAR UNA SOLA VEZ. No regenerar después.
if not (data.DIR_PROC / "splits.json").exists():
    s = data.crear_splits(df_clean)
    print({k: len(v) for k, v in s.items() if k != "meta"})
else:
    print("splits.json ya existe — se reutiliza (correcto).")
"""),

md("""
## 3. Comparación directa: preprocesamiento viejo vs. nuevo

Este es el experimento central del día. Mismo modelo (SVM RBF), misma muestra,
mismo split. Lo único que cambia es cómo se construyen las características.

- **Viejo**: imagen completa → gris 64×64 → escalado y PCA sobre TODO → split
- **Nuevo**: recorte central → RGB 64×64 → split → escalado y PCA solo con train
"""),

code("""
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

d = data.preparar_conjunto(df_clean, size=128)
clases = d["clases"]; le = d["label_encoder"]
print("Clases:", clases)

# Submuestra para que la comparación sea rápida (el run completo va en la sección 4)
N = 6000
rng = np.random.default_rng(42)
sel = rng.choice(len(d["train"]["y"]), size=min(N, len(d["train"]["y"])), replace=False)
Xtr_img = np.array(d["train"]["X"][np.sort(sel)])
ytr = d["train"]["y"][np.sort(sel)]
Xte_img = np.array(d["test"]["X"]); yte = d["test"]["y"]
print("train:", Xtr_img.shape, " test:", Xte_img.shape)
"""),

code("""
def entrenar_evaluar(nombre, Xtr, ytr, Xte, yte, clf, n_comp=0.95):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_comp, random_state=42)),
        ("clf", clf),
    ])
    t0 = time.time(); pipe.fit(Xtr, ytr); t_fit = time.time() - t0
    t0 = time.time(); proba = pipe.predict_proba(Xte); t_inf = time.time() - t0
    pred = proba.argmax(1)
    r = evaluate.evaluar(nombre, yte, pred, proba, clases, t_fit, t_inf)
    print(f"{nombre:28s} acc={r['accuracy']:.4f}  F1-macro={r['macro_f1']:.4f}"
          f"  AUC={r['roc_auc_macro']:.4f}  ({t_fit:.0f}s)")
    return pipe, r
"""),

code("""
# --- VIEJO: gris, sin color (la fuga no se reproduce aquí; se documenta aparte) ---
Xtr_viejo = features.a_vector(Xtr_img, size=64, gris=True)
Xte_viejo = features.a_vector(Xte_img, size=64, gris=True)

# --- NUEVO: recorte central (ya aplicado en el caché) + RGB ---
Xtr_nuevo = features.a_vector(Xtr_img, size=64, gris=False)
Xte_nuevo = features.a_vector(Xte_img, size=64, gris=False)

print("dims:", Xtr_viejo.shape[1], "->", Xtr_nuevo.shape[1])
"""),

code("""
svm = lambda: SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42)
_, r_viejo = entrenar_evaluar("SVM (gris, sin recorte)", Xtr_viejo, ytr, Xte_viejo, yte, svm())
_, r_nuevo = entrenar_evaluar("SVM (RGB + recorte)",     Xtr_nuevo, ytr, Xte_nuevo, yte, svm())

delta = (r_nuevo["accuracy"] - r_viejo["accuracy"]) * 100
print(f"\\n>>> Ganancia solo por preprocesamiento: {delta:+.1f} puntos de exactitud")
"""),

md("""
**Anotar ese número.** Es la justificación de por qué la fase 0 existía, y es un
buen primer resultado para la ponencia: mejora sustancial sin tocar el modelo.
"""),

md("""
## 4. Reentrenar los cuatro modelos originales, ya sin fuga
"""),

code("""
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV

modelos = {
    "Regresión logística": LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1),
    "SVM (RBF)":           SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42),
    "Árbol de decisión":   DecisionTreeClassifier(max_depth=18, class_weight="balanced", random_state=42),
    "SGD (log-loss)":      SGDClassifier(loss="log_loss", class_weight="balanced",
                                         max_iter=2000, random_state=42, n_jobs=-1),
}

resultados = {}
for nombre, clf in modelos.items():
    _, resultados[nombre] = entrenar_evaluar(nombre, Xtr_nuevo, ytr, Xte_nuevo, yte, clf)
"""),

code("""
display(evaluate.tabla_comparativa())
"""),

md("""
## 5. Parámetros morfológicos interpretables

Vector de 24 dimensiones por galaxia (concentración, colores proxy, asimetría,
Gini, momentos de Hu, textura). Alimenta el experimento tipo Banerji del día 2
sin necesidad del cruce con SDSS.

**Al presentar**: los JPEG del Galaxy Zoo Challenge tienen un estiramiento asinh
no lineal, así que estos son *proxies derivados de la imagen*, no fotometría
calibrada. Decirlo explícitamente.
"""),

code("""
ruta_par = data.DIR_PROC / "parametros.npz"
if not ruta_par.exists():
    imgs, ids_cache = data.cargar_cache(128)
    P = features.lote_parametros(np.array(imgs))     # ~4-6 min con 61k imágenes
    np.savez_compressed(ruta_par, P=P, ids=ids_cache,
                        nombres=np.array(features.NOMBRES_PARAMETROS))
    print("Guardado:", ruta_par)
else:
    print("Ya existe:", ruta_par)

z = np.load(ruta_par, allow_pickle=True)
P, ids_par = z["P"], z["ids"]
print(P.shape)
"""),

code("""
# ¿Los parámetros separan las clases? Vistazo rápido antes de invertir el día 2.
dfp = pd.DataFrame(P, columns=features.NOMBRES_PARAMETROS)
dfp["GalaxyID"] = ids_par
dfp = dfp.merge(df_clean[["GalaxyID", "label_grouped"]], on="GalaxyID")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col in zip(axes, ["C_concentracion", "color_R_G", "A_asimetria"]):
    for c in clases:
        v = dfp.loc[dfp.label_grouped == c, col]
        ax.hist(v, bins=60, alpha=0.5, label=c, density=True,
                range=np.percentile(dfp[col], [1, 99]))
    ax.set_title(col); ax.legend(fontsize=8)
plt.tight_layout(); plt.show()
"""),

md("""
Si la concentración y el color separan visiblemente `Smooth` de `Spiral`, el
experimento del día 2 vale la pena. Si no separan nada, hay un error en la
extracción — revisar antes de seguir.
"""),

md("""
## 6. Cierre del día 1

- [ ] `splits.json` creado y **no** se regenera más
- [ ] Ningún transformador ajustado fuera del `Pipeline`
- [ ] `reports/metrics.json` con los 4 modelos base
- [ ] Figuras ROC y precisión-recall en `reports/figures/`
- [ ] Ganancia por preprocesamiento anotada
- [ ] `parametros.npz` generado
- [ ] `pip freeze > requirements-lock-gpu.txt`
""")
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "GalaxIA", "language": "python", "name": "galaxia"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path(__file__).resolve().parents[1] / "notebooks" / "01_fase0_reparacion.ipynb"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print("Escrito:", out)
