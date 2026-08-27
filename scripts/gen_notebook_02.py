#!/usr/bin/env python3
"""Genera notebooks/02_dia2_knn_mlp.ipynb"""
import json
from pathlib import Path

md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s.strip().split("\n")}
code = lambda s: {"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": s.strip().split("\n")}

cells = [
md("""
# GalaxIA — Día 2: KNN y redes densas

Cuatro modelos nuevos, todos sobre el **mismo split congelado** del día 1:

| Modelo | Entradas | Qué prueba |
|---|---|---|
| KNN | PCA de píxeles | Calidad de la representación PCA |
| MLP-PCA | PCA de píxeles | Frontera no lineal sobre los mismos datos |
| MLP-Params | 24 parámetros morfológicos | **Experimento tipo Banerji 2010** |
| MLP-Fusión | PCA + parámetros | ¿Suman o se solapan? |

La CNN se lanza aparte desde terminal (última sección) para no bloquear el notebook.

**Referencia del día 1** — SVM (RGB + recorte): acc 0.8616, F1-macro 0.8357, AUC 0.9505.
"""),

code("""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from galaxia import data, labels, features, models, evaluate
data.fijar_semillas(42)
"""),

code("""
# Mismo etiquetado y mismo umbral que el día 1
df = labels.construir_etiquetas(data.cargar_csv())

p = data.DIR_PROC / "umbral.json"
UMBRAL = json.loads(p.read_text())["umbral"] if p.exists() else 0.6
print("Umbral:", UMBRAL)

df_clean = labels.filtrar_por_confianza(df, umbral=UMBRAL)
d = data.preparar_conjunto(df_clean, size=128)
clases, le = d["clases"], d["label_encoder"]

ytr, yva, yte = d["train"]["y"], d["val"]["y"], d["test"]["y"]
print(f"train {len(ytr)} | val {len(yva)} | test {len(yte)}")
print(dict(zip(clases, np.bincount(ytr))))
"""),

md("""
## 1. Características

Dos representaciones. Los píxeles a 64x64 RGB (12 288 dims) van al PCA;
los 24 parámetros morfológicos van directos.
"""),

code("""
t0 = time.time()
Xtr_px = features.a_vector(np.array(d["train"]["X"]), size=64)
Xva_px = features.a_vector(np.array(d["val"]["X"]),   size=64)
Xte_px = features.a_vector(np.array(d["test"]["X"]),  size=64)
print(f"píxeles: {Xtr_px.shape}  ({time.time()-t0:.0f}s)")
"""),

code("""
# Parámetros morfológicos (generados en el día 1). Si no existen, se calculan.
ruta_par = data.DIR_PROC / "parametros.npz"
if not ruta_par.exists():
    imgs, ids_cache = data.cargar_cache(128)
    P = features.lote_parametros(np.array(imgs))
    np.savez_compressed(ruta_par, P=P, ids=ids_cache,
                        nombres=np.array(features.NOMBRES_PARAMETROS))

z = np.load(ruta_par, allow_pickle=True)
mapa = {int(g): i for i, g in enumerate(z["ids"])}
sel = lambda ids: z["P"][[mapa[int(g)] for g in ids]]

Xtr_pa = sel(d["train"]["ids"])
Xva_pa = sel(d["val"]["ids"])
Xte_pa = sel(d["test"]["ids"])
print("parámetros:", Xtr_pa.shape)
"""),

md("""
## 2. KNN con GridSearchCV

El `Pipeline` lleva `memory=`, así que el PCA se calcula una vez por pliegue y
no una vez por combinación de hiperparámetros: 30 combinaciones x 5 pliegues
pasan de ~30 min a ~2-3 min.

KNN **sin** reducción de dimensionalidad colapsa por la maldición de la
dimensionalidad: en 12 288 dims todas las distancias se parecen. El PCA no es
opcional aquí, es lo que hace viable el método.
"""),

code("""
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

pipe = models.pipeline_pca(KNeighborsClassifier(n_jobs=-1), 0.95,
                           cache_dir=str(data.DIR_PROC / "cache_sk"))
gs = GridSearchCV(pipe, models.grid_knn(), cv=5, scoring="f1_macro",
                  n_jobs=-1, verbose=1, refit=True)

t0 = time.time(); gs.fit(Xtr_px, ytr); t_fit = time.time() - t0
print(f"\\n{t_fit/60:.1f} min | mejor F1-macro CV: {gs.best_score_:.4f}")
print(gs.best_params_)
"""),

code("""
t0 = time.time(); proba = gs.predict_proba(Xte_px); t_inf = time.time() - t0
r_knn = evaluate.evaluar("KNN", yte, proba.argmax(1), proba, clases, t_fit, t_inf)
print(f"KNN: acc={r_knn['accuracy']:.4f}  F1-macro={r_knn['macro_f1']:.4f}"
      f"  AUC={r_knn['roc_auc_macro']:.4f}")
np.savez_compressed(evaluate.DIR_REP / "pred_knn.npz",
                    y_true=yte, y_pred=proba.argmax(1), proba=proba,
                    ids=d["test"]["ids"])
"""),

code("""
# F1-macro vs k: muestra el sobreajuste en k pequeño y el subajuste en k grande.
cv = pd.DataFrame(gs.cv_results_)
fig, ax = plt.subplots(figsize=(7, 4.5))
for (w, m), g in cv.groupby(["param_clf__weights", "param_clf__metric"]):
    g = g.sort_values("param_clf__n_neighbors")
    ax.plot(g.param_clf__n_neighbors, g.mean_test_score, "o-", label=f"{w}/{m}", alpha=.8)
ax.set_xlabel("k (vecinos)"); ax.set_ylabel("F1-macro (CV)")
ax.set_title("KNN: sensibilidad a k"); ax.legend(fontsize=8); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig(evaluate.DIR_FIG / "knn_k.png", dpi=150); plt.show()
"""),

md("""
### Coste de inferencia

KNN no entrena pero **predice lento**: es O(n) por consulta. Es el contraste
exacto con la CNN, que es lo contrario. Ese par de números es una figura
interesante para la ponencia, más allá de la exactitud.
"""),

code("""
print(f"KNN   -> entrenar {r_knn['t_entrenamiento_s']:.0f}s | "
      f"predecir {len(yte)} galaxias: {r_knn['t_inferencia_s']:.2f}s "
      f"({r_knn['t_inferencia_s']/len(yte)*1000:.2f} ms/galaxia)")
"""),

md("""
### Galaxias similares (para la interfaz)

Lo que KNN da y ningún otro modelo de la lista: vecinos reales. En la app se
muestran como "galaxias que se parecen a esta".
"""),

code("""
import joblib
mejor = gs.best_estimator_
Ztr = mejor[:-1].transform(Xtr_px)          # scaler + PCA
Zte = mejor[:-1].transform(Xte_px)

from sklearn.neighbors import NearestNeighbors
nn = NearestNeighbors(n_neighbors=6).fit(Ztr)

joblib.dump({"pipeline": mejor, "nn": nn,
             "ids_train": d["train"]["ids"], "y_train": ytr,
             "clases": clases},
            data.RAIZ / "models" / "knn.pkl")

# Demostración visual
_, vec = nn.kneighbors(Zte[:3])
imgs, ids_cache = data.cargar_cache(128)
pos = {int(g): i for i, g in enumerate(ids_cache)}
fig, axes = plt.subplots(3, 6, figsize=(13, 7))
for f in range(3):
    axes[f, 0].imshow(imgs[pos[int(d["test"]["ids"][f])]])
    axes[f, 0].set_title("consulta", fontsize=9)
    for c in range(5):
        axes[f, c+1].imshow(imgs[pos[int(d["train"]["ids"][vec[f, c+1]])]])
        axes[f, c+1].set_title(clases[ytr[vec[f, c+1]]], fontsize=8)
    for a in axes[f]: a.axis("off")
plt.tight_layout(); plt.savefig(evaluate.DIR_FIG / "knn_vecinos.png", dpi=140); plt.show()
"""),

md("""
## 3. MLP sobre PCA

Misma entrada que el SVM del día 1, frontera de decisión distinta.
"""),

code("""
import tensorflow as tf, keras
gpus = tf.config.list_physical_devices("GPU")
for g in gpus: tf.config.experimental.set_memory_growth(g, True)
print("GPU:", gpus)
"""),

code("""
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# El PCA se ajusta SOLO con train. Se reutiliza para val y test.
prep = Pipeline([("sc", StandardScaler()), ("pca", PCA(0.95, random_state=42))])
Ztr = prep.fit_transform(Xtr_px)
Zva, Zte_ = prep.transform(Xva_px), prep.transform(Xte_px)
print("PCA:", Xtr_px.shape[1], "->", Ztr.shape[1], "componentes")
joblib.dump(prep, data.RAIZ / "models" / "pca_px.pkl")
"""),

code("""
def entrenar_mlp(nombre, Xtr, Xva, Xte, capas=(256, 128), epocas=120, batch=256):
    keras.backend.clear_session()
    m = models.build_mlp(Xtr.shape[1], len(clases), capas=capas)
    ruta = str(data.RAIZ / "models" / f"{evaluate._slug(nombre)}.keras")
    t0 = time.time()
    h = m.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=epocas, batch_size=batch,
              class_weight=models.pesos_clase(ytr),
              callbacks=models.callbacks_estandar(ruta, paciencia=15), verbose=0)
    t_fit = time.time() - t0
    t0 = time.time(); proba = np.asarray(m.predict(Xte, verbose=0), dtype=float)
    t_inf = time.time() - t0
    r = evaluate.evaluar(nombre, yte, proba.argmax(1), proba, clases, t_fit, t_inf)
    np.savez_compressed(evaluate.DIR_REP / f"pred_{evaluate._slug(nombre)}.npz",
                        y_true=yte, y_pred=proba.argmax(1), proba=proba,
                        ids=d["test"]["ids"])
    print(f"{nombre:16s} acc={r['accuracy']:.4f}  F1-macro={r['macro_f1']:.4f}"
          f"  AUC={r['roc_auc_macro']:.4f}  ({t_fit:.0f}s, {len(h.history['loss'])} ép.)")
    return m, h, r

mlp_pca, h_pca, r_pca = entrenar_mlp("MLP-PCA", Ztr, Zva, Zte_)
"""),

md("""
## 4. MLP sobre parámetros morfológicos — experimento tipo Banerji

Aquí no hay píxeles: solo 24 números interpretables (concentración, colores
proxy, asimetría, Gini, Hu, textura). Es la réplica del enfoque de Banerji et
al. 2010 que el resumen promete.

**Al presentar**: los JPEG del Galaxy Zoo Challenge tienen estiramiento asinh no
lineal, así que colores y concentración son *proxies derivados de la imagen*, no
fotometría SDSS calibrada. Decirlo explícitamente.
"""),

code("""
sc_pa = StandardScaler().fit(Xtr_pa)
Atr, Ava, Ate = sc_pa.transform(Xtr_pa), sc_pa.transform(Xva_pa), sc_pa.transform(Xte_pa)
joblib.dump(sc_pa, data.RAIZ / "models" / "scaler_params.pkl")

mlp_pa, h_pa, r_pa = entrenar_mlp("MLP-Params", Atr, Ava, Ate, capas=(128, 64))
"""),

code("""
# Fusión: ¿los parámetros aportan algo que el PCA no captura ya?
Ftr = np.hstack([Ztr, Atr]); Fva = np.hstack([Zva, Ava]); Fte = np.hstack([Zte_, Ate])
mlp_fu, h_fu, r_fu = entrenar_mlp("MLP-Fusion", Ftr, Fva, Fte, capas=(384, 192))
"""),

code("""
# ¿Qué parámetros pesan? Permutación sobre el modelo de 24 dims.
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator, ClassifierMixin

class Wrap(BaseEstimator, ClassifierMixin):
    def __init__(s, m): s.m = m; s.classes_ = np.arange(len(clases))
    def fit(s, X, y): return s
    def predict(s, X): return np.asarray(s.m.predict(X, verbose=0)).argmax(1)

imp = permutation_importance(Wrap(mlp_pa), Ate, yte, n_repeats=6,
                             random_state=42, scoring="f1_macro")
orden = imp.importances_mean.argsort()[::-1][:12]
fig, ax = plt.subplots(figsize=(7, 5))
ax.barh([features.NOMBRES_PARAMETROS[i] for i in orden][::-1],
        imp.importances_mean[orden][::-1],
        xerr=imp.importances_std[orden][::-1])
ax.set_xlabel("Caída de F1-macro al permutar")
ax.set_title("Importancia de los parámetros morfológicos")
plt.tight_layout(); plt.savefig(evaluate.DIR_FIG / "importancia_params.png", dpi=150)
plt.show()
"""),

md("""
Si `C_concentracion` y los colores encabezan la lista, el resultado reproduce lo
que reporta la literatura y es una frase directa para la ponencia.
"""),

md("""
## 5. Comparación parcial
"""),

code("""
display(evaluate.tabla_comparativa())
"""),

code("""
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
for nom, h in [("MLP-PCA", h_pca), ("MLP-Params", h_pa), ("MLP-Fusion", h_fu)]:
    ax[0].plot(h.history["val_loss"], label=nom)
    ax[1].plot(h.history["val_accuracy"], label=nom)
ax[0].set_title("Pérdida (validación)"); ax[1].set_title("Exactitud (validación)")
for a in ax: a.set_xlabel("Época"); a.legend(); a.grid(alpha=.3)
plt.tight_layout(); plt.savefig(evaluate.DIR_FIG / "curvas_mlp.png", dpi=150); plt.show()
"""),

md("""
## 6. Lanzar la CNN

Desde **otra terminal**, para que el notebook siga libre:

```bash
cd ~/Documentos/GalaxIA && source GalIA/bin/activate
python scripts/train_cnn.py --size 128 --epocas 40 --batch 128
```

Unos 25-40 s por época con `mixed_float16`; 40 épocas son 20-25 minutos.
Escribe en `metrics.json` igual que los demás modelos.

Si sobra tiempo, generen el caché de 224 y lancen una segunda corrida: a esa
resolución los brazos espirales se distinguen bastante mejor.

```bash
python scripts/build_cache.py --size 224
python scripts/train_cnn.py --size 224 --epocas 30 --batch 64
```
"""),

md("""
## 7. Cierre del día 2

- [ ] KNN, MLP-PCA, MLP-Params y MLP-Fusion en `metrics.json`
- [ ] `models/knn.pkl` con el índice de vecinos para la interfaz
- [ ] `models/pca_px.pkl` y `models/scaler_params.pkl` guardados
- [ ] `pred_*.npz` de cada modelo (los necesita McNemar el día 3)
- [ ] CNN lanzada
- [ ] Figura de importancia de parámetros revisada
""")
]

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "GalaxIA", "language": "python",
                                  "name": "galaxia"},
                   "language_info": {"name": "python", "version": "3.12"}},
      "nbformat": 4, "nbformat_minor": 5}

out = Path(__file__).resolve().parents[1] / "notebooks" / "02_dia2_knn_mlp.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print("Escrito:", out)
