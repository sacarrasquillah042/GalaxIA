# =====================================================================
# PARCHE — reemplaza la celda del GridSearchCV del KNN (sección 2)
#
# Problema: PCA(n_components=0.95) obliga a scikit-learn a usar SVD
# completo (necesita todos los valores singulares para elegir cuántos
# conservar). Sobre ~20000 x 12288 eso pide varios GB de workspace, y
# GridSearchCV(n_jobs=-1) lo hacía 16 veces en paralelo -> el kernel muere.
#
# Solución: calcular el PCA UNA vez con solver aleatorizado, y buscar los
# hiperparámetros del KNN sobre los datos ya reducidos.
# =====================================================================
import time
import numpy as np
import joblib
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------
# 1. PCA una sola vez, ajustado SOLO con train.
#    'randomized' es aproximado pero rapidísimo y con 400 componentes
#    la aproximación es excelente para este rango de varianza.
# ---------------------------------------------------------------------
N_MAX = 400
prep = Pipeline([
    ("sc", StandardScaler()),
    ("pca", PCA(n_components=N_MAX, svd_solver="randomized", random_state=42)),
])

t0 = time.time()
Ztr_full = prep.fit_transform(Xtr_px)
print(f"PCA ajustado en {time.time()-t0:.0f}s")

# Elegir k para el 95 % de varianza, ya con los valores singulares en mano
var = np.cumsum(prep["pca"].explained_variance_ratio_)
if var[-1] < 0.95:
    print(f"AVISO: {N_MAX} componentes solo cubren el {var[-1]*100:.1f} %. "
          f"Sube N_MAX a 800 y vuelve a correr esta celda.")
    K = N_MAX
else:
    K = int(np.searchsorted(var, 0.95) + 1)
print(f"{K} componentes explican el {var[K-1]*100:.1f} % de la varianza "
      f"(de {Xtr_px.shape[1]} dimensiones originales)")

Ztr = Ztr_full[:, :K]
Zva = prep.transform(Xva_px)[:, :K]
Zte_ = prep.transform(Xte_px)[:, :K]
del Ztr_full

joblib.dump({"prep": prep, "K": K}, data.RAIZ / "models" / "pca_px.pkl")
print("RAM del conjunto reducido:", round(Ztr.nbytes / 1e6, 1), "MB")

# ---------------------------------------------------------------------
# 2. GridSearch solo sobre el KNN. Ahora cada ajuste es trivial.
#
#    NOTA METODOLÓGICA (documentar en la ponencia): el PCA se ajusta con
#    todo el train antes de la validación cruzada, así que hay un contacto
#    leve entre pliegues al SELECCIONAR hiperparámetros. El conjunto de
#    test permanece intacto, de modo que la métrica final que se reporta
#    sigue siendo limpia.
# ---------------------------------------------------------------------
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

grid = {
    "n_neighbors": [3, 7, 15, 25, 41],
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan", "cosine"],
}
gs = GridSearchCV(KNeighborsClassifier(), grid, cv=5, scoring="f1_macro",
                  n_jobs=4, verbose=1)   # 4, no -1: cada worker copia los datos

t0 = time.time(); gs.fit(Ztr, ytr); t_fit = time.time() - t0
print(f"\n{t_fit/60:.1f} min | mejor F1-macro (CV): {gs.best_score_:.4f}")
print(gs.best_params_)

# ---------------------------------------------------------------------
# 3. Evaluación sobre el test congelado
# ---------------------------------------------------------------------
t0 = time.time(); proba = gs.predict_proba(Zte_); t_inf = time.time() - t0
r_knn = evaluate.evaluar("KNN", yte, proba.argmax(1), proba, clases, t_fit, t_inf)
print(f"KNN: acc={r_knn['accuracy']:.4f}  F1-macro={r_knn['macro_f1']:.4f}"
      f"  AUC={r_knn['roc_auc_macro']:.4f}")

np.savez_compressed(evaluate.DIR_REP / "pred_knn.npz",
                    y_true=yte, y_pred=proba.argmax(1), proba=proba,
                    ids=d["test"]["ids"])

# ---------------------------------------------------------------------
# 4. Índice de vecinos para la interfaz ("galaxias similares")
# ---------------------------------------------------------------------
from sklearn.neighbors import NearestNeighbors
nn = NearestNeighbors(n_neighbors=6, metric=gs.best_params_["metric"]).fit(Ztr)
joblib.dump({"knn": gs.best_estimator_, "prep": prep, "K": K, "nn": nn,
             "ids_train": d["train"]["ids"], "y_train": ytr, "clases": clases},
            data.RAIZ / "models" / "knn.pkl")
print("Guardado models/knn.pkl")
