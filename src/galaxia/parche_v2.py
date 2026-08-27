# =====================================================================
# PARCHE v2 — pegar esta celda DESPUÉS de cargar el CSV y volver a correr
# desde aquí. Corrige el recorrido del árbol (faltaba Q2 = ¿de canto?).
# =====================================================================
import importlib
from galaxia import labels, evaluate
importlib.reload(labels)
importlib.reload(evaluate)

df = data.cargar_csv()
df = labels.construir_etiquetas(df)      # ahora pasa por Q2

labels.diagnostico_arbol(df)

# ---------------------------------------------------------------------
# Elegir el umbral CON LOS DATOS, no a ojo.
# ---------------------------------------------------------------------
import numpy as np
import pandas as pd

conf_n = labels.confianza_normalizada(df)
filas = []
for u in [0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
    sub = df[(conf_n >= u) & (df.label_grouped != "Star/Artifact")]
    vc = sub.label_grouped.value_counts()
    filas.append({
        "umbral": u,
        "N": len(sub),
        "% del total": round(len(sub) / len(df) * 100, 1),
        **{c: int(vc.get(c, 0)) for c in ["Smooth", "Disk", "Spiral"]},
        "min/max clase": round(vc.min() / vc.max(), 3) if len(vc) else 0,
    })
tabla_umbrales = pd.DataFrame(filas)
display(tabla_umbrales)

# ---------------------------------------------------------------------
# Criterio: el umbral más alto que deje >= 2000 galaxias en la clase
# minoritaria. Sube la calidad de etiqueta sin vaciar ninguna clase.
# ---------------------------------------------------------------------
ok = tabla_umbrales[tabla_umbrales[["Smooth", "Disk", "Spiral"]].min(axis=1) >= 2000]
UMBRAL = float(ok.umbral.max()) if len(ok) else 0.5
print(f"\n>>> Umbral elegido: {UMBRAL}")

df_clean = labels.filtrar_por_confianza(df, umbral=UMBRAL, excluir_puntuales=True)
print(f"Conjunto limpio: {len(df_clean)} de {len(df)} ({len(df_clean)/len(df)*100:.1f} %)")
print(df_clean.label_grouped.value_counts().to_string())

# ---------------------------------------------------------------------
# IMPORTANTE: los splits anteriores ya no sirven (las etiquetas cambiaron).
# Regenerarlos UNA vez y subirlos a Git de inmediato.
# ---------------------------------------------------------------------
import os
p = data.DIR_PROC / "splits.json"
if p.exists():
    os.remove(p)
s = data.crear_splits(df_clean)
print({k: len(v) for k, v in s.items() if k != "meta"})

# ---------------------------------------------------------------------
# Verificación visual: 5 ejemplos reales por clase. Si una fila no se
# parece a lo que dice su título, el etiquetado sigue mal.
# ---------------------------------------------------------------------
import matplotlib.pyplot as plt

imgs, ids_cache = data.cargar_cache(128)
pos = {int(g): i for i, g in enumerate(ids_cache)}
clases_v = ["Smooth", "Disk", "Spiral"]

fig, axes = plt.subplots(len(clases_v), 5, figsize=(11, 2.4 * len(clases_v)))
for fila, c in enumerate(clases_v):
    sub = df_clean[df_clean.label_grouped == c].nlargest(200, "confianza")
    gids = sub.GalaxyID.sample(5, random_state=1).tolist()
    for col, g in enumerate(gids):
        ax = axes[fila, col]
        ax.imshow(imgs[pos[int(g)]])
        ax.axis("off")
        if col == 0:
            ax.set_title(c, loc="left", fontsize=11, fontweight="bold")
plt.tight_layout()
plt.show()
