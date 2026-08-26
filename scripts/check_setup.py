#!/usr/bin/env python3
"""
Verifica que la estructura del proyecto y los datos estén en su sitio.

    python scripts/check_setup.py

Correr después de check_gpu.py y antes de build_cache.py.
"""
from pathlib import Path
import sys

RAIZ = Path(__file__).resolve().parents[1]
OK, FALLO = "  [ok] ", "  [--] "
errores = []


def check(cond, msg_ok, msg_error=None):
    print((OK if cond else FALLO) + (msg_ok if cond else (msg_error or msg_ok)))
    if not cond and msg_error:
        errores.append(msg_error)
    return cond


print(f"Raíz detectada: {RAIZ}\n")

print("Estructura de carpetas")
for d in ["src/galaxia", "scripts", "notebooks", "data/raw",
          "data/processed", "models", "reports/figures"]:
    p = RAIZ / d
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)
        print(f"  [+] {d} (creada)")
    else:
        print(OK + d)

print("\nMódulos")
sys.path.insert(0, str(RAIZ / "src"))
for m in ["labels", "features", "data", "evaluate"]:
    try:
        __import__(f"galaxia.{m}")
        print(OK + f"galaxia.{m}")
    except Exception as e:
        check(False, "", f"galaxia.{m} no importa: {e}")

print("\nDatos")
csv = RAIZ / "data/raw/training_solutions_rev1.csv"
imgs = RAIZ / "data/raw/images_training_rev1"
check(csv.exists(), f"{csv.name}",
      f"Falta {csv} — copiar el CSV de Kaggle ahí")
if check(imgs.is_dir(), "images_training_rev1/",
         f"Falta {imgs} — descomprimir las imágenes ahí"):
    n = sum(1 for _ in imgs.glob("*.jpg"))
    check(n > 60000, f"{n} imágenes .jpg",
          f"Solo {n} imágenes; se esperaban ~61578. ¿Se descomprimió completo?")

print("\nCoherencia de rutas")
try:
    from galaxia import data as gdata
    check(gdata.RAIZ == RAIZ, f"data.RAIZ apunta a {gdata.RAIZ}",
          f"data.RAIZ={gdata.RAIZ} != {RAIZ}. La carpeta del proyecto debe "
          "contener src/galaxia/, no estar anidada de más.")
except Exception:
    pass

print("\nEstado del caché")
for s in (128, 224):
    p = RAIZ / f"data/processed/images_{s}.npy"
    print((OK if p.exists() else "  [ ] ") +
          f"images_{s}.npy" + (f" ({p.stat().st_size/1e9:.2f} GB)" if p.exists() else " (pendiente)"))
sp = RAIZ / "data/processed/splits.json"
print((OK if sp.exists() else "  [ ] ") +
      "splits.json" + ("" if sp.exists() else " (se crea en el notebook 01)"))

print()
if errores:
    print(f"{len(errores)} problema(s) por resolver:")
    for e in errores:
        print(f"  - {e}")
    raise SystemExit(1)
print("Estructura correcta. Siguiente: python scripts/build_cache.py --size 128")
