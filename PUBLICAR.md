# Publicar GalaxIA en la web

Objetivo: una URL pública para poner en la última diapositiva de la ponencia.

---

## Por qué hace falta un paso previo

La app local depende de tres cosas que **no pueden vivir en un servidor gratuito**:

| Dependencia | Tamaño / problema |
|---|---|
| `data/processed/images_128.npy` | 3 GB — imposible de subir |
| `data/raw/training_solutions_rev1.csv` | dataset de Kaggle, no se redistribuye |
| TensorFlow + `models/cnn_128.keras` | supera el límite de ~1 GB de RAM |

La solución es **precalcular** en la máquina con GPU y publicar solo el
resultado. La app detecta el paquete y funciona sin TensorFlow.

---

## Paso 1 — Generar el paquete web (máquina con GPU)

```bash
cd ~/Documentos/GalaxIA && source ~/Documentos/Entornos/GalIA/bin/activate
python scripts/export_web.py --n 150
```

Deja en `app/assets/web/` unos 35 MB con:

- `muestra.npz` — 150 galaxias del conjunto de prueba, estratificadas
- `gradcam.npz` — mapas de atención ya calculados con la CNN
- `vecinos.npz` — los 5 vecinos del KNN por cada consulta
- `predicciones.json` — probabilidades de cada modelo
- `resumen.json` — conteos de clases (evita leer el CSV en el servidor)

Si supera 90 MB, baje `--n`. Con `--n 90` bajan a ~20 MB.

**Compruebe que funciona en modo web** antes de publicar:

```bash
mv models models_off        # simula la ausencia del modelo
streamlit run app/Inicio.py # debe funcionar igual, con predicciones precalculadas
mv models_off models
```

---

## Paso 2 — Subir a GitHub

```bash
cd ~/Documentos/GalaxIA
git init
git add -A
git commit -m "GalaxIA: clasificacion morfologica de galaxias"
```

Revise **antes** de subir que no se cuelan archivos pesados:

```bash
git ls-files | xargs -I{} du -h {} 2>/dev/null | sort -rh | head -10
```

Nada debería superar 50 MB. `.gitignore` ya excluye `data/raw/`,
`data/processed/*.npy` y `models/*`, e incluye explícitamente
`app/assets/web/`.

Cree un repositorio en github.com (público es lo más simple) y:

```bash
git remote add origin https://github.com/USUARIO/galaxia.git
git branch -M main
git push -u origin main
```

---

## Paso 3 — Desplegar en Streamlit Community Cloud

1. Entre en https://share.streamlit.io con la cuenta de GitHub.
2. **New app** → seleccione el repositorio y la rama `main`.
3. **Main file path:** `app/Inicio.py`
4. Abra **Advanced settings** y ponga:
   - Python version: **3.12**
   - Requirements file: **`requirements-web.txt`**

   Este último punto es el más importante: si usa `requirements.txt` intentará
   instalar TensorFlow y el despliegue fallará por falta de memoria.

5. **Deploy.** Tarda 3–6 minutos la primera vez.

La URL queda como `https://USUARIO-galaxia.streamlit.app`. Se puede
personalizar desde los ajustes de la app.

---

## Paso 4 — Antes de la ponencia

**Genere el QR** con la URL final (por ejemplo en https://www.qr-code-generator.com)
y póngalo en la última diapositiva, junto a los datos de contacto.

**Despierte la app el día anterior.** Streamlit Cloud duerme las aplicaciones
sin tráfico; la primera visita tras el letargo tarda ~30 s en arrancar. Ábrala
unas horas antes y otra vez justo antes de la sesión.

**Grabe el video de respaldo de 60 segundos.** El wifi de los congresos falla,
y esta es la única parte de la ponencia que depende de una conexión.

---

## Diferencias entre la versión local y la publicada

| Función | Local | Publicada |
|---|---|---|
| Tipologías, Hubble, árbol GZ2 | ✅ | ✅ |
| Resultados y metodología | ✅ | ✅ |
| Clasificar galaxias del catálogo | ✅ | ✅ (precalculado) |
| Grad-CAM | ✅ en vivo | ✅ precalculado |
| Galaxias similares | ✅ | ✅ precalculado |
| **Subir una imagen propia** | ✅ | ❌ requiere el modelo completo |
| Galería completa de tipologías | ✅ | ✅ si subió `app/assets/galeria/` |

La app avisa al usuario cuando está en modo publicado, así que la limitación
queda explicada en pantalla y no parece un fallo.

---

## Si algo falla

**«Error installing requirements»** — casi siempre es que seleccionó
`requirements.txt` en vez de `requirements-web.txt`. Cámbielo en Settings.

**«This app has gone over its resource limits»** — algo está cargando
TensorFlow. Compruebe que `requirements-web.txt` no lo incluye y que
`models/` no se subió al repositorio.

**La app arranca pero el clasificador dice que faltan datos** — no se subió
`app/assets/web/`. Verifique con `git ls-files app/assets/web`.

**Cambios que no aparecen** — haga `git push`; Streamlit Cloud redespliega
automáticamente en un par de minutos.

---

## Alternativas si Streamlit Cloud no convence

- **Hugging Face Spaces** (gratis, admite hasta 16 GB con SDK Streamlit): más
  holgado en memoria, permitiría incluso subir el modelo y clasificar imágenes
  propias. Requiere Git LFS para el `.keras`.
- **Render** o **Railway**: capa gratuita más limitada, con arranques en frío.
- **Túnel temporal** (`ngrok http 8501`) desde su propio equipo: sirve para el
  día de la ponencia, pero la URL muere al cerrar el portátil.
