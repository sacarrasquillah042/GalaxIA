# Interfaz GalaxIA

## Requisitos nuevos

```bash
pip install -r requirements.txt   # ahora incluye plotly y streamlit>=1.40
```

`st.segmented_control` y `st.html` necesitan Streamlit 1.40 o superior.
Si la versión instalada es anterior, la app fallará al arrancar.

## Arrancar

```bash
python scripts/export_artefactos.py   # genera galería y muestra de test
streamlit run app/Inicio.py
```

## Estructura

```
app/
├── Inicio.py                 portada con métricas y gráficas
├── pages/
│   ├── 1_Tipologias.py       tipos, Hubble, árbol GZ2
│   ├── 2_Clasificador.py     predicción, Grad-CAM, vecinos
│   └── 3_Resultados.py       comparación, métricas, metodología
├── componentes/
│   ├── visual.py             ilustraciones SVG animadas
│   └── graficos.py           gráficas Plotly interactivas
├── contenido/
│   └── tipologias.json       todo el texto astronómico
└── assets/                   generado por export_artefactos.py
```

## Editar el contenido

Todo el texto astronómico vive en `contenido/tipologias.json`. Se puede
modificar sin tocar código: la app lo lee al arrancar.

## Sobre las ilustraciones

Los gráficos explicativos son SVG generados en `componentes/visual.py`, no
imágenes descargadas. Motivos: la app funciona sin conexión (el wifi de un
congreso falla), no depende de licencias de terceros y un esquema a medida
explica mejor un concepto concreto que una fotografía genérica.

Las únicas fotografías son las del propio SDSS, ya presentes en el conjunto de
datos, con la atribución indicada en la galería.
