"""Gráficas interactivas (Plotly) para las páginas de resultados e inicio."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

AZUL = "#4A90D9"
ORO = "#D4A032"
ROJO = "#C1440E"
GRIS = "#5F6978"
FONDO = "rgba(0,0,0,0)"

LAYOUT = dict(
    paper_bgcolor=FONDO, plot_bgcolor=FONDO,
    font=dict(family="sans-serif", size=13),
    margin=dict(l=50, r=20, t=50, b=50),
    hoverlabel=dict(font_size=13),
)


def barras_modelos(metricas: dict, metrica: str = "macro_f1",
                   titulo: str | None = None) -> go.Figure:
    """Ranking de modelos, ordenado y con la métrica elegida."""
    nombres = sorted(metricas, key=lambda k: metricas[k][metrica])
    valores = [metricas[n][metrica] for n in nombres]
    acc = [metricas[n]["accuracy"] for n in nombres]
    colores = [ORO if v == max(valores) else AZUL for v in valores]

    fig = go.Figure(go.Bar(
        x=valores, y=nombres, orientation="h", marker_color=colores,
        text=[f"{v:.4f}" for v in valores], textposition="outside",
        customdata=np.array(acc),
        hovertemplate="<b>%{y}</b><br>" +
                      f"{metrica}: " + "%{x:.4f}<br>" +
                      "Exactitud: %{customdata:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title=titulo or f"Comparación por {metrica}",
        xaxis_title=metrica, yaxis_title=None,
        xaxis=dict(range=[0, 1.05], gridcolor="rgba(255,255,255,.08)"),
        height=max(320, 42 * len(nombres)), **LAYOUT,
    )
    return fig


def radar_modelos(metricas: dict, seleccion: list[str]) -> go.Figure:
    """Perfil multimétrica de varios modelos superpuestos."""
    ejes = ["accuracy", "macro_f1", "roc_auc_macro", "avg_precision_macro"]
    etiquetas = ["Exactitud", "F1-macro", "AUC-ROC", "AP-macro"]
    fig = go.Figure()
    for n in seleccion:
        r = [metricas[n].get(e) or 0 for e in ejes]
        fig.add_trace(go.Scatterpolar(
            r=r + [r[0]], theta=etiquetas + [etiquetas[0]],
            fill="toself", name=n, opacity=0.55,
            hovertemplate="<b>%{fullData.name}</b><br>%{theta}: %{r:.4f}<extra></extra>",
        ))
    fig.update_layout(
        polar=dict(bgcolor=FONDO,
                   radialaxis=dict(visible=True, range=[0.5, 1.0],
                                   gridcolor="rgba(255,255,255,.12)")),
        title="Perfil comparado", height=430, **LAYOUT,
    )
    return fig


def coste_vs_exactitud(metricas: dict) -> go.Figure:
    """
    Exactitud frente a coste de inferencia, con anotaciones explicativas.

    Es la figura que responde a la pregunta práctica: para clasificar millones
    de objetos, ¿cuánto cuesta cada punto de F1?
    """
    fig = go.Figure()
    n_test = next(iter(metricas.values()))["n_test"]

    for n, r in metricas.items():
        t_inf = r.get("t_inferencia_s")
        t_fit = r.get("t_entrenamiento_s")
        if t_inf is None:
            continue
        ms = t_inf / max(n_test, 1) * 1000
        es_mejor = r["macro_f1"] == max(v["macro_f1"] for v in metricas.values())
        fig.add_trace(go.Scatter(
            x=[ms], y=[r["macro_f1"]], mode="markers+text",
            name=n, text=[n], textposition="top center",
            textfont=dict(size=10, color="#C8D0DC"),
            marker=dict(
                size=14 + 26 * min((t_fit or 1) / 900, 1),
                color=ORO if es_mejor else AZUL,
                line=dict(width=1.5, color="white"), opacity=0.85),
            hovertemplate=(f"<b>{n}</b><br>"
                           "Inferencia: %{x:.3f} ms/galaxia<br>"
                           "F1-macro: %{y:.4f}<br>"
                           f"Entrenamiento: {t_fit:.0f} s<br>"
                           "<i>El tamaño del punto es el coste de entrenamiento</i>"
                           "<extra></extra>"),
        ))

    fig.update_layout(
        title="Exactitud frente a coste de inferencia",
        xaxis=dict(title="Tiempo de inferencia (ms por galaxia, escala log)",
                   type="log", gridcolor="rgba(255,255,255,.08)"),
        yaxis=dict(title="F1-macro", gridcolor="rgba(255,255,255,.08)"),
        showlegend=False, height=470, **LAYOUT,
    )
    fig.add_annotation(
        x=0.02, y=0.04, xref="paper", yref="paper", showarrow=False,
        text="&#8592; barato de aplicar a un sondeo completo",
        font=dict(size=11, color=GRIS), xanchor="left")
    fig.add_annotation(
        x=0.98, y=0.04, xref="paper", yref="paper", showarrow=False,
        text="caro por objeto &#8594;", font=dict(size=11, color=GRIS),
        xanchor="right")
    return fig


def matriz_confusion(cm, clases, normalizar: bool = True) -> go.Figure:
    """Matriz de confusión como mapa de calor anotado."""
    cm = np.array(cm, dtype=float)
    conteos = cm.astype(int)
    if normalizar:
        z = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        texto = [[f"<b>{z[i,j]*100:.1f}%</b><br><span style='font-size:10px'>"
                  f"{conteos[i,j]}</span>" for j in range(len(clases))]
                 for i in range(len(clases))]
    else:
        z = cm
        texto = [[str(conteos[i, j]) for j in range(len(clases))]
                 for i in range(len(clases))]

    fig = go.Figure(go.Heatmap(
        z=z, x=[f"{c}" for c in clases], y=[f"{c}" for c in clases],
        text=texto, texttemplate="%{text}", textfont=dict(size=14),
        colorscale=[[0, "#0B1220"], [0.5, "#1E4C8A"], [1, ORO]],
        showscale=False,
        hovertemplate=("Real: <b>%{y}</b><br>Predicho: <b>%{x}</b><br>"
                       "Proporción: %{z:.3f}<extra></extra>"),
    ))
    fig.update_layout(
        title="Matriz de confusión" + (" (normalizada por fila)" if normalizar else ""),
        xaxis=dict(title="Clase predicha por el modelo", side="bottom"),
        yaxis=dict(title="Clase real (Galaxy Zoo)", autorange="reversed"),
        height=430, **LAYOUT,
    )
    return fig


def dona_clases(conteos: dict[str, int]) -> go.Figure:
    """Distribución de clases del conjunto."""
    fig = go.Figure(go.Pie(
        labels=list(conteos), values=list(conteos.values()), hole=0.55,
        marker=dict(colors=[ORO, AZUL, "#8FC4F0", GRIS],
                    line=dict(color="#0B0E17", width=2)),
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>%{value:,} galaxias<br>"
                      "%{percent}<extra></extra>",
    ))
    fig.update_layout(title="Distribución de clases", showlegend=False,
                      height=380, **LAYOUT)
    return fig


def curva_pedagogica(tipo: str = "roc") -> go.Figure:
    """
    Curva ROC o PR sintética anotada, para explicar cómo se leen.
    No son datos del proyecto: es material didáctico.
    """
    x = np.linspace(0.001, 1, 200)
    fig = go.Figure()
    if tipo == "roc":
        for pot, nom, col in [(0.15, "Clasificador excelente", ORO),
                              (0.5, "Clasificador aceptable", AZUL),
                              (1.0, "Azar (inútil)", GRIS)]:
            fig.add_trace(go.Scatter(
                x=x, y=x ** pot, name=nom, mode="lines",
                line=dict(color=col, width=3,
                          dash="dot" if pot == 1.0 else "solid"),
                hovertemplate=f"<b>{nom}</b><br>FPR %{{x:.2f}} &#8594; "
                              "TPR %{y:.2f}<extra></extra>"))
        fig.update_layout(
            title="Cómo se lee una curva ROC",
            xaxis=dict(title="Tasa de falsos positivos (FPR)",
                       gridcolor="rgba(255,255,255,.08)"),
            yaxis=dict(title="Tasa de verdaderos positivos = recall",
                       gridcolor="rgba(255,255,255,.08)"))
        fig.add_annotation(x=0.18, y=0.92, text="mejor aquí &#8598;",
                           showarrow=False, font=dict(color=ORO, size=12))
    else:
        base = 0.3
        for pot, nom, col in [(6, "Clasificador excelente", ORO),
                              (1.6, "Clasificador aceptable", AZUL)]:
            y = base + (1 - base) * (1 - x) ** (1 / pot)
            fig.add_trace(go.Scatter(
                x=x, y=y, name=nom, mode="lines",
                line=dict(color=col, width=3),
                hovertemplate=f"<b>{nom}</b><br>Recall %{{x:.2f}} &#8594; "
                              "Precisión %{y:.2f}<extra></extra>"))
        fig.add_hline(y=base, line=dict(color=GRIS, dash="dot", width=2),
                      annotation_text="línea base = prevalencia de la clase",
                      annotation_font_color=GRIS)
        fig.update_layout(
            title="Cómo se lee una curva precisión-recall",
            xaxis=dict(title="Recall (¿cuántas de las reales encontré?)",
                       gridcolor="rgba(255,255,255,.08)"),
            yaxis=dict(title="Precisión (¿cuántas de mis predicciones acerté?)",
                       range=[0, 1.05], gridcolor="rgba(255,255,255,.08)"))
    fig.update_layout(height=420, legend=dict(orientation="h", y=-0.22), **LAYOUT)
    return fig


def barras_probabilidad(clases, proba) -> go.Figure:
    """Probabilidades por clase de una predicción individual."""
    orden = np.argsort(proba)
    fig = go.Figure(go.Bar(
        x=[proba[i] for i in orden], y=[clases[i] for i in orden],
        orientation="h",
        marker_color=[ORO if i == int(np.argmax(proba)) else AZUL for i in orden],
        text=[f"{proba[i]*100:.1f}%" for i in orden], textposition="outside",
        hovertemplate="<b>%{y}</b>: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        xaxis=dict(range=[0, 1.15], showticklabels=False,
                   gridcolor="rgba(255,255,255,.06)"),
        height=170, margin=dict(l=70, r=20, t=8, b=8),
        paper_bgcolor=FONDO, plot_bgcolor=FONDO,
        font=dict(family="sans-serif", size=13),
    )
    return fig
