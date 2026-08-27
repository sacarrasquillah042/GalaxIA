"""
Evaluación unificada: se escribe una vez y sirve para los ocho modelos.

Cubre las métricas prometidas en el resumen COCOA (exactitud, matrices de
confusión, curvas ROC y curvas precisión-recall) más las métricas por clase que
ya existían en el notebook original (Specificity, NPV, G-Mean).

Todos los resultados se acumulan en reports/metrics.json, que es la única
fuente que consume la interfaz. No copiar números a mano a las diapositivas.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.preprocessing import label_binarize  # noqa: E402

RAIZ = Path(__file__).resolve().parents[2]
DIR_REP = RAIZ / "reports"
DIR_FIG = DIR_REP / "figures"


def evaluar(
    nombre: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    class_names: list[str],
    t_entrenamiento: float | None = None,
    t_inferencia: float | None = None,
    guardar: bool = True,
) -> dict:
    """Calcula todas las métricas, genera 3 figuras y devuelve un dict serializable."""
    DIR_FIG.mkdir(parents=True, exist_ok=True)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    y_proba = np.asarray(y_proba)
    n_cls = len(class_names)

    rep = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=range(n_cls))

    # ---- Specificity / NPV / G-Mean por clase (esquema One-vs-Rest) --------
    por_clase = {}
    for i, c in enumerate(class_names):
        TP = int(cm[i, i])
        FN = int(cm[i].sum() - TP)
        FP = int(cm[:, i].sum() - TP)
        TN = int(cm.sum() - TP - FN - FP)
        sens = TP / max(TP + FN, 1)
        spec = TN / max(TN + FP, 1)
        por_clase[c] = {
            "precision": rep[c]["precision"],
            "recall": rep[c]["recall"],
            "f1": rep[c]["f1-score"],
            "support": int(rep[c]["support"]),
            "specificity": spec,
            "npv": TN / max(TN + FN, 1),
            "g_mean": float(np.sqrt(sens * spec)),
        }

    # ---- ROC y Precisión-Recall One-vs-Rest -------------------------------
    y_bin = label_binarize(y_true, classes=list(range(n_cls)))
    if n_cls == 2:
        y_bin = np.hstack([1 - y_bin, y_bin])

    roc, pr = {}, {}
    curvas = {"roc": {}, "pr": {}}
    for i, c in enumerate(class_names):
        if y_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc[c] = float(auc(fpr, tpr))
        curvas["roc"][c] = (fpr, tpr)

        prec, rec, _ = precision_recall_curve(y_bin[:, i], y_proba[:, i])
        pr[c] = float(average_precision_score(y_bin[:, i], y_proba[:, i]))
        curvas["pr"][c] = (rec, prec)

    if guardar:
        _fig_matriz(cm, class_names, nombre)
        _fig_curvas(curvas["roc"], roc, nombre, "roc")
        _fig_curvas(curvas["pr"], pr, nombre, "pr", y_bin=y_bin)

    res = {
        "modelo": nombre,
        "accuracy": rep["accuracy"],
        "macro_f1": rep["macro avg"]["f1-score"],
        "weighted_f1": rep["weighted avg"]["f1-score"],
        "roc_auc": roc,
        "roc_auc_macro": float(np.mean(list(roc.values()))) if roc else None,
        "avg_precision": pr,
        "avg_precision_macro": float(np.mean(list(pr.values()))) if pr else None,
        "por_clase": por_clase,
        "matriz_confusion": cm.tolist(),
        "clases": class_names,
        "t_entrenamiento_s": t_entrenamiento,
        "t_inferencia_s": t_inferencia,
        "n_test": int(len(y_true)),
    }
    if guardar:
        registrar(res)
    return res


# --------------------------------------------------------------------------- #
def _fig_matriz(cm, class_names, nombre):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, norm, tit in zip(axes, [None, "true"], ["Conteos", "Normalizada por fila"]):
        if norm == "true":
            m = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        else:
            m = cm.astype(int)
        ConfusionMatrixDisplay(m, display_labels=class_names).plot(
            ax=ax, cmap="Blues", colorbar=False,
            values_format=".2f" if norm else "d",
        )
        ax.set_title(tit)
        ax.tick_params(axis="x", rotation=45)
    fig.suptitle(f"Matriz de confusión — {nombre}")
    fig.tight_layout()
    fig.savefig(DIR_FIG / f"cm_{_slug(nombre)}.png", dpi=150)
    plt.close(fig)


def _fig_curvas(curvas, scores, nombre, tipo, y_bin=None):
    fig, ax = plt.subplots(figsize=(7, 6))
    for c, (x, y) in curvas.items():
        ax.plot(x, y, lw=2, label=f"{c} ({scores[c]:.3f})")

    if tipo == "roc":
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Azar")
        ax.set_xlabel("Tasa de falsos positivos")
        ax.set_ylabel("Tasa de verdaderos positivos")
        ax.set_title(f"Curvas ROC (AUC) — {nombre}")
        loc = "lower right"
    else:
        if y_bin is not None:
            for i, c in enumerate(curvas):
                base = y_bin[:, i].mean()
                ax.axhline(base, ls=":", lw=0.8, color="gray")
        ax.set_xlabel("Recall (exhaustividad)")
        ax.set_ylabel("Precision (precisión)")
        ax.set_title(f"Curvas precisión-recall (AP) — {nombre}")
        loc = "best"

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc=loc, fontsize=9)
    fig.tight_layout()
    fig.savefig(DIR_FIG / f"{tipo}_{_slug(nombre)}.png", dpi=150)
    plt.close(fig)


def _slug(s: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in s.lower()).strip("_")


# --------------------------------------------------------------------------- #
def registrar(res: dict, ruta: Path | None = None) -> None:
    """Añade o reemplaza el resultado de un modelo en reports/metrics.json."""
    ruta = ruta or (DIR_REP / "metrics.json")
    ruta.parent.mkdir(parents=True, exist_ok=True)
    datos = json.loads(ruta.read_text()) if ruta.exists() else {}
    datos[res["modelo"]] = res
    ruta.write_text(json.dumps(datos, indent=2, ensure_ascii=False))


def tabla_comparativa(ruta: Path | None = None):
    """Tabla maestra de la ponencia, construida desde metrics.json."""
    import pandas as pd

    ruta = ruta or (DIR_REP / "metrics.json")
    datos = json.loads(ruta.read_text())
    filas = [
        {
            "Modelo": m,
            "Accuracy": r["accuracy"],
            "F1-macro": r["macro_f1"],
            "AUC-ROC": r["roc_auc_macro"],
            "AP-macro": r["avg_precision_macro"],
            "t entren. (s)": r["t_entrenamiento_s"],
            "t infer. (s)": r["t_inferencia_s"],
        }
        for m, r in datos.items()
    ]
    return pd.DataFrame(filas).sort_values("F1-macro", ascending=False).round(4)


def mcnemar_modelos(y_true, y_pred_a, y_pred_b, nombre_a="A", nombre_b="B") -> dict:
    """
    Prueba de McNemar: ¿la diferencia entre dos modelos es real o es ruido?

    Necesario para poder escribir en la ponencia que las redes neuronales fueron
    "superiores" con respaldo estadístico y no solo por comparar dos números.
    """
    from statsmodels.stats.contingency_tables import mcnemar

    ok_a = np.asarray(y_pred_a) == np.asarray(y_true)
    ok_b = np.asarray(y_pred_b) == np.asarray(y_true)
    tabla = [
        [int((ok_a & ok_b).sum()), int((ok_a & ~ok_b).sum())],
        [int((~ok_a & ok_b).sum()), int((~ok_a & ~ok_b).sum())],
    ]
    r = mcnemar(tabla, exact=False, correction=True)
    return {
        "comparacion": f"{nombre_a} vs {nombre_b}",
        "tabla": tabla,
        "statistic": float(r.statistic),
        "pvalue": float(r.pvalue),
        "significativo_0.05": bool(r.pvalue < 0.05),
    }


def accuracy_vs_confianza(y_true, y_pred, confianza, bins=8, nombre="modelo"):
    """
    Exactitud en función del consenso de los voluntarios.

    Es la figura que mejor comunica la idea central del trabajo: el modelo falla
    justamente donde los humanos también dudaron.
    """
    conf = np.asarray(confianza)
    ok = np.asarray(y_pred) == np.asarray(y_true)
    bordes = np.linspace(conf.min(), conf.max(), bins + 1)
    centros, accs, ns = [], [], []
    for lo, hi in zip(bordes[:-1], bordes[1:]):
        m = (conf >= lo) & (conf < hi)
        if m.sum() > 20:
            centros.append((lo + hi) / 2)
            accs.append(float(ok[m].mean()))
            ns.append(int(m.sum()))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(centros, accs, "o-", lw=2)
    ax.set_xlabel("Confianza del voto de los voluntarios")
    ax.set_ylabel("Exactitud del modelo")
    ax.set_title(f"Exactitud vs. consenso humano — {nombre}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    DIR_FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(DIR_FIG / f"conf_{_slug(nombre)}.png", dpi=150)
    plt.close(fig)
    return {"centros": centros, "accuracy": accs, "n": ns}
