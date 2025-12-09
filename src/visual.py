

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve
)
from sklearn.model_selection import learning_curve


def plot_confusion_matrix(y_true, y_pred, labels=None):

    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm.astype("float") / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm_percent, cmap="Blues", vmin=0, vmax=100)

    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")

    if labels:
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)

    for i in range(cm_percent.shape[0]):
        for j in range(cm_percent.shape[1]):
            ax.text(
                j,
                i,
                f"{cm_percent[i, j]:.1f}%",
                ha="center",
                va="center"
            )

    plt.title("Matriz de Confusão (%)")
    plt.colorbar(im, format="%.0f%%")
    plt.tight_layout()
    return fig

def plot_roc_curve(y_true, y_proba):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Curva ROC")
    ax.legend()
    plt.tight_layout()
    return fig

def plot_precision_recall_curve(y_true, y_proba):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Curva Precision–Recall")
    plt.tight_layout()
    return fig

def plot_learning_curve(
    estimator,
    X,
    y,
    cv,
    scoring="f1",
    n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 5)
):
    train_sizes, train_scores, val_scores = learning_curve(
        estimator=estimator,
        X=X,
        y=y,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        train_sizes=train_sizes
    )

    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(train_sizes, train_mean, label="Treino")
    ax.plot(train_sizes, val_mean, label="Validação")

    ax.set_xlabel("Tamanho do conjunto de treino")
    ax.set_ylabel(scoring.upper())
    ax.set_title("Learning Curve")
    ax.legend()
    plt.tight_layout()
    return fig

def plot_model_comparison(
    results_df,
    metrics,
    model_col="model"
):
    if isinstance(metrics, str):
        metrics = [metrics]

    n_models = len(results_df)
    n_metrics = len(metrics)

    x = np.arange(n_models)
    width = 0.8 / n_metrics

    fig, ax = plt.subplots(figsize=(8, 4))

    for i, metric in enumerate(metrics):
        ax.bar(
            x + i * width,
            results_df[metric],
            width,
            label=metric.replace("_mean", "").upper()
        )

    ax.set_xticks(x + width * (n_metrics - 1) / 2)
    ax.set_xticklabels(results_df[model_col], rotation=30, ha="right")

    ax.set_ylabel("Score")
    ax.set_title("Comparação de Modelos por Métricas")
    ax.legend()

    plt.tight_layout()
    return fig

