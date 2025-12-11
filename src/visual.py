import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve
)
from sklearn.model_selection import learning_curve

PALETA = ["#6EC6FF", "#4AA3FF", "#1E88E5", "#0D47A1"]


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
                va="center",
                color="black"
            )

    plt.title("Matriz de Confusão (%)")
    plt.colorbar(im, format="%.0f%%")
    plt.tight_layout()
    return fig


def plot_roc_curve(y_true, y_proba):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}", color=PALETA[2], linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color=PALETA[0])

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Curva ROC")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_precision_recall_curve(y_true, y_proba):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision, color=PALETA[3], linewidth=2)

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
    ax.plot(train_sizes, train_mean, label="Treino", color=PALETA[1], linewidth=2)
    ax.plot(train_sizes, val_mean, label="Validação", color=PALETA[3], linewidth=2)

    ax.set_xlabel("Tamanho do conjunto de treino")
    ax.set_ylabel(scoring.upper())
    ax.set_title("Learning Curve")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_model_comparison(results_dict):
    metricas = list(next(iter(results_dict.values())).keys())

    results_pct = {
        model: [v * 100 for v in results_dict[model].values()]
        for model in results_dict
    }

    medias = {model: np.mean(vals) for model, vals in results_pct.items()}
    ordenados = sorted(medias.items(), key=lambda x: x[1], reverse=True)
    modelos_ordenados = [x[0] for x in ordenados]

    valores_plot = [results_pct[m] for m in modelos_ordenados]

    x = np.arange(len(modelos_ordenados))
    largura = 0.8 / len(metricas)

    plt.figure(figsize=(12, 6))

    for i, metrica in enumerate(metricas):
        valores = [v[i] for v in valores_plot]
        bars = plt.bar(
            x + (i - (len(metricas) - 1) / 2) * largura,
            valores,
            width=largura,
            label=metrica.capitalize(),
            color=PALETA[i % len(PALETA)]
        )
        for bar, v in zip(bars, valores):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{v:.2f}%",
                ha="center",
                va="bottom"
            )

    ymax = max(max(vals) for vals in results_pct.values()) + 5
    plt.ylim(75, ymax)
    plt.ylabel("Percentual (%)")
    plt.xticks(x, modelos_ordenados)
    plt.title("Comparação dos Modelos por Ranking")
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.legend(title="Métricas", bbox_to_anchor=(1.15, 1))
    plt.tight_layout()
    plt.show()


def plot_model_heatmap(results_dict):
    """
    Cria um heatmap comparando todas as métricas de todos os modelos.
    """
    import seaborn as sns

    metricas = list(next(iter(results_dict.values())).keys())
    df = pd.DataFrame({
        model: [v * 100 for v in results_dict[model].values()]
        for model in results_dict
    }, index=metricas).T

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        df,
        annot=True,
        fmt=".2f",
        cmap=sns.color_palette(PALETA, as_cmap=True),
        cbar_kws={"label": "Percentual (%)"},
        linewidths=0.5
    )
    plt.title("Heatmap de Métricas dos Modelos")
    plt.ylabel("Modelos")
    plt.xlabel("Métricas")
    plt.tight_layout()
    plt.show()


def rank_models(results_dict):
    medias = {
        model: np.mean(list(metrics.values()))
        for model, metrics in results_dict.items()
    }
    return sorted(medias.items(), key=lambda x: x[1], reverse=True)
