"""Grafici za rad: raspodela godina, toplotne mape, predviđanja, MAE po starosnim grupama."""

from functools import lru_cache

import joblib
import matplotlib
matplotlib.use("Agg")  # crta direktno u fajl, bez otvaranja prozora
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluate import CROSS_RESULTS_PATH
from src.train import CNNS, DATASETS, ML_MODELS, MODELS_DIR, RESULTS_DIR, load_features

FIGURES_DIR = RESULTS_DIR / "figures"

NAMES = {"appa": "APPA-REAL", "fgnet": "FG-NET", "morph": "MORPH", "utkface": "UTKFace"}
CNN_NAMES = {"resnet50": "ResNet50", "convnext": "ConvNeXt-Tiny"}
ML_NAMES = {"mlp": "MLP", "xgboost": "XGBoost"}
AGE_BINS = [0, 10, 20, 30, 40, 50, 60, 70, 120]

# Slučajevi za grafik predviđenih naspram stvarnih godina: (cnn, ml, trening, test)
SCATTER_CASES = [
    ("convnext", "mlp", "morph", "fgnet"),
    ("convnext", "xgboost", "morph", "fgnet"),
    ("resnet50", "xgboost", "utkface", "morph"),
    ("convnext", "xgboost", "utkface", "morph"),
]

plt.rcParams.update({"font.size": 10, "savefig.dpi": 200, "savefig.bbox": "tight"})


@lru_cache(maxsize=None)
def cached_features(cnn: str, dataset: str):
    """Isto kao load_features, ali svaki fajl učitava samo jednom."""
    return load_features(cnn, dataset)


@lru_cache(maxsize=None)
def cached_model(cnn: str, ml: str, train: str):
    return joblib.load(MODELS_DIR / f"{cnn}_{ml}_{train}.joblib")


def predict(cnn: str, ml: str, train: str, test: str):
    """Vraća (stvarne godine, predviđene godine) za dati par trening -> test."""
    X, y, _ = cached_features(cnn, test)
    return y, cached_model(cnn, ml, train).predict(X)


def save(fig, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / name)
    plt.close(fig)
    print(f"Sačuvano: results/figures/{name}")


def plot_age_distributions() -> None:
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5), layout="constrained", sharex=True)
    for ax, name in zip(axes, DATASETS):
        _, y, groups = cached_features("convnext", name)
        ax.hist(y, bins=range(0, 120, 5), color="tab:blue", edgecolor="white")
        ax.set_title(f"{NAMES[name]}\n{len(y)} slika, {len(np.unique(groups))} osoba")
        ax.set_xlabel("Godine")
    axes[0].set_ylabel("Broj slika")
    save(fig, "age_distributions.png")


def plot_heatmaps(df: pd.DataFrame, metric: str = "mae") -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 9), layout="constrained")
    vmin, vmax = df[metric].min(), df[metric].max()
    labels = [NAMES[d] for d in DATASETS]
    combos = [(cnn, ml) for cnn in CNNS for ml in ML_MODELS]

    for ax, (cnn, ml) in zip(axes.flat, combos):
        subset = df[(df["cnn"] == cnn) & (df["ml"] == ml)]
        matrix = subset.pivot(index="train", columns="test", values=metric).loc[DATASETS, DATASETS]
        image = ax.imshow(matrix.values, cmap="RdYlGn_r", vmin=vmin, vmax=vmax)

        for i in range(len(DATASETS)):
            for j in range(len(DATASETS)):
                ax.text(j, i, f"{matrix.values[i, j]:.2f}", ha="center", va="center",
                        fontweight="bold" if i == j else "normal")
            # Okvir oko dijagonale (unakrsna validacija unutar skupa)
            ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1,
                                       fill=False, edgecolor="black", linewidth=2))

        ax.set_xticks(range(len(DATASETS)), labels, rotation=30, ha="right")
        ax.set_yticks(range(len(DATASETS)), labels)
        ax.set_xlabel("Skup za testiranje")
        ax.set_ylabel("Skup za treniranje")
        ax.set_title(f"{CNN_NAMES[cnn]} + {ML_NAMES[ml]}")

    fig.colorbar(image, ax=axes, shrink=0.6, label="MAE (godine)")
    fig.suptitle("MAE unakrsnog testiranja (uokvirena dijagonala = validacija unutar skupa)")
    save(fig, "heatmap_mae.png")


def plot_pred_vs_true() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), layout="constrained")
    for ax, (cnn, ml, train, test) in zip(axes.flat, SCATTER_CASES):
        y, pred = predict(cnn, ml, train, test)
        _, y_train, _ = cached_features(cnn, train)

        ax.scatter(y, pred, s=4, alpha=0.15 if len(y) > 5000 else 0.4)
        low = min(0, pred.min()) - 5
        high = max(y.max(), pred.max()) + 5
        ax.plot([low, high], [low, high], "k--", linewidth=1, label="idealno predviđanje")
        ax.axhspan(y_train.min(), y_train.max(), color="tab:green", alpha=0.1,
                   label=f"raspon godina u {NAMES[train]}")
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.set_aspect("equal")

        mae = np.mean(np.abs(pred - y))
        bias = np.mean(pred - y)  # > 0: model u proseku stari, < 0: podmlađuje
        ax.set_title(f"{CNN_NAMES[cnn]} + {ML_NAMES[ml]}: {NAMES[train]} → {NAMES[test]}\n"
                     f"MAE {mae:.2f}, pristrasnost {bias:+.2f}")
        ax.set_xlabel("Stvarne godine")
        ax.set_ylabel("Predviđene godine")
        ax.legend(loc="upper left", fontsize=8)

        print(f"  {cnn} + {ml}, {train} -> {test}: predviđanja od {pred.min():.1f} "
              f"do {pred.max():.1f} (trening: {y_train.min():.0f}-{y_train.max():.0f}), "
              f"pristrasnost {bias:+.2f}")
    save(fig, "pred_vs_true.png")


def plot_mae_by_age(cnn: str = "convnext", ml: str = "xgboost") -> None:
    labels = [f"{a}–{b - 1}" for a, b in zip(AGE_BINS[:-1], AGE_BINS[1:])]
    labels[-1] = "70+"

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), layout="constrained", sharey=True)
    for ax, test in zip(axes, DATASETS):
        for train in DATASETS:
            if train == test:
                continue
            y, pred = predict(cnn, ml, train, test)
            age_group = pd.cut(y, AGE_BINS, right=False, labels=labels)
            mae = pd.Series(np.abs(pred - y)).groupby(age_group, observed=False).mean()
            ax.plot(labels, mae.values, marker="o", label=f"trening: {NAMES[train]}")
        ax.set_title(f"Test: {NAMES[test]}")
        ax.set_xlabel("Starosna grupa")
        ax.tick_params(axis="x", rotation=45)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("MAE (godine)")
    fig.suptitle(f"MAE po starosnim grupama ({CNN_NAMES[cnn]} + {ML_NAMES[ml]})")
    save(fig, f"mae_by_age_{cnn}_{ml}.png")


def main():
    df = pd.read_csv(CROSS_RESULTS_PATH)
    plot_age_distributions()
    plot_heatmaps(df, "mae")
    plot_heatmaps(df, "rmse")
    print("Predviđene naspram stvarnih godina:")
    plot_pred_vs_true()
    plot_mae_by_age()


if __name__ == "__main__":
    main()