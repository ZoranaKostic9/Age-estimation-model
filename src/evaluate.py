"""Unakrsno testiranje: svaki model se evaluira na skupovima na kojima nije treniran."""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from src.train import (CNNS, DATASETS, HPO_DIR, ML_MODELS, MODELS_DIR,
                       RESULTS_DIR, load_features)

CROSS_RESULTS_PATH = RESULTS_DIR / "cross_dataset.csv"


def cumulative_score(y_true, y_pred, threshold: int = 5) -> float:
    """Procenat slika kod kojih je greška <= threshold godina."""
    return float(np.mean(np.abs(y_true - y_pred) <= threshold) * 100)


def main():
    rows = []
    for cnn in CNNS:
        # Karakteristike svih skupova za ovaj CNN učitavamo jednom
        features = {name: load_features(cnn, name) for name in DATASETS}

        for ml in ML_MODELS:
            for train_name in DATASETS:
                tag = f"{cnn}_{ml}_{train_name}"
                model = joblib.load(MODELS_DIR / f"{tag}.joblib")
                info = json.loads((HPO_DIR / f"{tag}.json").read_text(encoding="utf-8"))
                train_mean_age = features[train_name][1].mean()

                for test_name in DATASETS:
                    row = {"cnn": cnn, "ml": ml, "train": train_name, "test": test_name}

                    if test_name == train_name:
                        # Isti skup: uzimamo rezultat unakrsne validacije iz treniranja
                        row.update({"type": "cv", "mae": info["cv_mae"],
                                    "cs5": np.nan, "baseline_mae": np.nan, "improvement": np.nan})
                    else:
                        X, y, _ = features[test_name]
                        pred = model.predict(X)
                        baseline = np.full_like(y, train_mean_age)
                        mae = mean_absolute_error(y, pred)
                        baseline_mae = mean_absolute_error(y, baseline)
                        row.update({
                            "type": "cross",
                            "mae": round(mae, 3),
                            "cs5": round(cumulative_score(y, pred), 2),
                            "baseline_mae": round(baseline_mae, 3),
                            "improvement": round((1 - mae / baseline_mae) * 100, 2),
                        })
                    rows.append(row)
                print(f"[{tag}] evaluiran")

    df = pd.DataFrame(rows)
    df.to_csv(CROSS_RESULTS_PATH, index=False)

    # Matrice MAE: red = skup za treniranje, kolona = skup za testiranje
    for cnn in CNNS:
        for ml in ML_MODELS:
            subset = df[(df["cnn"] == cnn) & (df["ml"] == ml)]
            matrix = subset.pivot(index="train", columns="test", values="mae")
            print(f"\n=== {cnn} + {ml}: MAE (dijagonala = CV unutar skupa) ===")
            print(matrix.round(2).to_string())

    # Prosek unakrsnih rezultata po kombinaciji CNN + ML
       # Matrice relativnog poboljšanja (samo unakrsni parovi)
    cross = df[df["type"] == "cross"]
    for cnn in CNNS:
        for ml in ML_MODELS:
            subset = cross[(cross["cnn"] == cnn) & (cross["ml"] == ml)]
            matrix = subset.pivot(index="train", columns="test", values="improvement")
            print(f"\n=== {cnn} + {ml}: poboljšanje u odnosu na osnovnu liniju (%) ===")
            print(matrix.round(1).to_string())

    # Prosek po kombinaciji CNN + ML
    summary = cross.groupby(["cnn", "ml"]).agg(
        mae=("mae", "mean"),
        cs5=("cs5", "mean"),
        improvement=("improvement", "mean"),
        gore_od_baseline=("improvement", lambda s: int((s < 0).sum())),
    ).round(2)
    print("\n=== Prosek unakrsnog testiranja ===")
    print(summary.to_string())
    print(f"\nSačuvano: {CROSS_RESULTS_PATH}")

if __name__ == "__main__":
    main()