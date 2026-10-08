"""Unakrsno testiranje: svaki model se evaluira na skupovima na kojima nije treniran."""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold, cross_validate

from src.train import (CNNS, DATASETS, HPO_DIR, ML_MODELS, MODELS_DIR, RESULTS_DIR,
                       build_model, load_features, subsample_by_person)

CROSS_RESULTS_PATH = RESULTS_DIR / "cross_dataset.csv"
HPO_MAX_SAMPLES = 5000  # isto kao podrazumevani --max-hpo-samples u train.py


def cumulative_score(y_true, y_pred, threshold: int = 5) -> float:
    """Procenat slika kod kojih je greška <= threshold godina."""
    return float(np.mean(np.abs(y_true - y_pred) <= threshold) * 100)


def rmse_score(y_true, y_pred) -> float:
    """Koren srednje kvadratne greške, u godinama."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def cv_rmse(ml, train_name, features, info, info_path) -> float:
    """RMSE unutar skupa: ista unakrsna validacija kao u train.py, sa najboljim parametrima.
    Računa se jednom i upisuje u .json, pa se sledeći put samo čita."""
    if "cv_rmse" in info:
        return info["cv_rmse"]

    X, y, groups = features[train_name]
    X, y, groups = subsample_by_person(X, y, groups, HPO_MAX_SAMPLES)
    scores = cross_validate(
        build_model(ml, info["best_params"]), X, y, groups=groups,
        cv=GroupKFold(n_splits=info["n_folds"]),
        scoring=("neg_mean_absolute_error", "neg_root_mean_squared_error"),
    )
    mae = -scores["test_neg_mean_absolute_error"].mean()
    if abs(mae - info["cv_mae"]) > 0.01:
        print(f"  upozorenje: ponovljen CV MAE {mae:.3f} != sačuvan {info['cv_mae']}")

    info["cv_rmse"] = round(float(-scores["test_neg_root_mean_squared_error"].mean()), 3)
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info["cv_rmse"]


def main():
    rows = []
    for cnn in CNNS:
        features = {name: load_features(cnn, name) for name in DATASETS}

        for ml in ML_MODELS:
            for train_name in DATASETS:
                tag = f"{cnn}_{ml}_{train_name}"
                model = joblib.load(MODELS_DIR / f"{tag}.joblib")
                info_path = HPO_DIR / f"{tag}.json"
                info = json.loads(info_path.read_text(encoding="utf-8"))
                train_mean_age = features[train_name][1].mean()

                for test_name in DATASETS:
                    row = {"cnn": cnn, "ml": ml, "train": train_name, "test": test_name}

                    if test_name == train_name:
                        row.update({
                            "type": "cv",
                            "mae": info["cv_mae"],
                            "rmse": cv_rmse(ml, train_name, features, info, info_path),
                            "cs5": np.nan, "baseline_mae": np.nan, "improvement": np.nan,
                        })
                    else:
                        X, y, _ = features[test_name]
                        pred = model.predict(X)
                        baseline = np.full_like(y, train_mean_age)
                        mae = mean_absolute_error(y, pred)
                        baseline_mae = mean_absolute_error(y, baseline)
                        row.update({
                            "type": "cross",
                            "mae": round(mae, 3),
                            "rmse": round(rmse_score(y, pred), 3),
                            "cs5": round(cumulative_score(y, pred), 2),
                            "baseline_mae": round(baseline_mae, 3),
                            "improvement": round((1 - mae / baseline_mae) * 100, 2),
                        })
                    rows.append(row)
                print(f"[{tag}] evaluiran")

    df = pd.DataFrame(rows)
    df.to_csv(CROSS_RESULTS_PATH, index=False)

    # Matrice MAE i RMSE: red = skup za treniranje, kolona = skup za testiranje
    for metric in ["mae", "rmse"]:
        for cnn in CNNS:
            for ml in ML_MODELS:
                subset = df[(df["cnn"] == cnn) & (df["ml"] == ml)]
                matrix = subset.pivot(index="train", columns="test", values=metric)
                print(f"\n=== {cnn} + {ml}: {metric.upper()} (dijagonala = CV unutar skupa) ===")
                print(matrix.round(2).to_string())

    cross = df[df["type"] == "cross"]
    summary = cross.groupby(["cnn", "ml"]).agg(
        mae=("mae", "mean"),
        rmse=("rmse", "mean"),
        cs5=("cs5", "mean"),
        improvement=("improvement", "mean"),
        gore_od_baseline=("improvement", lambda s: int((s < 0).sum())),
    ).round(2)
    print("\n=== Prosek unakrsnog testiranja ===")
    print(summary.to_string())
    print(f"\nSačuvano: {CROSS_RESULTS_PATH}")


if __name__ == "__main__":
    main()