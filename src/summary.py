"""Pregled glavnih rezultata, izračunat iz results/cross_dataset.csv."""

import pandas as pd

from src.evaluate import CROSS_RESULTS_PATH

pd.set_option("display.width", 120)


def main():
    df = pd.read_csv(CROSS_RESULTS_PATH)
    cv = df[df["type"] == "cv"]          # validacija unutar skupa (dijagonala)
    cross = df[df["type"] == "cross"]    # unakrsno testiranje

    # 1. Rezultati po kombinaciji CNN + ML
    by_combo = cross.groupby(["cnn", "ml"])
    summary = pd.DataFrame({
        "MAE unutar": cv.groupby(["cnn", "ml"])["mae"].mean(),
        "MAE izmedju": by_combo["mae"].mean(),
        "RMSE unutar": cv.groupby(["cnn", "ml"])["rmse"].mean(),
        "RMSE izmedju": by_combo["rmse"].mean(),
        "CS(5) %": by_combo["cs5"].mean(),
        "poboljsanje %": by_combo["improvement"].mean(),
        "gore od baseline": by_combo["improvement"].apply(lambda s: int((s < 0).sum())),
    })
    summary["pad (god.)"] = summary["MAE izmedju"] - summary["MAE unutar"]
    summary["pad %"] = summary["pad (god.)"] / summary["MAE unutar"] * 100

    print("=== 1. Rezultati po kombinaciji CNN + ML ===")
    print(summary.round(2).to_string())

    best = summary["MAE izmedju"].idxmin()
    print(f"\nNajbolja kombinacija: {best[0]} + {best[1]} "
          f"(MAE izmedju skupova {summary.loc[best, 'MAE izmedju']:.2f}, "
          f"gore od baseline: {summary.loc[best, 'gore od baseline']} od 12)")

    # 2. Pad performansi na drugim skupovima
    print("\n=== 2. Pad performansi na drugim skupovima ===")
    print(f"MAE raste za {summary['pad (god.)'].min():.1f} do {summary['pad (god.)'].max():.1f} "
          f"godina ({summary['pad %'].min():.0f}% do {summary['pad %'].max():.0f}%)")

    # 3. CNN i ML algoritam
    print("\n=== 3. Poredjenje CNN i ML algoritma (prosecan MAE izmedju skupova) ===")
    print(cross.groupby("cnn")["mae"].mean().round(2).to_string())
    print(cross.groupby("ml")["mae"].mean().round(2).to_string())

    # 4. Skup za treniranje: poredi se unutar istog test skupa
    print("\n=== 4. Skup za treniranje (prosecan MAE 4 modela; rang unutar kolone) ===")
    pivot = cross.groupby(["train", "test"])["mae"].mean().unstack()
    pivot["prosecan rang"] = pivot.rank(axis=0).mean(axis=1)
    print(pivot.sort_values("prosecan rang").round(2).to_string())


if __name__ == "__main__":
    main()