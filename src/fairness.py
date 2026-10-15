"""
Analiza pristrasnosti (fairness): greska procene godina po polu i rasi.

Za svaki od 16 modela (CNN x ML x skup za obucavanje) i svaki test skup
koji ima demografske atribute racuna MAE, RMSE i ME (srednju gresku sa
znakom) posebno za svaku grupu. Koriste se vec sacuvani modeli i obelezja,
nista se ne obucava ponovo.

Dostupni atributi:
    MORPH   -> pol (iz naziva fajla, npr. 00013_00M19 -> M)
    UTKFace -> pol i rasa (iz parquet fajlova)
    FG-NET, APPA-REAL -> nemaju atribute, preskacu se kao test skupovi

Izlaz:
    results/fairness.csv
    results/figures/fairness_gender.png
    results/figures/fairness_race.png

Pokretanje:
    python src/fairness.py
"""

import glob
import re
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FEATURES_DIR = ROOT / "features"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"

CNNS = ["resnet50", "convnext"]
MLS = ["mlp", "xgboost"]
DATASETS = ["fgnet", "morph", "appa", "utkface"]
DATASET_NAMES = {"fgnet": "FG-NET", "morph": "MORPH", "appa": "APPA-REAL", "utkface": "UTKFace"}

GENDER_SR = {"M": "muski", "F": "zenski"}
RACE_SR = {"White": "bela", "Black": "crna", "Asian": "azijska",
           "Indian": "indijska", "Other": "ostale"}

MORPH_PATTERN = re.compile(r"^\d+_\d+([MF])\d+$", re.IGNORECASE)


# 1. Demografski atributi

def _stem(path):
    """Naziv fajla bez ekstenzije, malim slovima (raw ima .JPG, processed .jpg)."""
    return Path(str(path).replace("\\", "/")).stem.lower()


def _path_column(meta):
    for col in ("processed_path", "image_path", "path"):
        if col in meta.columns:
            return col
    raise KeyError(f"Meta fajl nema kolonu sa putanjom slike: {list(meta.columns)}")


def attributes_morph(meta):
    """Pol iz naziva fajla (slovo M ili F izmedju brojeva)."""
    stems = meta[_path_column(meta)].map(_stem)
    gender = stems.map(lambda s: (m := MORPH_PATTERN.match(s)) and m.group(1).upper())
    return pd.DataFrame({"gender": gender.map(GENDER_SR), "race": None})


def attributes_utkface(meta):
    """Pol i rasa iz originalnih parquet fajlova, spojeni po nazivu fajla."""
    files = glob.glob(str(ROOT / "data" / "raw" / "utkface" / "*.parquet"))
    if not files:
        raise FileNotFoundError("Nisu pronadjeni UTKFace parquet fajlovi u data/raw/utkface/")
    attrs = pd.concat(pd.read_parquet(f, columns=["file_name", "gender", "ethnicity"])
                      for f in files)
    attrs["key"] = attrs["file_name"].map(_stem)
    attrs = attrs.drop_duplicates("key").set_index("key")

    keys = meta[_path_column(meta)].map(_stem)
    gender = keys.map(attrs["gender"]).map({"Male": "muski", "Female": "zenski"})
    race = keys.map(attrs["ethnicity"]).map(RACE_SR)
    return pd.DataFrame({"gender": gender.values, "race": race.values})


ATTRIBUTE_LOADERS = {"morph": attributes_morph, "utkface": attributes_utkface}


# 2. Modeli i obelezja

def load_features(cnn, dataset):
    X = np.load(FEATURES_DIR / cnn / f"{dataset}.npy")
    meta = pd.read_csv(FEATURES_DIR / cnn / f"{dataset}_meta.csv")
    assert len(X) == len(meta), f"{cnn}/{dataset}: broj redova u .npy i _meta.csv se razlikuje"
    return X, meta


def find_model(cnn, ml, train):
    """Pronalazi sacuvani .joblib model po imenima CNN-a, ML algoritma i skupa."""
    candidates = []
    for p in ROOT.rglob("*.joblib"):
        rel = p.relative_to(ROOT).as_posix().lower()
        if not rel.startswith(".venv") and cnn in rel and ml in rel and train in rel:
            candidates.append(p)
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Za {cnn}+{ml}+{train} ocekivan je tacno jedan .joblib model, nadjeno: {candidates}")
    return candidates[0]


# 3. Metrike po grupama


AGE_BINS = [0, 20, 30, 40, 50, 60, 200]          # 0-19, 20-29, ..., 50-59, 60+
MIN_PER_BIN = 10                                  # manji uzorci se ne koriste za MAE_std


def group_metrics(y_true, y_pred, groups):
    """MAE, RMSE i ME po grupi, plus MAE standardizovan po godinama.

    MAE_std (direktna standardizacija): greska grupe se racuna posebno u
    svakom starosnom opsegu, a zatim se ti MAE-ovi usrednjavaju sa tezinama
    koje odgovaraju raspodeli godina CELOG test skupa. Tako sve grupe imaju
    "istu" starosnu strukturu i razlika u MAE_std ne moze poticati od toga
    sto je neka grupa u proseku starija ili mladja.
    """
    df = pd.DataFrame({"y": y_true, "p": y_pred, "g": groups}).dropna(subset=["g"])
    df["err"] = df["p"] - df["y"]
    df["bin"] = pd.cut(df["y"], AGE_BINS, right=False)
    ref_weights = df["bin"].value_counts(normalize=True)   # raspodela godina celog test skupa

    rows = []
    for g, part in df.groupby("g"):
        err = part["err"]
        per_bin = part.groupby("bin", observed=True)["err"].agg(
            mae=lambda e: e.abs().mean(), n="size")
        per_bin = per_bin[per_bin["n"] >= MIN_PER_BIN]
        w = ref_weights.reindex(per_bin.index)
        rows.append({
            "group": g,
            "n": len(part),
            "mean_age": part["y"].mean(),
            "mae": err.abs().mean(),
            "mae_std": (per_bin["mae"] * w).sum() / w.sum() if w.sum() > 0 else np.nan,
            "rmse": np.sqrt((err ** 2).mean()),
            "me": err.mean(),   # > 0 model "stari" grupu, < 0 "podmladjuje"
        })
    return rows


def evaluate_all():
    rows = []
    for cnn in CNNS:
        # obelezja i atributi test skupova ucitavaju se jednom po CNN-u
        tests = {}
        for test in ATTRIBUTE_LOADERS:
            X, meta = load_features(cnn, test)
            tests[test] = (X, meta["age"].to_numpy(), ATTRIBUTE_LOADERS[test](meta))
            missing = tests[test][2]["gender"].isna().sum()
            if missing:
                print(f"  upozorenje: {cnn}/{test} - {missing} slika bez oznake pola")

        for ml in MLS:
            for train in DATASETS:
                model = joblib.load(find_model(cnn, ml, train))
                for test, (X, y, attrs) in tests.items():
                    if test == train:
                        continue
                    pred = model.predict(X)
                    for attribute in ("gender", "race"):
                        if attrs[attribute].isna().all():
                            continue
                        for r in group_metrics(y, pred, attrs[attribute].to_numpy()):
                            rows.append({"cnn": cnn, "ml": ml, "train": train,
                                         "test": test, "attribute": attribute, **r})
                print(f"  gotovo: {cnn} + {ml}, obucen na {DATASET_NAMES[train]}")
    return pd.DataFrame(rows)


# 4. Ispis i grafici

def _gender_table(df, value):
    g = df[df["attribute"] == "gender"].pivot_table(
        index=["cnn", "ml", "train", "test"], columns="group", values=value)
    g["razlika"] = g["zenski"] - g["muski"]
    return g


def _race_table(df, value):
    r = df[df["attribute"] == "race"].pivot_table(
        index=["cnn", "ml", "train"], columns="group", values=value)
    r["razlika"] = r.max(axis=1) - r.min(axis=1)
    r["najlosija"] = r.drop(columns="razlika").idxmax(axis=1)
    return r


def print_summary(df):
    pd.set_option("display.width", 160)

    print("\n=== Prosecne godine i broj slika po grupi ===")
    info = df.groupby(["attribute", "test", "group"])[["n", "mean_age"]].first()
    print(info.round(1).to_string())

    for value, label in (("mae", "MAE"), ("mae_std", "MAE standardizovan po godinama")):
        g = _gender_table(df, value)
        print(f"\n=== Pol: {label}, razlika (zenski - muski) ===")
        print(g.round(2).to_string())
        print(f"Prosecna razlika: {g['razlika'].mean():.2f} godina; zene imaju vecu "
              f"gresku u {(g['razlika'] > 0).sum()} od {len(g)} evaluacija")

    for value, label in (("mae", "MAE"), ("mae_std", "MAE standardizovan po godinama")):
        r = _race_table(df, value)
        print(f"\n=== Rasa (test UTKFace): {label}, razlika (max - min) ===")
        print(r.round(2).to_string())
        print("Koliko puta je grupa najlosija:", r["najlosija"].value_counts().to_dict())

    print("\n=== Smer greske (ME) po grupi, prosek svih modela ===")
    me = df.groupby(["attribute", "test", "group"])["me"].mean().round(2)
    print(me.to_string())


def _grouped_bars(ax, data, groups, title, ylabel="MAE (godine)"):
    trains = list(data.index)
    width = 0.8 / len(groups)
    x = np.arange(len(trains))
    for i, grp in enumerate(groups):
        ax.bar(x + i * width - 0.4 + width / 2, data[grp], width, label=grp)
    ax.set_xticks(x, [DATASET_NAMES[t] for t in trains])
    ax.set_xlabel("Skup za obucavanje")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)


def plot_gender(df, cnn="convnext", ml="xgboost"):
    sub = df[(df["attribute"] == "gender") & (df["cnn"] == cnn) & (df["ml"] == ml)]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=True)
    for row, (value, label) in enumerate((("mae", "MAE"), ("mae_std", "MAE std."))):
        for ax, test in zip(axes[row], ["morph", "utkface"]):
            data = sub[sub["test"] == test].pivot(index="train", columns="group", values=value)
            _grouped_bars(ax, data, ["muski", "zenski"],
                          f"{label}, test: {DATASET_NAMES[test]}", f"{label} (godine)")
    axes[0, 0].legend(title="Pol")
    fig.suptitle(f"Greska po polu ({cnn} + {ml}); std. = standardizovano po godinama")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fairness_gender.png", dpi=150)
    plt.close(fig)


def plot_race(df, cnn="convnext", ml="xgboost"):
    sub = df[(df["attribute"] == "race") & (df["cnn"] == cnn) & (df["ml"] == ml)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
    for ax, (value, label) in zip(axes, (("mae", "MAE"), ("mae_std", "MAE std."))):
        data = sub.pivot(index="train", columns="group", values=value)
        groups = [g for g in RACE_SR.values() if g in data.columns]
        _grouped_bars(ax, data, groups, f"{label}, test: UTKFace", f"{label} (godine)")
    axes[0].legend(title="Rasa", ncol=3, fontsize=8)
    fig.suptitle(f"Greska po rasi ({cnn} + {ml}); std. = standardizovano po godinama")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fairness_race.png", dpi=150)
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print("Racunam greske po grupama...")
    df = evaluate_all()
    df.round(3).to_csv(RESULTS_DIR / "fairness.csv", index=False)
    print_summary(df)
    plot_gender(df)
    plot_race(df)
    print(f"\nSacuvano: {RESULTS_DIR / 'fairness.csv'}")
    print(f"Grafici:  {FIG_DIR / 'fairness_gender.png'}, {FIG_DIR / 'fairness_race.png'}")


if __name__ == "__main__":
    main()