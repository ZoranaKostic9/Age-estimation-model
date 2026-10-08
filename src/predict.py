"""Procena godina za sve slike u folderu (ili za jednu sliku)."""

import os
os.environ["HF_HUB_OFFLINE"] = "1"   # koristi već preuzete težine, bez provere na internetu

import cv2
cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)   # prikazuj samo greške

import argparse
import re
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.datasets import IMAGE_EXTENSIONS, PROJECT_ROOT
from src.extract_features import build_model
from src.preprocess import MODEL_PATH, process_image, write_image
from src.train import DATASETS, MODELS_DIR

ALIGNED_DIR = PROJECT_ROOT / "data" / "moje_slike_poravnate"
OUTPUT_CSV = PROJECT_ROOT / "data" / "moje_procene.csv"


# Opciono: "zorana_23.jpg" -> stvarne godine 23
AGE_IN_NAME = re.compile(r"_(\d{1,3})$")


def list_images(path: Path):
    """Ako je putanja fajl, vraća samo njega; ako je folder, sve slike u njemu."""
    if path.is_file():
        return [path]
    return sorted(p for p in path.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def main():
    parser = argparse.ArgumentParser(description="Procena godina za moje slike")
    parser.add_argument("paths", nargs="+",
                        help="jedna ili više slika, ili folder, npr. data/moje_slike")
    parser.add_argument("--cnn", default="convnext", choices=["resnet50", "convnext"])
    parser.add_argument("--ml", default="xgboost", choices=["mlp", "xgboost"])
    args = parser.parse_args()

    images = [image for path in args.paths for image in list_images(Path(path))]
    if not images:
        print("Nema slika na zadatim putanjama.")
        return

    # Detektor, mreža i 4 modela se učitavaju jednom i koriste za sve slike
    detector = cv2.FaceDetectorYN.create(str(MODEL_PATH), "", (320, 320),
                                         score_threshold=0.6)
    model, transform = build_model(args.cnn)
    regressors = {
        name: joblib.load(MODELS_DIR / f"{args.cnn}_{args.ml}_{name}.joblib")
        for name in DATASETS
    }
    ALIGNED_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for image_path in images:
        # 1. Predobrada, isto kao za skupove podataka
        face, detected = process_image(detector, image_path)
        if face is None:
            print(f"[{image_path.name}] ne može da se pročita, preskačem")
            continue

        face_path = ALIGNED_DIR / f"{image_path.stem}.jpg"
        write_image(face_path, face)

        # 2. Ekstrakcija karakteristika iz sačuvanog lica
        image = Image.open(face_path).convert("RGB")
        with torch.inference_mode():
            features = model(transform(image).unsqueeze(0)).numpy()

        
        # 3. Predviđanje: svaki od 4 modela posebno
        row = {"slika": image_path.name, "lice_nadjeno": detected}
        for name, regressor in regressors.items():
            row[name] = round(float(regressor.predict(features)[0]), 1)

        # Ako su stvarne godine u imenu fajla: greška SVAKOG modela posebno
        match = AGE_IN_NAME.search(image_path.stem)
        if match:
            row["stvarne"] = int(match.group(1))
            for name in DATASETS:
                row[f"greska_{name}"] = round(row[name] - row["stvarne"], 1)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nProcena godina ({args.cnn} + {args.ml}):\n")
    print(df.to_string(index=False))

    if "stvarne" in df.columns:
        known = df.dropna(subset=["stvarne"])
        print(f"\nGreška po modelu na {len(known)} slika sa poznatim godinama:")
        print(f"  {'model':<10} {'MAE':>6} {'RMSE':>6}")
        for name in DATASETS:
            errors = known[name] - known["stvarne"]
            mae = errors.abs().mean()
            rmse = np.sqrt((errors ** 2).mean())
            print(f"  {name:<10} {mae:6.1f} {rmse:6.1f}")


if __name__ == "__main__":
    main()