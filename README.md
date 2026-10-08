# Procena godina na osnovu slike lica: evaluacija unakrsnim testiranjem

Završni rad: evaluacija modela za procenu godina čoveka na osnovu slike lica primenom **unakrsnog testiranja** (*cross-dataset evaluation*).


## Pristup

Koristi se **transferno učenje sa ekstrakcijom karakteristika**: pretrenirane konvolucione
mreže (zamrznute težine) pretvaraju sliku lica u vektor karakteristika, a godine predviđaju
klasični algoritmi mašinskog učenja.

```
slika lica ──► detekcija i poravnanje ──► CNN (bez poslednjeg sloja) ──► vektor karakteristika
                   (YuNet, 224×224)        ResNet50 / ConvNeXt-Tiny          │
                                                                              ▼
                                         procenjene godine ◄── StandardScaler → PCA → MLP / XGBoost
```

Svaki model se trenira na jednom skupu podataka, a evaluira na **svim ostalim skupovima zasebno**.

| Komponenta | Varijante |
|---|---|
| CNN za ekstrakciju | `resnet50.tv2_in1k`, `convnext_tiny.fb_in1k` (timm, ImageNet-1k) |
| ML algoritam | MLP (`scikit-learn`), XGBoost |
| Skupovi podataka | APPA-REAL, FG-NET, MORPH II, UTKFace |
| Ukupno modela | 2 CNN × 2 ML × 4 skupa = **16** |

## Skupovi podataka

| Skup | Slika | Osoba | Godine | Izvor oznaka |
|---|---|---|---|---|
| APPA-REAL | 7.591 | — | 1–100 | `labels.csv` |
| FG-NET | 1.002 | 82 | 0–69 | ime fajla (`001A02.JPG`) |
| MORPH II | 50.015 | 19.033 | 16–77 | ime fajla (`00013_00M19.JPG`) |
| UTKFace | 23.705 | — | 1–116 | `.parquet` (Hugging Face) |

Skupovi podataka **nisu deo repozitorijuma** zbog veličine i licenci. Treba ih preuzeti
sa zvaničnih izvora i smestiti u sledeću strukturu:

```
data/raw/
├── appa/
│   ├── labels.csv            # kolone: file_name, real_age
│   └── final_files/*.jpg
├── fgnet/
│   └── images/*.JPG
├── morph/
│   └── Dataset/Images/{Train,Validation,Test}/*.JPG
└── utkface/
    └── train-0000X-of-00003.parquet
```

## Struktura repozitorijuma

```
├── src/
│   ├── datasets.py          # učitavanje skupova u zajednički format
│   ├── preprocess.py        # detekcija, poravnanje i isecanje lica
│   ├── extract_features.py  # ekstrakcija karakteristika (ResNet50, ConvNeXt)
│   ├── train.py             # optimizacija hiperparametara i treniranje
│   ├── evaluate.py          # unakrsno testiranje
│   └── plots.py             # grafici
├── models/
│   └── face_detection_yunet_2023mar.onnx   # detektor lica (OpenCV Zoo)
├── results/
│   ├── hpo/                 # najbolji hiperparametri i CV MAE (.json)
│   ├── cross_dataset.csv    # svi rezultati unakrsnog testiranja
│   └── figures/             # grafici
└── requirements.txt
```

Folderi `data/`, `features/` i `results/models/` se prave tokom izvršavanja i nisu na Git-u.

## Instalacija

Python 3.13, Windows (Git Bash):

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
```

Na Linux-u / macOS-u: `source .venv/bin/activate`.

## Pokretanje

Sve komande se pokreću iz korena projekta. Navedena vremena su izmerena na procesoru
(12 niti, bez GPU-a).

| Korak | Komanda | Izlaz | Vreme |
|---|---|---|---|
| 1. Učitavanje skupova | `python -m src.datasets` | `data/metadata.csv` | ~1 min |
| 2. Predobrada lica | `python -m src.preprocess` | `data/processed/`, `data/metadata_processed.csv` | ~1,2 h |
| 3. Ekstrakcija karakteristika | `python -m src.extract_features --model resnet50`<br>`python -m src.extract_features --model convnext` | `features/<model>/<skup>.npy` | ~2,3 h + ~2 h |
| 4. Optimizacija i treniranje | `python -m src.train` | `results/models/`, `results/hpo/` | ~1,5 h |
| 5. Unakrsno testiranje | `python -m src.evaluate` | `results/cross_dataset.csv` | ~2 min |
| 6. Grafici | `python -m src.plots` | `results/figures/` | ~2 min |

Koraci 2–4 podržavaju nastavak posle prekida: već završeni delovi se preskaču.
Korisne opcije:

```bash
python -m src.preprocess --limit 20                                  # proba na 20 slika po skupu
python -m src.extract_features --model resnet50 --datasets fgnet --limit 256   # merenje brzine
python -m src.train --cnn convnext --ml xgboost --train morph --trials 25      # jedna kombinacija
```

## Metodologija

**Predobrada.** Detektor lica YuNet (OpenCV, prag 0,6) pronalazi lice i 5 karakterističnih
tačaka. Lice se afinom transformacijom poravnava na ArcFace šablon (faktor kadra 0,9) i
seče na 224×224 piksela; ivice se popunjavaju refleksijom. Slike bez detektovanog lica
(73 od 82.313) se izostavljaju.

**Ekstrakcija.** Mreže su pretrenirane na ImageNet-1k, sa uklonjenim klasifikacionim
slojem. Izlaz je vektor od 2048 (ResNet50) ili 768 (ConvNeXt-Tiny) karakteristika.

**Optimizacija hiperparametara.** Posebno za svaku od 16 kombinacija:
- Optuna (TPE), 25 proba;
- `GroupKFold` sa 3 folda **po osobi**, tako da ista osoba nije i u treningu i u validaciji;
- podskup do 5.000 slika, biranih po osobama; konačan model se trenira na celom skupu;
- broj PCA komponenti (128 / 256 / 512) je takođe hiperparametar.

**Metrike.**

| Metrika | Opis |
|---|---|
| MAE | srednja apsolutna greška u godinama |
| CS(5) | procenat slika sa greškom ≤ 5 godina |
| Osnovna linija | MAE modela koji uvek predviđa prosečne godine iz skupa za treniranje |
| Poboljšanje | `(1 − MAE / MAE_osnovne_linije) × 100%` |

Radi ponovljivosti rezultata, početna vrednost generatora pseudoslučajnih brojeva (*seed*)
postavljena je na 42 u svim koracima.

## Rezultati

Prosek preko 12 unakrsnih parova (trening ≠ test):

| CNN + ML | MAE unutar skupa | MAE između skupova | CS(5) | Poboljšanje | Gore od osnovne linije |
|---|---|---|---|---|---|
| ResNet50 + MLP | 8,21 | 13,61 | 25,4% | 11,1% | 3 / 12 |
| ResNet50 + XGBoost | 8,17 | 12,50 | 25,0% | 15,9% | 2 / 12 |
| ConvNeXt-Tiny + MLP | 7,56 | 12,50 | 25,7% | 17,1% | 4 / 12 |
| **ConvNeXt-Tiny + XGBoost** | **7,61** | **11,58** | **27,9%** | **24,2%** | **0 / 12** |

![MAE unakrsnog testiranja](results/figures/heatmap_mae.png)

![MAE po starosnim grupama](results/figures/mae_by_age_convnext_xgboost.png)

### Glavni zaključci

1. Unakrsno testiranje povećava MAE za 4–5,5 godina u odnosu na validaciju unutar skupa.
2. ConvNeXt-Tiny daje bolje karakteristike od ResNet50 i manje je osetljiv na uslove snimanja (osvetljenje, kvalitet slike, pozadina, izraz lica).
3. MLP i XGBoost su izjednačeni unutar skupa, ali XGBoost bolje generalizuje.
4. Raznovrsnost i pokrivenost godina u skupu za treniranje su bitniji od broja slika:
   APPA-REAL i UTKFace generalizuju najbolje, a MORPH, iako najveći, najlošije.
5. Greške su najveće za uzraste koji nisu zastupljeni u skupu za treniranje (deca, stariji).
