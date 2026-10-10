# meat-quality-analysis
Computer vision method that analyses images of packaged pork rashers.

## Repository layout

```
meat_quality/   project source code (Python package)
notebooks/      step-by-step analysis notebooks
data/           downloaded dataset (git-ignored)
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; use source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
```

## Getting the data

The dataset is hosted on Roboflow and is not stored in this repository
(`data/` is git-ignored). To reproduce the results, download exactly this
version:

| Field     | Value                         |
|-----------|-------------------------------|
| Workspace | `hello-2aqe0`                 |
| Project   | `pork-rasher-error-packaging` |
| Version   | 4                             |
| Format    | YOLOv12                       |

1. Open the dataset on Roboflow Universe:
   <https://universe.roboflow.com/hello-2aqe0/pork-rasher-error-packaging/dataset/4>
   (signing in with a free account may be required to download).
2. Select **version 4**, click **Download Dataset**, choose the **YOLOv12**
   format and **Download zip to computer**.
3. Unzip the contents into `data/pork-rasher-error-packaging-4/`, so that
   the folder contains `data.yaml` and `train/`, `valid/` and `test/`
   subfolders.

The dataset contains 3549 images at 720x540 (3349 train / 120 valid / 80
test) with five classes: `loose-meat`, `packaging-error`, `twisted-meat`,
`unsealed` and `wrinkle`.

## Data licence

"Pork Rasher Error(Packaging)" v4, provided by a Roboflow user via
[Roboflow Universe](https://universe.roboflow.com/hello-2aqe0/pork-rasher-error-packaging),
is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The code in this repository is licensed separately (see `LICENSE`).
