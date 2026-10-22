# meat-quality-analysis
Computer vision method that analyses images of packaged pork rashers.

## Repository layout

```
meat_quality/   project source code (Python package)
data/           downloaded dataset (git-ignored)
```

## Setup

Developed with Python 3.12. Create a virtual environment and install the
dependencies:

```bash
python -m venv C:\Users\<you>\venvs\meat      # any folder; see note below
C:\Users\<you>\venvs\meat\Scripts\activate    # macOS/Linux: source <folder>/bin/activate
pip install -r requirements.txt
```

On Windows, keep the environment in a short path outside OneDrive. PyTorch
contains deeply nested files that exceed Windows' 260-character path limit
inside long folders (such as the Microsoft Store Python's package folder),
and OneDrive would otherwise sync thousands of package files.

Training was run on Google Colab (Tesla T4 GPU, Python 3.13,
Ultralytics 8.4.183, torch 2.11.0).

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
