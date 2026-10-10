# meat-quality-analysis
Computer vision method that analyses images of packaged pork rashers.

## Repository layout

```
src/         project source code (meat_quality package)
notebooks/   exploratory analysis and experiments
tests/       pytest test functions
docs/        generated HTML documentation
data/        downloaded dataset (git-ignored)
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

1. Sign in to [Roboflow](https://app.roboflow.com) (a free account is enough)
   and open the project above.
2. Select **version 4**, click **Download Dataset**, choose the **YOLOv12**
   format and **Download zip to computer**.
3. Unzip the contents into `data/pork-rasher-error-packaging-4/`, so that
   the folder contains `data.yaml` and `train/`, `valid/` and `test/`
   subfolders.
