# meat-quality-analysis
Computer vision method that analyses images of packaged pork rashers.

## Repository layout

```
scripts/     one-off utilities (e.g. dataset download)
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

The dataset is hosted on Roboflow (project `pork-rasher-error-packaging`,
version 4, YOLO format) and is not stored in this repository.

1. Copy `.env.example` to `.env` and set `ROBOFLOW_API_KEY` to your Roboflow API key.
2. Run:

   ```bash
   python scripts/download_data.py
   ```

The images and labels are written to `data/pork-rasher-error-packaging-4/`.
