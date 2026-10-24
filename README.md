# meat-quality-analysis
Computer vision method that analyses images of packaged pork rashers.

## Repository layout

```
meat_quality/   project source code
    data.py       load and explore the dataset
    metrics.py    match predicted to labelled boxes and show tray pass/fail counts
    train.py      train a YOLO model (on a GPU: Google Colab or Kaggle)
    evaluate.py   evaluate a trained model on a dataset split
    benchmark.py  time predictions per tray; model size and FLOPs
data/           downloaded dataset (git-ignored)
runs/           trained models and training outputs (git-ignored)
results.txt     final validation output of each training run
```

## Setup

Developed with Python 3.12. Create a virtual environment and install the following dependencies:

```bash
python -m venv C:\Users\<you>\venvs\meat
C:\Users\<you>\venvs\meat\Scripts\activate    # macOS/Linux: source <folder>/bin/activate
pip install -r requirements.txt
pip install -e .        # makes the meat_quality package importable from any folder
```

On Windows, keep the environment in a short path outside OneDrive. PyTorch contains files that exceed Windows' 260-character path limit inside long folders, and OneDrive would sync thousands of package files.

Training was run on Google Colab and Kaggle (Tesla T4 GPU, Python 3.13, Ultralytics 8.4.183, torch 2.11.0).

## Getting the data

The dataset is hosted on Roboflow and is not stored in this repository. To reproduce the results, download exactly this
version:

- Workspace: `hello-2aqe0`
- Project: `pork-rasher-error-packaging`
- Version: 4
- Format: YOLOv12

1. Open the dataset on Roboflow Universe:
   <https://universe.roboflow.com/hello-2aqe0/pork-rasher-error-packaging/dataset/4>
2. Select version 4, click Download Dataset, choose the YOLOv12 format and Download zip to computer.
3. Unzip the contents into `data/pork-rasher-error-packaging-4/`, so that the folder contains `data.yaml` and `train/`, `valid/` and `test/` subfolders.

The dataset contains 3549 images at 720x540 (3349 train / 120 valid / 80 test) with five classes: `loose-meat`, `packaging-error`, `twisted-meat`, `unsealed` and `wrinkle`.

## Usage

Run from the repository root with the environment active.

```bash
python meat_quality/data.py          # class counts and label-count charts
python meat_quality/train.py --scale 0.2       # train the selected model
python meat_quality/evaluate.py      # evaluate the selected model on the validation set
python meat_quality/evaluate.py --split test   # final check on the test set
python meat_quality/benchmark.py     # time per tray, model size and FLOPs
```

The selected model is YOLO11n at 736 px trained with scale augmentation 0.2. Its weights are expected at `runs/yolo11n_736_scale0.2/weights/best.pt` (not stored in Git). Each script lists its options with `--help`.

## Data licence

"Pork Rasher Error(Packaging)" v4, provided by a Roboflow user via
[Roboflow Universe](https://universe.roboflow.com/hello-2aqe0/pork-rasher-error-packaging), is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The code in this repository is licensed separately (see `LICENSE`).