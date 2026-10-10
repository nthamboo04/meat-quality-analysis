"""Step-by-step preparation of the pork rasher packaging dataset."""

# Step 1: Import libraries
# Only what is needed for the data steps so far; more are added as later
# steps need them.
from pathlib import Path

import yaml

# Step 2: Load the data, class_names and num_classes
# The Roboflow export puts the class names in data.yaml, and the images in
# train/, valid/ and test/ folders, each with images/ and labels/ subfolders.
# The path is built from this file's location so the script works no matter
# which folder it is run from.
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "pork-rasher-error-packaging-4"

with open(DATA_DIR / "data.yaml") as f:
    data_config = yaml.safe_load(f)

class_names = data_config["names"]
num_classes = data_config["nc"]

print("class_names:", class_names)
print("num_classes:", num_classes)

for split in ["train", "valid", "test"]:
    n_images = len(list((DATA_DIR / split / "images").glob("*.jpg")))
    print(f"{split}: {n_images} images")
