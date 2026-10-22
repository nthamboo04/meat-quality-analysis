"""Step-by-step preparation of the pork rasher packaging dataset."""

# Step 1: Import libraries
# Only what is needed for the data steps so far; more are added as later
# steps need them.
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
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

# Step 3: Display one image of each class
# Each image has a label file with the same name in labels/. Every line in it
# is one labelled box: "class_id x_centre y_centre width height", where the
# four box values are fractions of the image width/height (0 to 1).
# For each class, find the first training image containing that class and
# draw its boxes of that class in red.
train_images = sorted((DATA_DIR / "train" / "images").glob("*.jpg"))

fig, axes = plt.subplots(1, num_classes, figsize=(4 * num_classes, 4))

for class_id, class_name in enumerate(class_names):
    for image_path in train_images:
        label_path = DATA_DIR / "train" / "labels" / (image_path.stem + ".txt")
        boxes = [line.split() for line in label_path.read_text().splitlines() if line.strip()]
        class_boxes = [box for box in boxes if int(box[0]) == class_id]
        if class_boxes:
            break

    image = plt.imread(image_path)
    img_height, img_width = image.shape[:2]

    ax = axes[class_id]
    ax.imshow(image)
    for _, x_centre, y_centre, width, height in class_boxes:
        # Convert from fractions of the image to pixels, and from the box
        # centre to its top-left corner (which matplotlib needs).
        w = float(width) * img_width
        h = float(height) * img_height
        x = float(x_centre) * img_width - w / 2
        y = float(y_centre) * img_height - h / 2
        ax.add_patch(patches.Rectangle((x, y), w, h, edgecolor="red", facecolor="none", linewidth=2))
    ax.set_title(class_name)
    ax.axis("off")

plt.tight_layout()
plt.show()

# Step 4: Count the labels in each class
# For every split, count how many boxes there are of each class, and how many
# images have no boxes at all (packs with no defect). These counts show how
# balanced the classes are, which affects what the model can be trained and
# evaluated on.
splits = ["train", "valid", "test"]
box_counts = {}       # box_counts[split][class_id] = number of boxes
no_defect_counts = {} # no_defect_counts[split] = images with no boxes

for split in splits:
    box_counts[split] = [0] * num_classes
    no_defect_counts[split] = 0
    for label_path in (DATA_DIR / split / "labels").glob("*.txt"):
        boxes = [line.split() for line in label_path.read_text().splitlines() if line.strip()]
        if not boxes:
            no_defect_counts[split] += 1
        for box in boxes:
            box_counts[split][int(box[0])] += 1

# Print the counts as a table: one row per class, one column per split.
print()
print(f"{'class':16s}" + "".join(f"{split:>8s}" for split in splits))
for class_id, class_name in enumerate(class_names):
    print(f"{class_name:16s}" + "".join(f"{box_counts[split][class_id]:8d}" for split in splits))
print(f"{'(no defect)':16s}" + "".join(f"{no_defect_counts[split]:8d}" for split in splits))

# Plot one bar chart per split. Each has its own y-axis because the splits are
# very different sizes (train is about 40 times bigger than test).
bar_labels = class_names + ["(no defect)"]
bar_colours = ["#2a78d6"] * num_classes + ["#a3a29b"]  # defects blue, no defect grey

fig, axes = plt.subplots(1, len(splits), figsize=(15, 4.5))
for ax, split in zip(axes, splits):
    values = box_counts[split] + [no_defect_counts[split]]
    bars = ax.bar(bar_labels, values, color=bar_colours)
    ax.bar_label(bars, padding=2)
    n_images = len(list((DATA_DIR / split / "images").glob("*.jpg")))
    ax.set_title(f"{split} ({n_images} images)")
    ax.set_ylabel("number of boxes / images")
    ax.tick_params(axis="x", rotation=45)
    ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.show()
