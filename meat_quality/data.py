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
