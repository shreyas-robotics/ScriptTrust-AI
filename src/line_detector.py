import cv2
import os
import sys
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

if len(sys.argv) > 1:
    image_path = sys.argv[1]
else:
    image_path = "data/answers/answer_02.jpg"

output_dir = "data/lines"
debug_dir = "reports/debug"

os.makedirs(output_dir, exist_ok=True)
os.makedirs(debug_dir, exist_ok=True)


# ============================================================
# LOAD IMAGE
# ============================================================

image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(
        f"Could not find image: {image_path}"
    )

height, width = image.shape[:2]


# ============================================================
# REMOVE OLD LINE CROPS
# ============================================================

for file in os.listdir(output_dir):
    if file.endswith(".jpg"):
        os.remove(
            os.path.join(output_dir, file)
        )


# ============================================================
# CROP TO ANSWER AREA
# ============================================================

top = int(height * 0.25)
bottom = int(height * 0.75)
left = int(width * 0.08)
right = int(width * 0.97)

roi = image[
    top:bottom,
    left:right
]


# ============================================================
# CONVERT TO HSV
# ============================================================

hsv = cv2.cvtColor(
    roi,
    cv2.COLOR_BGR2HSV
)

H = hsv[:, :, 0]
S = hsv[:, :, 1]
V = hsv[:, :, 2]


# ============================================================
# DETECT DARK BLUE HANDWRITING
# ============================================================

blue = (
    (H >= 85) &
    (H <= 140)
)

dark = V < 165

saturated = S > 45

ink = blue & dark & saturated


mask = np.zeros_like(V)

mask[ink] = 255


# ============================================================
# REMOVE SMALL NOISE
# ============================================================

kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (2, 2)
)

mask = cv2.morphologyEx(
    mask,
    cv2.MORPH_OPEN,
    kernel
)


# ============================================================
# CONNECT LETTER PARTS
# ============================================================

kernel = cv2.getStructuringElement(
    cv2.MORPH_RECT,
    (3, 2)
)

mask = cv2.morphologyEx(
    mask,
    cv2.MORPH_CLOSE,
    kernel
)


# ============================================================
# SAVE DEBUG MASK
# ============================================================

cv2.imwrite(
    os.path.join(
        debug_dir,
        "06_blue_ink_mask.jpg"
    ),
    mask
)


# ============================================================
# HORIZONTAL PROJECTION
# ============================================================

row_pixels = np.sum(
    mask > 0,
    axis=1
)

smooth = np.convolve(
    row_pixels,
    np.ones(5) / 5,
    mode="same"
)


# ============================================================
# FIND WRITING ROWS
# ============================================================

threshold = max(
    4,
    np.percentile(
        smooth,
        70
    )
)

active = smooth > threshold

regions = []

start = None

for y, value in enumerate(active):

    if value and start is None:
        start = y

    elif not value and start is not None:

        end = y

        if end - start >= 6:
            regions.append(
                [start, end]
            )

        start = None


if start is not None:
    regions.append(
        [start, len(active)]
    )


# ============================================================
# MERGE FRAGMENTS
# ============================================================

merged = []

for y1, y2 in regions:

    if not merged:
        merged.append(
            [y1, y2]
        )
        continue

    previous = merged[-1]

    if y1 - previous[1] <= 10:
        previous[1] = y2
    else:
        merged.append(
            [y1, y2]
        )


# ============================================================
# SAVE LINE CROPS
# ============================================================

count = 0

for y1, y2 in merged:

    if y2 - y1 > 45:
        continue

    region = mask[
        y1:y2,
        :
    ]

    columns = np.where(
        np.sum(
            region > 0,
            axis=0
        ) > 0
    )[0]

    if len(columns) == 0:
        continue

    x1 = max(
        0,
        left + columns[0] - 15
    )

    x2 = min(
        width,
        left + columns[-1] + 15
    )

    crop_top = max(
        0,
        top + y1 - 12
    )

    crop_bottom = min(
        height,
        top + y2 + 12
    )

    crop = image[
        crop_top:crop_bottom,
        x1:x2
    ]

    # Sanity checks
    if crop.shape[0] < 20:
        continue

    if crop.shape[1] < 150:
        continue

    count += 1

    filename = os.path.join(
        output_dir,
        f"line_{count}.jpg"
    )

    cv2.imwrite(
        filename,
        crop
    )


# ============================================================
# RESULT
# ============================================================

print()
print(
    f"Input image: {image_path}"
)

print(
    f"Detected {count} handwriting lines."
)

print()

for i in range(
    1,
    count + 1
):
    print(
        f"line_{i}.jpg"
    )

print()

print(
    "Debug mask:"
)

print(
    "reports/debug/06_blue_ink_mask.jpg"
)

print()