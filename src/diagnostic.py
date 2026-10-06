import cv2
import os
import numpy as np


image_path = "data/answers/answer_02.jpg"
output_dir = "reports/debug"

os.makedirs(output_dir, exist_ok=True)


# ------------------------------------------------------------
# Load image
# ------------------------------------------------------------

image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(image_path)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

height, width = gray.shape

print()
print("IMAGE INFORMATION")
print("------------------")
print("Width :", width)
print("Height:", height)
print("Min pixel:", gray.min())
print("Max pixel:", gray.max())
print("Mean pixel:", gray.mean())
print()


# ------------------------------------------------------------
# Save grayscale image
# ------------------------------------------------------------

cv2.imwrite(
    os.path.join(output_dir, "01_gray.jpg"),
    gray
)


# ------------------------------------------------------------
# Test several thresholds
# ------------------------------------------------------------

for threshold in [120, 140, 160, 180, 200]:

    binary = cv2.threshold(
        gray,
        threshold,
        255,
        cv2.THRESH_BINARY_INV
    )[1]

    filename = os.path.join(
        output_dir,
        f"02_threshold_{threshold}.jpg"
    )

    cv2.imwrite(filename, binary)

    dark_pixels = np.sum(binary > 0)

    print(
        f"Threshold {threshold}: "
        f"{dark_pixels} dark pixels"
    )


# ------------------------------------------------------------
# Detect long horizontal structures
# ------------------------------------------------------------

binary = cv2.threshold(
    gray,
    180,
    255,
    cv2.THRESH_BINARY_INV
)[1]

kernel = cv2.getStructuringElement(
    cv2.MORPH_RECT,
    (80, 1)
)

horizontal = cv2.morphologyEx(
    binary,
    cv2.MORPH_OPEN,
    kernel
)

cv2.imwrite(
    os.path.join(
        output_dir,
        "03_horizontal_lines.jpg"
    ),
    horizontal
)


# ------------------------------------------------------------
# Remove detected horizontal lines
# ------------------------------------------------------------

clean = cv2.subtract(
    binary,
    horizontal
)

cv2.imwrite(
    os.path.join(
        output_dir,
        "04_clean.jpg"
    ),
    clean
)


# ------------------------------------------------------------
# Create horizontal projection visualization
# ------------------------------------------------------------

row_pixels = np.sum(
    clean > 0,
    axis=1
)

projection = np.zeros(
    (height, 600, 3),
    dtype=np.uint8
)

maximum = max(
    1,
    row_pixels.max()
)

for y in range(height):

    length = int(
        (row_pixels[y] / maximum) * 590
    )

    cv2.line(
        projection,
        (0, y),
        (length, y),
        (255, 255, 255),
        1
    )


cv2.imwrite(
    os.path.join(
        output_dir,
        "05_projection.jpg"
    ),
    projection
)


print()
print("Diagnostic files created in:")
print(output_dir)
print()