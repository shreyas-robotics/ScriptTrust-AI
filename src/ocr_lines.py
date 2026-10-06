from PIL import Image, ImageEnhance, ImageFilter
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import os
import numpy as np

# Load better handwritten OCR model
processor = TrOCRProcessor.from_pretrained(
    "microsoft/trocr-base-handwritten"
)

model = VisionEncoderDecoderModel.from_pretrained(
    "microsoft/trocr-base-handwritten"
)

lines_dir = "data/lines"

files = sorted(
    f for f in os.listdir(lines_dir)
    if f.endswith(".jpg")
)

print("\n--- OCR RESULTS ---\n")

for file in files:

    image_path = os.path.join(lines_dir, file)

    image = Image.open(image_path).convert("RGB")

    # Check for mostly empty crop
    gray = image.convert("L")
    pixels = np.array(gray)

    if np.sum(pixels < 180) < 500:
        print(f"{file}: SKIPPED (empty)")
        continue

    # Enlarge handwriting
    image = image.resize(
        (image.width * 2, image.height * 2)
    )

    # Improve contrast
    image = ImageEnhance.Contrast(image).enhance(1.5)

    # Sharpen handwriting
    image = image.filter(ImageFilter.SHARPEN)

    # OCR
    pixel_values = processor(
        images=image,
        return_tensors="pt"
    ).pixel_values

    generated_ids = model.generate(
        pixel_values,
        max_new_tokens=100
    )

    text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    print(f"{file}: {text}")