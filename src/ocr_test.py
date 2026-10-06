from PIL import Image, ImageEnhance, ImageFilter
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

image_path = "data/lines/line_3.jpg"

# Load image
image = Image.open(image_path).convert("RGB")

# Enlarge the handwriting
image = image.resize(
    (image.width * 2, image.height * 2)
)

# Improve contrast
enhancer = ImageEnhance.Contrast(image)
image = enhancer.enhance(1.5)

# Slight sharpening
image = image.filter(ImageFilter.SHARPEN)

# Load TrOCR
processor = TrOCRProcessor.from_pretrained(
    "microsoft/trocr-base-handwritten"
)

model = VisionEncoderDecoderModel.from_pretrained(
    "microsoft/trocr-base-handwritten"
)

# Prepare image
pixel_values = processor(
    images=image,
    return_tensors="pt"
).pixel_values

# Generate text
generated_ids = model.generate(
    pixel_values,
    max_new_tokens=100
)

text = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True
)[0]

print("\n--- PREPROCESSED OCR ---")
print(text)