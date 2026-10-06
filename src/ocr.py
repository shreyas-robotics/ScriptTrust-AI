from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Load the pretrained handwriting OCR model
processor = TrOCRProcessor.from_pretrained(
    "microsoft/trocr-small-handwritten"
)

model = VisionEncoderDecoderModel.from_pretrained(
    "microsoft/trocr-small-handwritten"
)

# Path to handwritten answer
image_path = "data/answers/answer_01.jpg"

# Open image
image = Image.open(image_path).convert("RGB")

# Prepare image for the model
pixel_values = processor(
    images=image,
    return_tensors="pt"
).pixel_values

# Generate text
generated_ids = model.generate(pixel_values)

# Convert model output to readable text
text = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True
)[0]

print("\n--- OCR RESULT ---")
print(text)