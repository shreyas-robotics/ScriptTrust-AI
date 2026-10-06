from sentence_transformers import SentenceTransformer, util


# ============================================================
# MODEL
# ============================================================

print("\nLoading semantic model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Semantic model loaded.\n")


# ============================================================
# STUDENT OCR TEXT
# ============================================================

student_answer = """
MREG records the electrical activity of the heart.
using electrodes placed on the skin. The electrodes
correct the electrical signals produced by the heart.
which are amplified and displayed as waveforms.
these waveforms help to assess the heart's rhythm.
"""


# ============================================================
# RUBRIC CRITERIA
# ============================================================

criteria = [
    {
        "name": "Electrical activity of the heart",
        "reference": "An ECG records the electrical activity of the heart."
    },
    {
        "name": "Mentions electrodes",
        "reference": "The electrical activity is detected using electrodes placed on the skin."
    },
    {
        "name": "Detection of electrical signals",
        "reference": "The electrodes detect electrical signals produced by the heart."
    },
    {
        "name": "Recording/display as waveform",
        "reference": "The electrical signals are amplified and recorded or displayed as waveforms."
    },
    {
        "name": "Generally correct overall explanation",
        "reference": "The answer correctly explains the general working principle of an ECG."
    }
]


# ============================================================
# SPLIT STUDENT ANSWER
# ============================================================

student_sentences = [
    line.strip()
    for line in student_answer.split("\n")
    if line.strip()
]


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

student_embeddings = model.encode(
    student_sentences,
    convert_to_tensor=True
)


# ============================================================
# SEMANTIC EVALUATION
# ============================================================

print("--- SEMANTIC EVALUATION ---\n")

total_score = 0

for criterion in criteria:

    reference_embedding = model.encode(
        criterion["reference"],
        convert_to_tensor=True
    )

    similarities = util.cos_sim(
        student_embeddings,
        reference_embedding
    ).flatten()

    best_score = float(similarities.max())

    # Threshold for prototype
    passed = best_score >= 0.45

    if passed:
        total_score += 1
        symbol = "✓"
    else:
        symbol = "✗"

    print(
        f"{symbol} {criterion['name']}"
    )

    print(
        f"   Semantic similarity: {best_score:.3f}"
    )


# ============================================================
# FINAL SCORE
# ============================================================

print()
print(
    f"Semantic Score: {total_score}/5"
)
print()