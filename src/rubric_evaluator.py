from sentence_transformers import CrossEncoder


# ============================================================
# LOAD NLI MODEL
# ============================================================

print("\nLoading NLI model...")

model = CrossEncoder(
    "cross-encoder/nli-MiniLM2-L6-H768"
)

print("NLI model loaded.\n")


# ============================================================
# STUDENT ANSWER — ANSWER 3
# ============================================================

student_answer = """
An ECG is a test that is used to check the heart.
It uses small electrodes on the skin to pick up
signals from the heart. These signals are shown as
a graph and help to find if the heart is working
normally or not.
"""


# ============================================================
# RUBRIC
# ============================================================

criteria = [
    {
        "name": "Electrical activity of the heart",
        "hypothesis": (
            "An ECG records the electrical activity "
            "of the heart."
        )
    },
    {
        "name": "Mentions electrodes",
        "hypothesis": (
            "The ECG uses electrodes placed on the skin."
        )
    },
    {
        "name": "Detection of electrical signals",
        "hypothesis": (
            "The electrodes detect electrical signals "
            "produced by the heart."
        )
    },
    {
        "name": "Recording/display as waveform",
        "hypothesis": (
            "The electrical signals are amplified and "
            "recorded or displayed as waveforms."
        )
    },
    {
        "name": "Generally correct overall explanation",
        "hypothesis": (
            "The answer correctly explains the general "
            "working principle of an ECG."
        )
    }
]


# ============================================================
# EVALUATION
# ============================================================

print("--- NLI RUBRIC EVALUATION ---\n")

total = 0

for criterion in criteria:

    pair = [
        (
            student_answer,
            criterion["hypothesis"]
        )
    ]

    scores = model.predict(pair)

    scores = scores[0]

    # Model label order:
    # contradiction, entailment, neutral
    contradiction = float(scores[0])
    entailment = float(scores[1])
    neutral = float(scores[2])

    # Convert logits to probabilities
    import numpy as np

    probabilities = np.exp(scores)
    probabilities = probabilities / probabilities.sum()

    contradiction = float(probabilities[0])
    entailment = float(probabilities[1])
    neutral = float(probabilities[2])

    # Prototype grading rule
    if entailment >= 0.60:
        result = "✓"
        total += 1
    else:
        result = "✗"

    print(result, criterion["name"])

    print(
        f"   Entailment:    {entailment:.3f}"
    )

    print(
        f"   Neutral:       {neutral:.3f}"
    )

    print(
        f"   Contradiction: {contradiction:.3f}"
    )

    print()


# ============================================================
# FINAL SCORE
# ============================================================

print(
    f"NLI Score: {total}/5"
)