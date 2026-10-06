from sentence_transformers import SentenceTransformer, util


# ============================================================
# LOAD SEMANTIC MODEL
# ============================================================

print("\nLoading semantic model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Semantic model loaded.\n")


# ============================================================
# ANSWER 6 OCR TEXT
# ============================================================

student_answer = """
MR.000 is a machine that takes pictures of the mean.
It is used in hospitals to see the size of the heart.
It can also show if there is any blood pressure problem.
The machine gives images on a screen which doctors
use for treatment.
"""

text = student_answer.lower()


# ============================================================
# RUBRIC
# ============================================================

criteria = [

    {
        "name": "Electrical activity of the heart",
        "reference": (
            "An ECG records the electrical activity "
            "of the heart."
        ),
        "required": [
            "electrical",
            "activity",
            "heart"
        ],
        "threshold": 0.55
    },

    {
        "name": "Mentions electrodes",
        "reference": (
            "The ECG uses electrodes placed on the skin."
        ),
        "required": [
            "electrode"
        ],
        "supporting": [
            "pad",
            "pads",
            "skin",
            "body"
        ],
        "threshold": 0.55
    },

    {
        "name": "Detection of electrical signals",
        "reference": (
            "The electrodes detect electrical signals "
            "produced by the heart."
        ),
        "required": [
            "electrical",
            "signal"
        ],
        "supporting": [
            "detect",
            "detects",
            "pick up",
            "pick",
            "measure"
        ],
        "threshold": 0.60
    },

    {
        "name": "Recording/display as waveform",
        "reference": (
            "The electrical signals are amplified and "
            "recorded or displayed as waveforms."
        ),
        "required": [
            "waveform"
        ],
        "supporting": [
            "recorded",
            "recording",
            "displayed",
            "display",
            "screen",
            "graph"
        ],
        "threshold": 0.60
    },

    {
        "name": "Generally correct overall explanation",
        "reference": (
            "The answer gives a generally correct "
            "explanation of how an ECG works."
        ),
        "required": [
            "ecg",
            "heart"
        ],
        "threshold": 0.50
    }
]


# ============================================================
# STUDENT EMBEDDING
# ============================================================

student_embedding = model.encode(
    student_answer,
    convert_to_tensor=True
)


# ============================================================
# EVALUATION
# ============================================================

print("--- HYBRID RUBRIC EVALUATION ---\n")

total = 0


for criterion in criteria:

    reference_embedding = model.encode(
        criterion["reference"],
        convert_to_tensor=True
    )

    similarity = float(
        util.cos_sim(
            student_embedding,
            reference_embedding
        )[0][0]
    )

    required_present = all(
        word in text
        for word in criterion["required"]
    )

    supporting = criterion.get(
        "supporting",
        []
    )

    supporting_present = True

    if supporting:
        supporting_present = any(
            phrase in text
            for phrase in supporting
        )

    # --------------------------------------------------------
    # DECISION
    # --------------------------------------------------------

    if criterion["name"] in [
        "Electrical activity of the heart",
        "Generally correct overall explanation"
    ]:

        passed = (
            required_present
            and similarity >= criterion["threshold"]
        )

    else:

        passed = (
            required_present
            and supporting_present
            and similarity >= criterion["threshold"]
        )

    if passed:
        symbol = "✓"
        total += 1
    else:
        symbol = "✗"

    print(
        f"{symbol} {criterion['name']}"
    )

    print(
        f"   Semantic similarity: {similarity:.3f}"
    )

    print(
        f"   Required concepts: {required_present}"
    )

    if supporting:
        print(
            f"   Supporting evidence: "
            f"{supporting_present}"
        )

    print()


# ============================================================
# FINAL SCORE
# ============================================================

print(
    f"Hybrid Score: {total}/5"
)

print()