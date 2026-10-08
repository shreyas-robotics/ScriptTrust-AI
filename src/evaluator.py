import os
import re
import sys

import numpy as np

# ============================================================
# WINDOWS OUTPUT FIX
# ============================================================

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


from PIL import Image, ImageEnhance, ImageFilter

from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
)

from sentence_transformers import (
    SentenceTransformer,
    util,
)


# ============================================================
# FILES
# ============================================================

question_file = "data/questions/ecg_question.txt"
reference_file = "data/questions/ecg_reference_answer.txt"
rubric_file = "data/questions/ecg_rubric.txt"

lines_dir = "data/lines"


# ============================================================
# FILE READER
# ============================================================

def read_file(path):

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        return file.read().strip()


question = read_file(question_file)
reference_answer = read_file(reference_file)
rubric_text = read_file(rubric_file)


# ============================================================
# OCR MODEL
# ============================================================

print("\nLoading OCR model...")

processor = TrOCRProcessor.from_pretrained(
    "microsoft/trocr-base-handwritten"
)

ocr_model = VisionEncoderDecoderModel.from_pretrained(
    "microsoft/trocr-base-handwritten"
)

print("OCR model loaded.")


# ============================================================
# OCR FUNCTION
# ============================================================

def ocr_line(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    # Check whether the crop contains enough dark pixels
    gray = image.convert("L")

    pixels = np.array(gray)

    if np.sum(pixels < 180) < 500:
        return ""

    # Enlarge handwriting
    image = image.resize(
        (
            image.width * 2,
            image.height * 2,
        )
    )

    # Improve contrast
    image = ImageEnhance.Contrast(
        image
    ).enhance(1.5)

    # Sharpen
    image = image.filter(
        ImageFilter.SHARPEN
    )

    # TrOCR preprocessing
    pixel_values = processor(
        images=image,
        return_tensors="pt",
    ).pixel_values

    # OCR generation
    generated_ids = ocr_model.generate(
        pixel_values,
        max_new_tokens=100,
    )

    text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )[0]

    return text.strip()


# ============================================================
# RUN OCR
# ============================================================

print("\n--- OCR ---\n")

ocr_lines = []

files = sorted(
    file_name
    for file_name in os.listdir(lines_dir)
    if file_name.lower().endswith(".jpg")
)


for file_name in files:

    image_path = os.path.join(
        lines_dir,
        file_name,
    )

    text = ocr_line(
        image_path
    )

    if text:
        ocr_lines.append(text)

    print(
        f"{file_name}: {text}"
    )


if not ocr_lines:

    print(
        "\nNo readable handwriting detected."
    )

    raise SystemExit


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(text):

    text = text.lower()

    replacements = {

        # Common ECG OCR errors
        "mreg": "ecg",
        "mrecg": "ecg",
        "mrec": "ecg",
        "elg": "ecg",
        "mr.cg": "ecg",
        "mr cg": "ecg",
        "mr. cg": "ecg",
    }

    for wrong, correct in replacements.items():

        text = text.replace(
            wrong,
            correct,
        )

    # Remove punctuation
    text = re.sub(
        r"[^\w\s]",
        " ",
        text,
    )

    # Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


normalized_lines = [
    normalize_text(line)
    for line in ocr_lines
]

student_answer = " ".join(
    normalized_lines
)

student_text = student_answer.lower()


# ============================================================
# SEMANTIC MODEL
# ============================================================

print("\nLoading semantic model...")

semantic_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Semantic model loaded.")


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

        "required_all": [
            "electrical",
            "activity",
            "heart",
        ],

        "required_any": [],

        "supporting": [],

        "threshold": 0.55,
    },


    {
        "name": "Mentions electrodes",

        "reference": (
            "The ECG uses electrodes placed on "
            "the skin."
        ),

        "required_all": [],

        "required_any": [
            "electrode",
            "electrodes",
            "pad",
            "pads",
        ],

        "supporting": [
            "skin",
            "body",
        ],

        "threshold": 0.55,
    },


    {
        "name": "Detection of electrical signals",

        "reference": (
            "The electrodes detect electrical signals "
            "produced by the heart."
        ),

        "required_all": [
            "signal",
        ],

        "required_any": [],

        "supporting": [
            "detect",
            "detects",
            "pick up",
            "pick",
            "measure",
            "record",
        ],

        "threshold": 0.60,
    },


    {
        "name": "Recording/display as waveform",

        "reference": (
            "The electrical signals are recorded "
            "or displayed as waveforms."
        ),

        "required_all": [
            "waveform",
        ],

        "required_any": [],

        "supporting": [
            "recorded",
            "recording",
            "displayed",
            "display",
            "screen",
        ],

        "threshold": 0.60,
    },


    {
        "name": "Generally correct overall explanation",

        "reference": (
            "The answer correctly explains how an ECG "
            "works by describing the heart's electrical "
            "activity, electrodes, signals, and waveform output."
        ),

        "required_all": [
            "heart",
        ],

        "required_any": [],

        "supporting": [
            "electrical",
            "electrode",
            "electrodes",
            "signal",
            "signals",
            "waveform",
            "waveforms",
        ],

        # Changed from 0.55 to 0.60
        "threshold": 0.60,
    },

]


# ============================================================
# STUDENT SENTENCES
# ============================================================

student_sentences = [

    line.strip()

    for line in normalized_lines

    if line.strip()

]


student_embeddings = semantic_model.encode(

    student_sentences,

    convert_to_tensor=True,

)


# ============================================================
# HELPERS
# ============================================================

def contains_any(text, phrases):

    return any(
        phrase.lower() in text
        for phrase in phrases
    )


def canonical_concept(phrase):

    """
    Treat simple singular/plural variants as
    the same concept.
    """

    phrase = phrase.lower().strip()

    if " " in phrase:
        return phrase

    if phrase.endswith("s"):
        return phrase[:-1]

    return phrase


def count_unique_supporting_concepts(
    text,
    phrases,
):

    """
    Count unique supporting concepts rather than
    counting singular/plural variants separately.
    """

    found_concepts = set()

    for phrase in phrases:

        if phrase.lower() in text:

            concept = canonical_concept(
                phrase
            )

            found_concepts.add(
                concept
            )

    return len(found_concepts)


def get_confidence(
    similarity,
    required_present,
    supporting_present,
    passed,
):

    similarity_component = np.clip(
        (similarity - 0.25) / 0.65,
        0,
        1,
    )

    concept_component = (
        1.0
        if required_present
        else 0.0
    )

    support_component = (
        1.0
        if supporting_present
        else 0.0
    )

    confidence = (
        0.55 * similarity_component
        + 0.30 * concept_component
        + 0.15 * support_component
    )

    confidence = int(
        round(
            confidence * 100
        )
    )

    if not passed:

        confidence = min(
            confidence,
            55,
        )

    return confidence


def choose_evidence(
    criterion,
    similarities,
    required_present,
):

    if not required_present:
        return None

    required_all = criterion[
        "required_all"
    ]

    required_any = criterion[
        "required_any"
    ]

    supporting_phrases = criterion[
        "supporting"
    ]

    candidates = []

    for index, sentence in enumerate(
        normalized_lines
    ):

        all_hits = sum(
            word.lower() in sentence
            for word in required_all
        )

        any_hits = sum(
            word.lower() in sentence
            for word in required_any
        )

        supporting_hits = sum(
            phrase.lower() in sentence
            for phrase in supporting_phrases
        )

        if (
            all_hits > 0
            or any_hits > 0
            or supporting_hits > 0
        ):

            candidates.append(
                (
                    index,
                    all_hits,
                    any_hits,
                    supporting_hits,
                    float(
                        similarities[index]
                    ),
                )
            )

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: (
            item[1],
            item[2],
            item[3],
            item[4],
        ),
        reverse=True,
    )

    best_index = candidates[0][0]

    return normalized_lines[
        best_index
    ]


# ============================================================
# EVALUATION
# ============================================================

print(
    "\n--- EXPLAINABLE HYBRID EVALUATION ---\n"
)

total_score = 0

results = []


for index, criterion in enumerate(
    criteria,
    start=1,
):

    # --------------------------------------------------------
    # Semantic similarity
    # --------------------------------------------------------

    reference_embedding = semantic_model.encode(
        criterion["reference"],
        convert_to_tensor=True,
    )

    similarities = util.cos_sim(
        student_embeddings,
        reference_embedding,
    ).flatten()

    best_similarity = float(
        similarities.max()
    )

    # --------------------------------------------------------
    # Required concepts
    # --------------------------------------------------------

    required_all = criterion[
        "required_all"
    ]

    required_any = criterion[
        "required_any"
    ]

    all_present = all(
        word.lower() in student_text
        for word in required_all
    )

    any_present = (
        True
        if not required_any
        else contains_any(
            student_text,
            required_any,
        )
    )

    required_present = (
        all_present
        and any_present
    )

    # --------------------------------------------------------
    # Supporting concepts
    # --------------------------------------------------------

    supporting_phrases = criterion[
        "supporting"
    ]

    if supporting_phrases:

        supporting_count = (
            count_unique_supporting_concepts(
                student_text,
                supporting_phrases,
            )
        )

        supporting_present = (
            supporting_count >= 1
        )

    else:

        supporting_count = 0
        supporting_present = True

    # --------------------------------------------------------
    # Criterion 5 special rule
    # --------------------------------------------------------

    if index == 5:

        # Criterion 5 represents the overall quality
        # of the explanation. It requires at least
        # three distinct supporting concepts AND
        # explicit mention of waveform/output.
        supporting_present = (
            supporting_count >= 3
            and "waveform" in student_text
        )

    # --------------------------------------------------------
    # Final decision
    # --------------------------------------------------------

    passed = (
        required_present
        and supporting_present
        and best_similarity
        >= criterion["threshold"]
    )

    # --------------------------------------------------------
    # Score
    # --------------------------------------------------------

    if passed:

        total_score += 1

    # --------------------------------------------------------
    # Feedback
    # --------------------------------------------------------

    if passed:

        feedback = (
            "Criterion satisfied."
        )

    elif not required_present:

        missing = []

        for word in required_all:

            if word.lower() not in student_text:

                missing.append(word)

        if required_any:

            if not contains_any(
                student_text,
                required_any,
            ):

                missing.append(
                    "one of: "
                    + ", ".join(required_any)
                )

        if missing:

            feedback = (
                "Missing required concept(s): "
                + ", ".join(missing)
                + "."
            )

        else:

            feedback = (
                "Required concept not sufficiently "
                "supported."
            )

    elif not supporting_present:

        feedback = (
            "The answer does not provide "
            "enough supporting concepts."
        )

    elif best_similarity < criterion["threshold"]:

        feedback = (
            "The answer meaning is not "
            "sufficiently close to this criterion."
        )

    else:

        feedback = (
            "Criterion not sufficiently supported."
        )

    # --------------------------------------------------------
    # Evidence
    # --------------------------------------------------------

    evidence = choose_evidence(
        criterion,
        similarities,
        required_present,
    )

    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    confidence = get_confidence(
        best_similarity,
        required_present,
        supporting_present,
        passed,
    )

    # --------------------------------------------------------
    # Borderline
    # --------------------------------------------------------

    borderline = (
        abs(
            best_similarity
            - criterion["threshold"]
        ) <= 0.08
    )

    # --------------------------------------------------------
    # Store result
    # --------------------------------------------------------

    results.append(
        {
            "name":
                criterion["name"],

            "similarity":
                best_similarity,

            "confidence":
                confidence,

            "passed":
                passed,

            "evidence":
                evidence,

            "feedback":
                feedback,

            "required":
                required_present,

            "supporting":
                supporting_present,

            "borderline":
                borderline,
        }
    )

    # --------------------------------------------------------
    # Console output
    # --------------------------------------------------------

    status = (
        "PASS"
        if passed
        else "FAIL"
    )

    print(
        f"[{status}] "
        f"{criterion['name']}"
    )

    print(
        f"   Semantic similarity: "
        f"{best_similarity:.3f}"
    )

    print(
        f"   Confidence: "
        f"{confidence}%"
    )

    print(
        f"   Required concepts: "
        f"{required_present}"
    )

    if supporting_phrases:

        print(
            f"   Supporting evidence: "
            f"{supporting_present}"
        )

    if evidence:

        print(
            f'   Evidence: "{evidence}"'
        )

    else:

        print(
            "   Evidence: "
            "No sufficient evidence found."
        )

    print(
        f"   Feedback: "
        f"{feedback}"
    )

    print()


# ============================================================
# OVERALL CONFIDENCE
# ============================================================

overall_confidence = int(
    round(
        np.mean(
            [
                result["confidence"]
                for result in results
            ]
        )
    )
)


# ============================================================
# HUMAN REVIEW
# ============================================================

low_overall_confidence = (
    overall_confidence < 60
)

failed_borderline = any(
    (
        not result["passed"]
        and result["borderline"]
    )
    for result in results
)

ambiguous_failure = any(
    (
        not result["passed"]
        and result["similarity"] >= 0.50
    )
    for result in results
)

review_needed = (
    low_overall_confidence
    or failed_borderline
    or ambiguous_failure
)


# ============================================================
# FINAL RESULT
# ============================================================

print(
    "--- FINAL RESULT ---\n"
)

print(
    f"Question: {question}"
)

print(
    "\nMaximum Marks: 5"
)

print(
    f"Final Score: "
    f"{total_score}/5"
)

print(
    f"Overall Confidence: "
    f"{overall_confidence}%"
)

if review_needed:

    print(
        "Human Review: RECOMMENDED"
    )

else:

    print(
        "Human Review: NOT CURRENTLY REQUIRED"
    )


# ============================================================
# REFERENCE ANSWER
# ============================================================

print(
    "\nReference Answer:"
)

print(
    reference_answer
)


# ============================================================
# RUBRIC
# ============================================================

print(
    "\nRubric:"
)

print(
    rubric_text
)


# ============================================================
# STUDENT OCR ANSWER
# ============================================================

print(
    "\nStudent OCR Answer:"
)

print(
    student_answer
)

print()