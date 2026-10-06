import io
import os
import re
import sys
import hashlib
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape

import streamlit as st
from PIL import Image

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as PDFImage,
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ANSWERS_DIR = PROJECT_ROOT / "data" / "answers"
REPORTS_DIR = PROJECT_ROOT / "reports"

CURRENT_IMAGE = ANSWERS_DIR / "current_upload.jpg"

DETECTOR = PROJECT_ROOT / "src" / "line_detector.py"
EVALUATOR = PROJECT_ROOT / "src" / "evaluator.py"


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "evaluation_output": None,
    "detector_stdout": "",
    "detector_stderr": "",
    "evaluator_stderr": "",
    "upload_key": None,
    "examiner_mark": None,
    "examiner_note": "",
    "mark_finalized": False,
}

for key, value in DEFAULT_STATE.items():

    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="ScriptTrust-AI",
    page_icon="S",
    layout="wide",
)


# ============================================================
# CUSTOM UI STYLE
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.6rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 1.15rem;
        margin-bottom: 0.4rem;
    }

    .description {
        color: #9aa0aa;
        font-size: 0.95rem;
    }

    .score-card {
        padding: 1.1rem;
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.10);
        background: rgba(255,255,255,0.035);
        min-height: 125px;
    }

    .score-label {
        font-size: 0.85rem;
        color: #a5aab4;
        margin-bottom: 0.35rem;
    }

    .score-value {
        font-size: 2rem;
        font-weight: 700;
    }

    .score-small {
        font-size: 0.82rem;
        color: #8f95a0;
        margin-top: 0.35rem;
    }

    .section-title {
        font-size: 1.35rem;
        font-weight: 650;
        margin-top: 0.4rem;
        margin-bottom: 0.8rem;
    }

    .review-card {
        padding: 1rem 1.2rem;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.10);
        background: rgba(255,255,255,0.035);
        margin-top: 0.8rem;
    }

    .status-label {
        font-weight: 650;
        font-size: 0.95rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">ScriptTrust-AI</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">AI Handwritten Exam Evaluation System</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="description">'
    "AI-assisted assessment with semantic rubric evaluation "
    "and human-in-the-loop review."
    "</div>",
    unsafe_allow_html=True,
)

st.divider()


# ============================================================
# QUESTION
# ============================================================

QUESTION = "Explain the working principle of an ECG."

st.markdown(
    '<div class="section-title">Question</div>',
    unsafe_allow_html=True,
)

st.info(QUESTION)


# ============================================================
# RESULT PARSER
# ============================================================

def parse_results(output):

    lines = output.splitlines()

    results = []
    current = None

    score = 0
    maximum = 5
    overall_confidence = 0
    review = "UNKNOWN"

    for raw_line in lines:

        line = raw_line.strip()

        # ----------------------------------------------------
        # CRITERION HEADER
        # ----------------------------------------------------

        criterion_match = re.match(
            r"^(?:\[(PASS|FAIL)\]|(PASS|FAIL)|([✓✗]))"
            r"\s*(?::\s*)?(.*)$",
            line,
            re.IGNORECASE,
        )

        if criterion_match:

            if current is not None:
                results.append(current)

            status_word = (
                criterion_match.group(1)
                or criterion_match.group(2)
            )

            symbol = criterion_match.group(3)

            name = criterion_match.group(4).strip()

            if symbol == "✓":
                status = "PASS"

            elif symbol == "✗":
                status = "FAIL"

            elif status_word:
                status = status_word.upper()

            else:
                status = "UNKNOWN"

            current = {
                "status": status,
                "name": name,
                "similarity": 0.0,
                "confidence": 0,
                "required": True,
                "supporting": True,
                "evidence": "",
                "feedback": "",
            }

            continue

        # ----------------------------------------------------
        # FINAL SCORE
        # ----------------------------------------------------

        score_match = re.search(
            r"Final Score:\s*(\d+)\s*/\s*(\d+)",
            line,
            re.IGNORECASE,
        )

        if score_match:

            score = int(score_match.group(1))
            maximum = int(score_match.group(2))

            continue

        # ----------------------------------------------------
        # OVERALL CONFIDENCE
        # ----------------------------------------------------

        confidence_match = re.search(
            r"Overall Confidence:\s*(\d+)%",
            line,
            re.IGNORECASE,
        )

        if confidence_match:

            overall_confidence = int(
                confidence_match.group(1)
            )

            continue

        # ----------------------------------------------------
        # HUMAN REVIEW
        # ----------------------------------------------------

        review_match = re.search(
            r"Human Review:\s*(.+)",
            line,
            re.IGNORECASE,
        )

        if review_match:

            review = review_match.group(1).strip()

            continue

        if current is None:
            continue

        # ----------------------------------------------------
        # SEMANTIC SIMILARITY
        # ----------------------------------------------------

        similarity_match = re.search(
            r"Semantic similarity:\s*([0-9.]+)",
            line,
            re.IGNORECASE,
        )

        if similarity_match:

            current["similarity"] = float(
                similarity_match.group(1)
            )

            continue

        # ----------------------------------------------------
        # CRITERION CONFIDENCE
        # ----------------------------------------------------

        criterion_confidence_match = re.search(
            r"Confidence:\s*(\d+)%",
            line,
            re.IGNORECASE,
        )

        if criterion_confidence_match:

            current["confidence"] = int(
                criterion_confidence_match.group(1)
            )

            continue

        # ----------------------------------------------------
        # REQUIRED CONCEPTS
        # ----------------------------------------------------

        required_match = re.search(
            r"Required concepts:\s*(True|False)",
            line,
            re.IGNORECASE,
        )

        if required_match:

            current["required"] = (
                required_match.group(1).lower()
                == "true"
            )

            continue

        # ----------------------------------------------------
        # SUPPORTING EVIDENCE
        # ----------------------------------------------------

        supporting_match = re.search(
            r"Supporting evidence:\s*(True|False)",
            line,
            re.IGNORECASE,
        )

        if supporting_match:

            current["supporting"] = (
                supporting_match.group(1).lower()
                == "true"
            )

            continue

        # ----------------------------------------------------
        # EVIDENCE
        # ----------------------------------------------------

        if line.lower().startswith("evidence:"):

            current["evidence"] = (
                line.split(":", 1)[1].strip()
            )

            continue

        # ----------------------------------------------------
        # FEEDBACK
        # ----------------------------------------------------

        if line.lower().startswith("feedback:"):

            current["feedback"] = (
                line.split(":", 1)[1].strip()
            )

            continue

    if current is not None:

        results.append(current)

    return (
        results,
        score,
        maximum,
        overall_confidence,
        review,
    )


# ============================================================
# PDF REPORT GENERATOR
# ============================================================

def build_pdf_report(
    image,
    results,
    ai_score,
    maximum,
    overall_confidence,
    review,
    final_mark,
    examiner_note,
    question,
):

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
        title="ScriptTrust-AI Evaluation Report",
        author="ScriptTrust-AI",
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=20,
        leading=24,
        spaceAfter=8,
    )

    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Heading2"],
        alignment=TA_CENTER,
        fontSize=11,
        leading=14,
        spaceAfter=14,
    )

    heading_style = ParagraphStyle(
        "ReportHeading",
        parent=styles["Heading2"],
        fontSize=13,
        leading=16,
        spaceBefore=10,
        spaceAfter=6,
    )

    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["BodyText"],
        fontSize=9,
        leading=12,
    )

    small_style = ParagraphStyle(
        "ReportSmall",
        parent=styles["BodyText"],
        fontSize=7.5,
        leading=10,
    )

    story = []

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "ScriptTrust-AI",
            title_style,
        )
    )

    story.append(
        Paragraph(
            "Handwritten Exam Evaluation Report",
            subtitle_style,
        )
    )

    story.append(
        Paragraph(
            "<b>Date:</b> "
            + datetime.now().strftime("%d-%m-%Y %H:%M"),
            body_style,
        )
    )

    story.append(Spacer(1, 8))

    # --------------------------------------------------------
    # QUESTION
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "Question",
            heading_style,
        )
    )

    story.append(
        Paragraph(
            escape(question),
            body_style,
        )
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "Evaluation Summary",
            heading_style,
        )
    )

    review_text = (
        "Recommended"
        if "RECOMMENDED" in review.upper()
        else "Not required"
    )

    summary_data = [
        [
            "AI Suggested Score",
            f"{ai_score}/{maximum}",
        ],
        [
            "Examiner Final Score",
            f"{final_mark}/{maximum}",
        ],
        [
            "Overall Confidence",
            f"{overall_confidence}%",
        ],
        [
            "Human Review",
            review_text,
        ],
    ]

    summary_table = Table(
        summary_data,
        colWidths=[180, 180],
    )

    summary_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, -1),
                    colors.whitesmoke,
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey,
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP",
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    9,
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),
            ]
        )
    )

    story.append(summary_table)

    # --------------------------------------------------------
    # RUBRIC
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "Rubric Evaluation",
            heading_style,
        )
    )

    table_data = [
        [
            Paragraph("<b>Criterion</b>", small_style),
            Paragraph("<b>Status</b>", small_style),
            Paragraph("<b>Similarity</b>", small_style),
            Paragraph("<b>Confidence</b>", small_style),
            Paragraph(
                "<b>Evidence / Feedback</b>",
                small_style,
            ),
        ]
    ]

    for index, result in enumerate(
        results,
        start=1,
    ):

        status_text = (
            "SATISFIED"
            if result["status"] == "PASS"
            else "NOT SATISFIED"
        )

        details = []

        if result["evidence"].strip():

            details.append(
                "<b>Evidence:</b> "
                + escape(result["evidence"])
            )

        if result["feedback"].strip():

            details.append(
                "<b>Feedback:</b> "
                + escape(result["feedback"])
            )

        if not details:

            details.append("No additional details.")

        table_data.append(
            [
                Paragraph(
                    f"{index}. "
                    f"{escape(result['name'])}",
                    small_style,
                ),
                Paragraph(
                    status_text,
                    small_style,
                ),
                Paragraph(
                    f"{result['similarity']:.3f}",
                    small_style,
                ),
                Paragraph(
                    f"{result['confidence']}%",
                    small_style,
                ),
                Paragraph(
                    "<br/>".join(details),
                    small_style,
                ),
            ]
        )

    rubric_table = Table(
        table_data,
        colWidths=[
            125,
            70,
            60,
            65,
            155,
        ],
        repeatRows=1,
    )

    rubric_table.setStyle(
        TableStyle(
            [
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.4,
                    colors.grey,
                ),
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.lightgrey,
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP",
                ),
                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    5,
                ),
                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    5,
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    5,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    5,
                ),
            ]
        )
    )

    story.append(rubric_table)

    # --------------------------------------------------------
    # EXAMINER NOTE
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "Examiner Note",
            heading_style,
        )
    )

    note = (
        examiner_note.strip()
        if examiner_note.strip()
        else "No examiner note provided."
    )

    story.append(
        Paragraph(
            escape(note),
            body_style,
        )
    )

    # --------------------------------------------------------
    # ANSWER IMAGE
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "Student Handwritten Answer",
            heading_style,
        )
    )

    temp_image = None

    try:

        with tempfile.NamedTemporaryFile(
            suffix=".jpg",
            delete=False,
        ) as temp_file:

            temp_image = temp_file.name

        image.convert("RGB").save(
            temp_image,
            format="JPEG",
            quality=95,
        )

        pdf_image = PDFImage(
            temp_image,
            width=6.7 * inch,
            height=8.5 * inch,
            kind="proportional",
        )

        story.append(pdf_image)

    finally:

        pass

    story.append(Spacer(1, 12))

    story.append(
        Paragraph(
            "AI-assisted assessment. "
            "The final mark is controlled by the examiner.",
            small_style,
        )
    )

    doc.build(story)

    if temp_image:

        try:

            Path(temp_image).unlink()

        except OSError:

            pass

    buffer.seek(0)

    return buffer.getvalue()


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_results(output):

    (
        results,
        score,
        maximum,
        overall_confidence,
        review,
    ) = parse_results(output)

    passed_count = sum(
        1
        for result in results
        if result["status"] == "PASS"
    )

    total_criteria = len(results)

    # ========================================================
    # SUMMARY
    # ========================================================

    st.divider()

    st.markdown(
        '<div class="section-title">Evaluation Summary</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.markdown(
            f"""
            <div class="score-card">
                <div class="score-label">AI Suggested Score</div>
                <div class="score-value">{score}/{maximum}</div>
                <div class="score-small">
                    Automated evaluation
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:

        st.markdown(
            f"""
            <div class="score-card">
                <div class="score-label">Overall Confidence</div>
                <div class="score-value">{overall_confidence}%</div>
                <div class="score-small">
                    AI confidence
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:

        st.markdown(
            f"""
            <div class="score-card">
                <div class="score-label">Rubric Criteria</div>
                <div class="score-value">
                    {passed_count}/{total_criteria}
                </div>
                <div class="score-small">
                    Criteria satisfied
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:

        if "RECOMMENDED" in review.upper():

            st.markdown(
                """
                <div class="score-card">
                    <div class="score-label">Review Status</div>
                    <div class="score-value">Review</div>
                    <div class="score-small">
                        Human examiner required
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        else:

            st.markdown(
                """
                <div class="score-card">
                    <div class="score-label">Review Status</div>
                    <div class="score-value">Clear</div>
                    <div class="score-small">
                        No review flag
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ========================================================
    # REVIEW NOTICE
    # ========================================================

    if "RECOMMENDED" in review.upper():

        st.warning(
            "This answer should be reviewed by a human examiner "
            "before the mark is finalized."
        )

    else:

        st.success(
            "The system did not flag this answer for human review."
        )

    # ========================================================
    # RUBRIC EVALUATION
    # ========================================================

    st.markdown(
        '<div class="section-title">Rubric Evaluation</div>',
        unsafe_allow_html=True,
    )

    if not results:

        st.error(
            "Criterion results could not be extracted."
        )

    else:

        for index, result in enumerate(
            results,
            start=1,
        ):

            if result["status"] == "PASS":

                st.success(
                    f"✓ Criterion {index}: "
                    f"{result['name']} — SATISFIED"
                )

            else:

                st.error(
                    f"✗ Criterion {index}: "
                    f"{result['name']} — NOT SATISFIED"
                )

            with st.expander(
                f"Criterion {index} Details"
            ):

                col1, col2 = st.columns(2)

                with col1:

                    st.write(
                        "**Semantic similarity:** "
                        f"{result['similarity']:.3f}"
                    )

                    st.write(
                        "**Confidence:** "
                        f"{result['confidence']}%"
                    )

                with col2:

                    st.write(
                        "**Required concepts:** "
                        + (
                            "Present"
                            if result["required"]
                            else "Missing"
                        )
                    )

                    st.write(
                        "**Supporting evidence:** "
                        + (
                            "Present"
                            if result["supporting"]
                            else "Missing"
                        )
                    )

                st.markdown("**Evidence**")

                if (
                    result["evidence"]
                    and result["evidence"].lower()
                    != "no sufficient evidence found."
                ):

                    st.info(result["evidence"])

                else:

                    st.write(
                        "No sufficient evidence found."
                    )

                st.markdown("**Feedback**")

                if result["feedback"]:

                    st.write(result["feedback"])

                else:

                    st.write(
                        "No additional feedback."
                    )

    # ========================================================
    # TECHNICAL OUTPUT
    # ========================================================

    with st.expander(
        "Technical Evaluation Log"
    ):

        st.code(
            output,
            language="text",
        )


# ============================================================
# UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload handwritten answer",
    type=["jpg", "jpeg", "png"],
)


# ============================================================
# MAIN APPLICATION
# ============================================================

if uploaded_file is not None:

    file_bytes = uploaded_file.getvalue()

    current_upload_key = (
        uploaded_file.name
        + ":"
        + hashlib.md5(file_bytes).hexdigest()
    )

    # --------------------------------------------------------
    # RESET WHEN NEW FILE IS UPLOADED
    # --------------------------------------------------------

    if current_upload_key != st.session_state["upload_key"]:

        st.session_state["upload_key"] = current_upload_key

        st.session_state["evaluation_output"] = None
        st.session_state["detector_stdout"] = ""
        st.session_state["detector_stderr"] = ""
        st.session_state["evaluator_stderr"] = ""

        st.session_state["examiner_mark"] = None
        st.session_state["examiner_note"] = ""

        st.session_state["mark_finalized"] = False

    # --------------------------------------------------------
    # READ IMAGE
    # --------------------------------------------------------

    try:

        image = Image.open(
            io.BytesIO(file_bytes)
        ).convert("RGB")

    except Exception as error:

        st.error(
            f"Could not read image: {error}"
        )

        st.stop()

    # --------------------------------------------------------
    # SAVE IMAGE
    # --------------------------------------------------------

    ANSWERS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    image.save(
        CURRENT_IMAGE,
        format="JPEG",
        quality=95,
    )

    # --------------------------------------------------------
    # DISPLAY IMAGE
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Uploaded Answer</div>',
        unsafe_allow_html=True,
    )

    st.image(
        image,
        caption="Student handwritten answer",
        width="stretch",
    )

    # ========================================================
    # EVALUATE
    # ========================================================

    if st.button(
        "Evaluate Answer",
        type="primary",
        key="evaluate_answer",
    ):

        # ----------------------------------------------------
        # LINE DETECTION
        # ----------------------------------------------------

        with st.spinner(
            "Detecting handwriting lines..."
        ):

            detector_result = subprocess.run(
                [
                    sys.executable,
                    str(DETECTOR),
                    str(CURRENT_IMAGE),
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )

        if detector_result.returncode != 0:

            st.error(
                "Line detection failed."
            )

            with st.expander(
                "Technical Details"
            ):

                st.code(
                    detector_result.stderr,
                    language="text",
                )

            st.stop()

        # ----------------------------------------------------
        # DETECTED LINES
        # ----------------------------------------------------

        detected_match = re.search(
            r"Detected\s+(\d+)\s+handwriting lines",
            detector_result.stdout,
            re.IGNORECASE,
        )

        detected_lines = (
            int(detected_match.group(1))
            if detected_match
            else 0
        )

        if detected_lines == 0:

            st.error(
                "No handwriting lines were detected."
            )

            with st.expander(
                "Line Detection Log"
            ):

                st.code(
                    detector_result.stdout,
                    language="text",
                )

            st.stop()

        # ----------------------------------------------------
        # AI EVALUATION
        # ----------------------------------------------------

        with st.spinner(
            f"Reading {detected_lines} handwriting lines "
            "and evaluating the answer..."
        ):

            environment = os.environ.copy()

            environment["PYTHONIOENCODING"] = "utf-8"
            environment["PYTHONUTF8"] = "1"

            evaluator_result = subprocess.run(
                [
                    sys.executable,
                    str(EVALUATOR),
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=environment,
            )

        if evaluator_result.returncode != 0:

            st.error(
                "AI evaluation failed."
            )

            with st.expander(
                "Technical Details"
            ):

                st.code(
                    evaluator_result.stderr,
                    language="text",
                )

            st.stop()

        # ----------------------------------------------------
        # SAVE RESULTS
        # ----------------------------------------------------

        st.session_state["evaluation_output"] = (
            evaluator_result.stdout
        )

        st.session_state["detector_stdout"] = (
            detector_result.stdout
        )

        st.session_state["detector_stderr"] = (
            detector_result.stderr
        )

        st.session_state["evaluator_stderr"] = (
            evaluator_result.stderr
        )

        (
            _results,
            ai_score,
            _maximum,
            _overall_confidence,
            _review,
        ) = parse_results(
            evaluator_result.stdout
        )

        st.session_state["examiner_mark"] = ai_score
        st.session_state["examiner_note"] = ""
        st.session_state["mark_finalized"] = False


# ============================================================
# DISPLAY SAVED RESULTS
# ============================================================

if st.session_state["evaluation_output"]:

    display_results(
        st.session_state["evaluation_output"]
    )

    (
        results,
        ai_score,
        maximum,
        overall_confidence,
        review,
    ) = parse_results(
        st.session_state["evaluation_output"]
    )

    # ========================================================
    # EXAMINER REVIEW
    # ========================================================

    st.divider()

    st.markdown(
        '<div class="section-title">Examiner Review</div>',
        unsafe_allow_html=True,
    )

    if st.session_state["mark_finalized"]:

        st.success(
            "Assessment finalized by examiner."
        )

    else:

        st.info(
            f"AI suggested mark: **{ai_score}/{maximum}**"
        )

    review_col1, review_col2 = st.columns(2)

    with review_col1:

        st.number_input(
            "Final Mark",
            min_value=0,
            max_value=maximum,
            step=1,
            key="examiner_mark",
        )

    with review_col2:

        st.text_input(
            "Examiner Note",
            key="examiner_note",
            placeholder="Optional reason for the final mark",
        )

    # ========================================================
    # FINALIZE
    # ========================================================

    if not st.session_state["mark_finalized"]:

        if st.button(
            "Finalize Mark",
            type="primary",
            key="finalize_mark",
        ):

            st.session_state["mark_finalized"] = True

            st.rerun()

    # ========================================================
    # FINALIZED RESULT
    # ========================================================

    if st.session_state["mark_finalized"]:

        st.markdown(
            """
            <div class="review-card">
                <div class="status-label">
                    Finalized Assessment
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        final_col1, final_col2 = st.columns(2)

        with final_col1:

            st.metric(
                "AI Suggested",
                f"{ai_score}/{maximum}",
            )

        with final_col2:

            st.metric(
                "Examiner Final",
                f"{st.session_state['examiner_mark']}/{maximum}",
            )

        if st.session_state["examiner_note"].strip():

            st.info(
                f"Examiner Note: "
                f"{st.session_state['examiner_note']}"
            )

        # ====================================================
        # PDF REPORT
        # ====================================================

        try:

            pdf_bytes = build_pdf_report(
                image=image,
                results=results,
                ai_score=ai_score,
                maximum=maximum,
                overall_confidence=overall_confidence,
                review=review,
                final_mark=st.session_state["examiner_mark"],
                examiner_note=st.session_state["examiner_note"],
                question=QUESTION,
            )

            st.download_button(
                "Download Evaluation Report (PDF)",
                data=pdf_bytes,
                file_name=(
                    "ScriptTrust-AI_Evaluation_Report.pdf"
                ),
                mime="application/pdf",
                type="secondary",
                use_container_width=True,
            )

        except Exception as error:

            st.error(
                f"Could not generate PDF report: {error}"
            )

    # ========================================================
    # LOGS
    # ========================================================

    with st.expander(
        "Line Detection Log"
    ):

        st.code(
            st.session_state["detector_stdout"],
            language="text",
        )

        if st.session_state["detector_stderr"].strip():

            st.caption("Detector warnings")

            st.code(
                st.session_state["detector_stderr"],
                language="text",
            )

    if st.session_state["evaluator_stderr"].strip():

        with st.expander(
            "Evaluation Warnings"
        ):

            st.code(
                st.session_state["evaluator_stderr"],
                language="text",
            )


# ============================================================
# INITIAL MESSAGE
# ============================================================

elif uploaded_file is None:

    st.info(
        "Upload a handwritten answer image to begin."
    )