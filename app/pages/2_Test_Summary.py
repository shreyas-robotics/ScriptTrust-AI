import streamlit as st
import pandas as pd


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="ScriptTrust-AI - Test Summary",
    page_icon="S",
    layout="wide",
)


# ============================================================
# HEADER
# ============================================================

st.title("ScriptTrust-AI")

st.subheader("Test Dataset Evaluation Summary")

st.caption(
    "Validation results for the handwritten ECG answer dataset."
)

st.divider()


# ============================================================
# TEST DATA
# ============================================================

data = [
    {
        "Answer": "answer_01.jpg",
        "Expected Score": 5,
        "AI Score": 5,
        "Result": "PASS",
    },
    {
        "Answer": "answer_02.jpg",
        "Expected Score": 4,
        "AI Score": 4,
        "Result": "PASS",
    },
    {
        "Answer": "answer_03.jpg",
        "Expected Score": 2,
        "AI Score": 2,
        "Result": "PASS",
    },
    {
        "Answer": "answer_04.jpg",
        "Expected Score": 3,
        "AI Score": 3,
        "Result": "PASS",
    },
    {
        "Answer": "answer_05.jpg",
        "Expected Score": 1,
        "AI Score": 1,
        "Result": "PASS",
    },
    {
        "Answer": "answer_06.jpg",
        "Expected Score": 0,
        "AI Score": 0,
        "Result": "PASS",
    },
]


df = pd.DataFrame(data)


# ============================================================
# CALCULATIONS
# ============================================================

total_answers = len(df)

exact_matches = (
    df["Expected Score"] == df["AI Score"]
).sum()

accuracy = (
    exact_matches / total_answers
) * 100

mean_absolute_error = (
    (df["Expected Score"] - df["AI Score"])
    .abs()
    .mean()
)


# ============================================================
# SUMMARY CARDS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Test Answers",
        total_answers,
    )

with col2:

    st.metric(
        "Exact Score Matches",
        f"{exact_matches}/{total_answers}",
    )

with col3:

    st.metric(
        "Score Accuracy",
        f"{accuracy:.0f}%",
    )

with col4:

    st.metric(
        "Mean Absolute Error",
        f"{mean_absolute_error:.2f}",
    )


st.divider()


# ============================================================
# VALIDATION STATUS
# ============================================================

st.markdown("### Validation Status")

if exact_matches == total_answers:

    st.success(
        "All test answers received the expected score."
    )

else:

    st.warning(
        f"{exact_matches} of {total_answers} "
        "test answers matched the expected score."
    )


# ============================================================
# RESULTS TABLE
# ============================================================

st.markdown("### Test Results")

display_df = df.copy()

display_df["Expected Score"] = (
    display_df["Expected Score"].astype(str) + "/5"
)

display_df["AI Score"] = (
    display_df["AI Score"].astype(str) + "/5"
)

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# SCORE COMPARISON
# ============================================================

st.markdown("### Expected vs AI Score")

chart_df = df.set_index("Answer")[
    ["Expected Score", "AI Score"]
]

st.bar_chart(chart_df)


# ============================================================
# CONCLUSION
# ============================================================

st.markdown("### Conclusion")

st.write(
    f"ScriptTrust-AI achieved an exact score match on "
    f"{exact_matches} out of {total_answers} test answers, "
    f"giving a score accuracy of {accuracy:.0f}%."
)

st.write(
    f"The mean absolute error between expected and AI-generated "
    f"scores is {mean_absolute_error:.2f} marks."
)