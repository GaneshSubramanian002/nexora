"""
Candidate explanation module.

Converts the ranking engine's numerical results into
human-readable, data-driven explanations.

This module does NOT use an LLM or API.
"""


def _percent(value):
    """Convert a score in either 0-1 or 0-100 format to percentage."""
    if value is None:
        return 0.0

    value = float(value)

    if value <= 1:
        return value * 100

    return value


def explain_candidate(candidate):
    """
    Generate an explanation for one ranked candidate.

    Expected input:
        {
            "candidate_id": "resume_07",
            "final_score": 91.4,
            "semantic_score": 0.94,
            "keyword_score": 0.89,
            "required_skill_score": 1.0,
            "matched_required": [...],
            "missing_required": [...],
            "matched_preferred": [...],
            "missing_preferred": [...]
        }

    Returns a dictionary containing:
        - scores
        - matched/missing skills
        - strengths
        - concerns
        - overall explanation
    """

    candidate_id = candidate.get("candidate_id", "Unknown candidate")

    final_score = _percent(candidate.get("final_score", 0))
    semantic_score = _percent(candidate.get("semantic_score", 0))
    keyword_score = _percent(candidate.get("keyword_score", 0))
    required_score = _percent(candidate.get("required_skill_score", 0))

    matched_required = candidate.get("matched_required", []) or []
    missing_required = candidate.get("missing_required", []) or []
    matched_preferred = candidate.get("matched_preferred", []) or []
    missing_preferred = candidate.get("missing_preferred", []) or []

    strengths = []
    concerns = []

    # Semantic score
    if semantic_score >= 85:
        strengths.append(
            f"Strong semantic match ({semantic_score:.1f}%)."
        )
    elif semantic_score >= 70:
        strengths.append(
            f"Good semantic match ({semantic_score:.1f}%)."
        )
    else:
        concerns.append(
            f"Lower semantic match ({semantic_score:.1f}%)."
        )

    # Keyword score
    if keyword_score >= 85:
        strengths.append(
            f"Strong keyword coverage ({keyword_score:.1f}%)."
        )
    elif keyword_score >= 70:
        strengths.append(
            f"Good keyword coverage ({keyword_score:.1f}%)."
        )
    else:
        concerns.append(
            f"Lower keyword coverage ({keyword_score:.1f}%)."
        )

    # Required skills
    if required_score >= 90:
        strengths.append("Matches almost all required skills.")
    elif required_score >= 70:
        strengths.append("Matches most required skills.")
    else:
        concerns.append("Several required skills are missing.")

    if matched_required:
        strengths.append(
            "Required skills matched: "
            + ", ".join(matched_required)
            + "."
        )

    if missing_required:
        concerns.append(
            "Missing required skills: "
            + ", ".join(missing_required)
            + "."
        )

    # Preferred skills
    if matched_preferred:
        strengths.append(
            "Preferred skills matched: "
            + ", ".join(matched_preferred)
            + "."
        )

    if missing_preferred:
        concerns.append(
            "Missing preferred skills: "
            + ", ".join(missing_preferred)
            + "."
        )

    # Overall explanation
    if final_score >= 85:
        overall = (
            f"{candidate_id} is a strong candidate with an overall "
            f"score of {final_score:.1f}%."
        )
    elif final_score >= 70:
        overall = (
            f"{candidate_id} is a good candidate with an overall "
            f"score of {final_score:.1f}%."
        )
    else:
        overall = (
            f"{candidate_id} has a lower overall match with a score "
            f"of {final_score:.1f}%."
        )

    return {
        "candidate_id": candidate_id,
        "final_score": final_score,
        "semantic_score": semantic_score,
        "keyword_score": keyword_score,
        "required_skill_score": required_score,
        "matched_required": matched_required,
        "missing_required": missing_required,
        "matched_preferred": matched_preferred,
        "missing_preferred": missing_preferred,
        "strengths": strengths,
        "concerns": concerns,
        "overall": overall,
    }
