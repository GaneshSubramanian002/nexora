"""
Candidate comparison module.

Explains why one candidate ranked higher than another
using the actual ranking metrics.
"""

from .explanation import _percent


def compare_candidates(candidate_a, candidate_b):
    """
    Compare two candidates and explain why one ranked higher.

    The comparison is based on:
        - final score
        - semantic score
        - keyword score
        - required skill score
        - number of matched required skills
    """

    id_a = candidate_a.get("candidate_id", "Candidate A")
    id_b = candidate_b.get("candidate_id", "Candidate B")

    score_a = _percent(candidate_a.get("final_score", 0))
    score_b = _percent(candidate_b.get("final_score", 0))

    semantic_a = _percent(candidate_a.get("semantic_score", 0))
    semantic_b = _percent(candidate_b.get("semantic_score", 0))

    keyword_a = _percent(candidate_a.get("keyword_score", 0))
    keyword_b = _percent(candidate_b.get("keyword_score", 0))

    required_a = _percent(candidate_a.get("required_skill_score", 0))
    required_b = _percent(candidate_b.get("required_skill_score", 0))

    matched_required_a = candidate_a.get("matched_required", []) or []
    matched_required_b = candidate_b.get("matched_required", []) or []

    reasons = []

    # Overall score
    if score_a != score_b:
        winner = id_a if score_a > score_b else id_b

        reasons.append({
            "category": "Overall Score",
            "winner": winner,
            "text": (
                f"{winner} has the higher overall score."
            )
        })

    # Semantic score
    if semantic_a != semantic_b:
        winner = id_a if semantic_a > semantic_b else id_b

        reasons.append({
            "category": "Semantic Match",
            "winner": winner,
            "text": (
                f"{id_a}: {semantic_a:.1f}% vs "
                f"{id_b}: {semantic_b:.1f}%."
            )
        })

    # Keyword score
    if keyword_a != keyword_b:
        winner = id_a if keyword_a > keyword_b else id_b

        reasons.append({
            "category": "Keyword Match",
            "winner": winner,
            "text": (
                f"{id_a}: {keyword_a:.1f}% vs "
                f"{id_b}: {keyword_b:.1f}%."
            )
        })

    # Required skills
    if required_a != required_b:
        winner = id_a if required_a > required_b else id_b

        reasons.append({
            "category": "Required Skills",
            "winner": winner,
            "text": (
                f"{id_a}: {required_a:.1f}% vs "
                f"{id_b}: {required_b:.1f}%."
            )
        })

    # Matched required skill count
    if len(matched_required_a) != len(matched_required_b):
        winner = (
            id_a
            if len(matched_required_a) > len(matched_required_b)
            else id_b
        )

        reasons.append({
            "category": "Matched Required Skills",
            "winner": winner,
            "text": (
                f"{id_a} matched {len(matched_required_a)} "
                f"required skills, while {id_b} matched "
                f"{len(matched_required_b)}."
            )
        })

    # Determine final winner
    if score_a > score_b:
        winner = id_a
        loser = id_b
    elif score_b > score_a:
        winner = id_b
        loser = id_a
    else:
        winner = "Tie"
        loser = None

    difference = abs(score_a - score_b)

    if winner == "Tie":
        summary = (
            f"{id_a} and {id_b} have the same overall score."
        )
    else:
        summary = (
            f"{winner} ranked higher by {difference:.1f} "
            f"percentage points."
        )

    return {
        "winner": winner,
        "candidate_a": id_a,
        "candidate_b": id_b,
        "score_a": score_a,
        "score_b": score_b,
        "score_difference": difference,
        "summary": summary,
        "reasons": reasons,
    }
