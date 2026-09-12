# Owned by Person 4 (explainability). Not implemented here - out of scope
# for Person 2.
from src.explainability import (
    explain_candidate,
    compare_candidates,
    check_jd_fairness,
)


candidate_a = {
    "candidate_id": "resume_07",
    "final_score": 91.4,
    "semantic_score": 0.94,
    "keyword_score": 0.89,
    "required_skill_score": 1.0,
    "matched_required": [
        "Python",
        "SQL",
        "Pandas",
        "FastAPI",
    ],
    "missing_required": [],
    "matched_preferred": [
        "Docker",
    ],
    "missing_preferred": [
        "AWS",
    ],
}


candidate_b = {
    "candidate_id": "resume_12",
    "final_score": 82.1,
    "semantic_score": 0.81,
    "keyword_score": 0.79,
    "required_skill_score": 0.75,
    "matched_required": [
        "Python",
        "SQL",
        "Pandas",
    ],
    "missing_required": [
        "FastAPI",
    ],
    "matched_preferred": [],
    "missing_preferred": [
        "Docker",
        "AWS",
    ],
}


def test_explain_candidate():
    result = explain_candidate(candidate_a)

    assert result["candidate_id"] == "resume_07"
    assert result["final_score"] == 91.4
    assert result["semantic_score"] == 94.0
    assert result["keyword_score"] == 89.0
    assert result["required_skill_score"] == 100.0

    assert "Python" in result["matched_required"]
    assert "FastAPI" in result["matched_required"]

    assert result["missing_required"] == []

    assert "Docker" in result["matched_preferred"]
    assert "AWS" in result["missing_preferred"]

    assert len(result["strengths"]) > 0


def test_compare_candidates():
    result = compare_candidates(
        candidate_a,
        candidate_b,
    )

    assert result["winner"] == "resume_07"
    assert result["score_a"] == 91.4
    assert result["score_b"] == 82.1

    assert round(result["score_difference"], 1) == 9.3

    assert len(result["reasons"]) > 0


def test_jd_fairness():
    jd = """
    We are looking for a young, energetic developer.
    Must have excellent communication skills.
    Native speaker preferred.
    """

    result = check_jd_fairness(jd)

    assert result["status"] == "REVIEW RECOMMENDED"
    assert result["warning_count"] > 0

    phrases = [
        warning["phrase"]
        for warning in result["warnings"]
    ]

    assert "young" in phrases
    assert "energetic" in phrases
    assert "must have" in phrases
    assert "native speaker" in phrases


if __name__ == "__main__":
    test_explain_candidate()
    test_compare_candidates()
    test_jd_fairness()

    print()
    print("====================================")
    print("ALL EXPLAINABILITY TESTS PASSED")
    print("====================================")
