"""
Job description fairness checker.

This is a lightweight, rule-based screening tool.
It flags potentially problematic wording in a job description.

It does not make a legal determination of discrimination.
"""


import re


FAIRNESS_PATTERNS = {
    "must have": (
        "Absolute requirement wording may unnecessarily "
        "narrow the candidate pool."
    ),
    "native speaker": (
        "Language requirement may exclude otherwise "
        "qualified candidates."
    ),
    "young": (
        "Age-related wording may introduce age bias."
    ),
    "recent graduate": (
        "Could unnecessarily restrict candidates based "
        "on graduation timing."
    ),
    "rockstar": (
        "Potentially subjective or exclusionary wording."
    ),
    "ninja": (
        "Potentially subjective or exclusionary wording."
    ),
    "culture fit": (
        "Subjective wording may introduce hidden bias."
    ),
    "energetic": (
        "Could be interpreted as an age-related preference."
    ),
}


def check_jd_fairness(job_description):
    """
    Scan a job description for potentially biased wording.

    Returns:
        {
            "status": "CLEAR" or "REVIEW RECOMMENDED",
            "warning_count": int,
            "warnings": [...]
        }
    """

    if not job_description:
        return {
            "status": "CLEAR",
            "warning_count": 0,
            "warnings": [],
        }

    text = job_description.lower()
    warnings = []

    for phrase, explanation in FAIRNESS_PATTERNS.items():
        pattern = r"\b" + re.escape(phrase) + r"\b"

        if re.search(pattern, text):
            warnings.append({
                "phrase": phrase,
                "explanation": explanation,
            })

    status = (
        "REVIEW RECOMMENDED"
        if warnings
        else "CLEAR"
    )

    return {
        "status": status,
        "warning_count": len(warnings),
        "warnings": warnings,
    }
