import json

from ranking_engine import (
    rank_candidates,
    print_ranking_table,
    get_top_candidates
)

from src.explainability import (
    explain_candidate,
    compare_candidates,
    check_jd_fairness
)


# ============================================================
# LOAD PARSED RESUMES
# ============================================================

JSON_FILE = "data/output/resumes.json"

with open(JSON_FILE, "r", encoding="utf-8") as file:
    resumes = json.load(file)

resumes = [
    resume
    for resume in resumes
    if resume.get("parsing_status") == "success"
]

print(f"\nLoaded {len(resumes)} resumes successfully.")


# ============================================================
# JOB DESCRIPTION
# ============================================================

jd_text = """
We are looking for a Full Stack Developer Intern.

Required Skills:
React
Node.js
Express
MongoDB
REST API
Git

Preferred Skills:
TypeScript
Docker
AWS

The candidate should have experience developing web
applications, backend REST services, APIs and scalable
full stack applications.
"""


# ============================================================
# FAIRNESS CHECK
# ============================================================

print("\n")
print("=" * 70)
print("JOB DESCRIPTION FAIRNESS CHECK")
print("=" * 70)

fairness_result = check_jd_fairness(jd_text)

print(f"\nStatus: {fairness_result['status']}")
print(f"Warnings: {fairness_result['warning_count']}")

if fairness_result["warnings"]:

    for warning in fairness_result["warnings"]:
        print(f"\n⚠ {warning['phrase']}")
        print(f"  {warning['explanation']}")

else:
    print("\nNo potentially biased wording detected.")


# ============================================================
# RUN RANKING ENGINE
# ============================================================

results = rank_candidates(
    jd_text,
    resumes
)


# ============================================================
# COMPLETE RANKING
# ============================================================

print_ranking_table(results)


# ============================================================
# TOP 3
# ============================================================

top_candidates = get_top_candidates(
    results,
    3
)

print("\nTOP 3 CANDIDATES")
print("-" * 50)

for candidate in top_candidates:

    print(
        f"#{candidate['rank']} "
        f"{candidate['name']} "
        f"-> {candidate['final_score']:.2f}"
    )


# ============================================================
# PERSON 4 EXPLANATIONS
# ============================================================

print("\n")
print("=" * 70)
print("TOP 3 CANDIDATE EXPLANATIONS")
print("=" * 70)


explanations = []

for candidate in top_candidates:

    # Adapter: ranking_engine format -> explainability format
    explainable_candidate = {
        "candidate_id": candidate["name"],
        "final_score": candidate["final_score"],
        "semantic_score": candidate["semantic_score"],
        "keyword_score": candidate["keyword_score"],
        "required_skill_score": candidate["required_skill_score"],
        "matched_required": candidate.get(
            "matched_required_skills", []
        ),
        "missing_required": candidate.get(
            "missing_required_skills", []
        ),
        "matched_preferred": candidate.get(
            "matched_preferred_skills", []
        ),
        "missing_preferred": []
    }

    explanation = explain_candidate(
        explainable_candidate
    )

    explanations.append(explanation)

    print("\n")
    print("-" * 70)
    print(f"RANK #{candidate['rank']} - {candidate['name']}")
    print("-" * 70)

    print(
        f"\nFinal Score: "
        f"{explanation['final_score']:.2f}%"
    )

    print(
        f"Semantic Match: "
        f"{explanation['semantic_score']:.2f}%"
    )

    print(
        f"Keyword Match: "
        f"{explanation['keyword_score']:.2f}%"
    )

    print(
        f"Required Skills: "
        f"{explanation['required_skill_score']:.2f}%"
    )

    print("\nOverall:")
    print(explanation["overall"])

    print("\nStrengths:")

    for strength in explanation["strengths"]:
        print(f"  + {strength}")

    print("\nConcerns:")

    if explanation["concerns"]:

        for concern in explanation["concerns"]:
            print(f"  - {concern}")

    else:
        print("  None")


# ============================================================
# CANDIDATE COMPARISON
# ============================================================

if len(results) >= 2:

    candidate_a = results[0]
    candidate_b = results[1]

    comparison_a = {
        "candidate_id": candidate_a["name"],
        "final_score": candidate_a["final_score"],
        "semantic_score": candidate_a["semantic_score"],
        "keyword_score": candidate_a["keyword_score"],
        "required_skill_score": candidate_a[
            "required_skill_score"
        ],
        "matched_required": candidate_a.get(
            "matched_required_skills", []
        )
    }

    comparison_b = {
        "candidate_id": candidate_b["name"],
        "final_score": candidate_b["final_score"],
        "semantic_score": candidate_b["semantic_score"],
        "keyword_score": candidate_b["keyword_score"],
        "required_skill_score": candidate_b[
            "required_skill_score"
        ],
        "matched_required": candidate_b.get(
            "matched_required_skills", []
        )
    }

    comparison = compare_candidates(
        comparison_a,
        comparison_b
    )

    print("\n")
    print("=" * 70)
    print("WHY IS CANDIDATE #1 RANKED ABOVE CANDIDATE #2?")
    print("=" * 70)

    print(f"\n{comparison['summary']}")

    print("\nReasons:")

    for reason in comparison["reasons"]:

        print(
            f"  • {reason['category']}: "
            f"{reason['text']}"
        )

    print("=" * 70)
