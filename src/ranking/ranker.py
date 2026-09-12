import json
import os

from .semantic_match import semantic_match
from .keyword_match import keyword_match


def calculate_experience_score(resume):
    """
    Estimate experience using the experience section.

    This is intentionally lightweight for the hackathon.
    """

    experience = resume.get("sections", {}).get("experience", "")

    if not experience:
        return 0.0

    text = experience.lower()

    # Internship / work experience indicators
    indicators = [
        "intern",
        "internship",
        "developer",
        "engineer",
        "software",
        "worked",
        "experience"
    ]

    matches = sum(1 for word in indicators if word in text)

    score = min(matches * 15, 100)

    return round(score, 2)


def rank_candidates(resumes, job_description, required_skills=None):
    """
    Rank all candidates against a job description.
    """

    if required_skills is None:
        required_skills = []

    ranked = []

    for resume in resumes:

        semantic_score = semantic_match(
            resume,
            job_description
        )

        keyword_score = keyword_match(
            resume,
            required_skills
        )

        experience_score = calculate_experience_score(
            resume
        )

        # Required skills score
        required_score = keyword_score

        # Final weighted score
        final_score = (
            semantic_score * 0.40
            + required_score * 0.30
            + keyword_score * 0.15
            + experience_score * 0.15
        )

        candidate = {
            **resume,

            "scores": {
                "semantic": round(semantic_score, 2),
                "required": round(required_score, 2),
                "keyword": round(keyword_score, 2),
                "experience": round(experience_score, 2),
                "final": round(final_score, 2)
            }
        }

        ranked.append(candidate)

    # Highest score first
    ranked.sort(
        key=lambda candidate: candidate["scores"]["final"],
        reverse=True
    )

    # Assign ranks
    for index, candidate in enumerate(ranked, start=1):
        candidate["rank"] = index

    return ranked


def load_resumes(path="data/output/resumes.json"):
    """Load parsed resumes."""

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Resume data not found: {path}"
        )

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


if __name__ == "__main__":

    resumes = load_resumes()

    job_description = """
    Software Engineer with experience in Python,
    Machine Learning, REST API, SQL and AWS.
    """

    required_skills = [
        "Python",
        "Machine Learning",
        "REST API",
        "SQL",
        "AWS"
    ]

    ranked = rank_candidates(
        resumes,
        job_description,
        required_skills
    )

    print("\n")
    print("=" * 100)
    print("NEXORA - CANDIDATE RANKING")
    print("=" * 100)

    print(
        f"{'Rank':<6}"
        f"{'Candidate':<25}"
        f"{'Semantic':<12}"
        f"{'Required':<12}"
        f"{'Keyword':<12}"
        f"{'Experience':<12}"
        f"{'FINAL':<10}"
    )

    print("-" * 100)

    for candidate in ranked:

        scores = candidate["scores"]

        print(
            f"{candidate['rank']:<6}"
            f"{candidate.get('name', 'Unknown'):<25}"
            f"{scores['semantic']:<12.2f}"
            f"{scores['required']:<12.2f}"
            f"{scores['keyword']:<12.2f}"
            f"{scores['experience']:<12.2f}"
            f"{scores['final']:<10.2f}"
        )

    print("=" * 100)

    print("\nTOP 3 CANDIDATES\n")

    for candidate in ranked[:3]:
        print(
            f"{candidate['rank']}. "
            f"{candidate.get('name', 'Unknown')} "
            f"- {candidate['scores']['final']:.2f}"
        )
