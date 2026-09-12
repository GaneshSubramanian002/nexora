def keyword_match(resume, required_skills):
    """
    Calculate how many required skills are present
    in the candidate's extracted skills.
    """

    if not required_skills:
        return 0.0

    candidate_skills = {
        skill.lower()
        for skill in resume.get("skills", [])
    }

    required = {
        skill.lower()
        for skill in required_skills
    }

    matched = candidate_skills.intersection(required)

    score = (len(matched) / len(required)) * 100

    return round(score, 2)
