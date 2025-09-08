import re
PROFANITY = set([
    "damn","hell","stupid","idiot","hate","sucks","crap","shit","fuck","fucking","worst","garbage","trash","awful",
    "terrible","disgusting","pathetic","useless","pointless"
])


def estimate_aggressiveness(text: str) -> int: #empirical weighting 
    t = text or ""
    exclam = t.count("!")
    caps = sum(1 for c in t if c.isupper())
    letters = sum(1 for c in t if c.isalpha())
    caps_ratio = (caps / letters) if letters else 0.0
    words = re.findall(r"[a-zA-Z']+", t.lower())
    prof_hits = sum(1 for w in words if w in PROFANITY)

    # weighted score in [1..10]
    score = 1
    score += min(5, prof_hits * 2)
    score += min(3, exclam // 2)
    score += 2 if caps_ratio > 0.25 else 0
    return int(max(1, min(10, score)))
