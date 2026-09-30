"""Route questions using their text alone; no predictions or labels are read."""

import re


PATTERNS = (
    ("direction_binding", r"방향|어느\s*쪽|가려면"),
    ("attribute_binding", r"(?:[0-9]+\s*층|몇\s*층|시즌|승리|승수|몇\s*승|승패|가격|금액|얼마|[0-9,]+\s*원|몇\s*번.*출구|출구.*몇\s*번)"),
    ("exact_entity", r"이름|상호|회사명|브랜드|명칭|저자|첫\s*단어"),
)


def route_question(question: str) -> str:
    text = str(question)
    for route, pattern in PATTERNS:
        if re.search(pattern, text):
            return route
    return "unchanged"
