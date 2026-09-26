"""Submission model.

Three fields only. The reference passage is a Submission whose id is "reference".
"""

from __future__ import annotations

from dataclasses import dataclass, asdict


MAX_WORDS = 100


def word_count(text: str) -> int:
    """Return the number of whitespace-separated words."""
    return len(text.split())


@dataclass(frozen=True)
class Submission:
    """One piece of content to embed and score.

    Attributes:
        id: Stable identifier. The fixed reference uses ``reference``.
        text: The content itself, at most 100 words.
        topic: A category tag such as ``story``, ``slogan``, or ``prompt``.
    """

    id: str
    text: str
    topic: str

    def __post_init__(self) -> None:
        if not str(self.id).strip():
            raise ValueError("id must not be empty")
        if not str(self.topic).strip():
            raise ValueError("topic must not be empty")
        cleaned = self.text.strip()
        count = word_count(cleaned)
        if not cleaned:
            raise ValueError("text must not be empty")
        if count > MAX_WORDS:
            raise ValueError(f"text is {count} words; the limit is {MAX_WORDS}")
        object.__setattr__(self, "id", str(self.id).strip())
        object.__setattr__(self, "topic", str(self.topic).strip())
        object.__setattr__(self, "text", cleaned)

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "Submission":
        return cls(id=payload["id"], text=payload["text"], topic=payload["topic"])
