"""Build the cached corpus and the three scoring cases.

The 50 background submissions are taken from the Gemini cache in
``data/corpus.json`` when that file exists, so a rebuild does not spend
another 50 generations. The three test submissions are generated here:

- novel and relevant
- a close paraphrase of one corpus item
- novel and off-topic

Embeddings for every text are stored on the fixture, so tests score from
disk.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from novelty.gemini_client import active_embed_model, active_generate_model, embed_texts, generate_text
from novelty.models import Submission


ROOT = Path(__file__).resolve().parents[1]
LEGACY_CORPUS_PATH = ROOT / "data" / "corpus.json"
FIXTURE_PATH = ROOT / "tests" / "fixtures" / "corpus.json"

REFERENCE_TEXT = (
    "Rooftop gardens in cities cool the rooms under them, soak up rain before it "
    "floods drains, and produce herbs and vegetables close to the kitchens that use "
    "them. They also offer nectar and shelter for bees and other pollinators that "
    "would otherwise pass over a hot roof. A well-kept roof garden turns empty "
    "membrane into useful green infrastructure."
)


def reference_submission() -> Submission:
    return Submission(id="reference", text=REFERENCE_TEXT, topic="prompt")


def load_fixture(path: Path = FIXTURE_PATH) -> dict:
    """Load the corpus fixture, including cached embedding vectors."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Create it with: python -m novelty.synth_data"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def submissions_from_fixture(payload: dict) -> tuple[Submission, list[Submission], dict[str, Submission]]:
    reference = Submission.from_dict(payload["reference"])
    corpus = [Submission.from_dict(item) for item in payload["corpus"]]
    cases = {name: Submission.from_dict(item) for name, item in payload["cases"].items()}
    return reference, corpus, cases


def vectors_from_fixture(payload: dict) -> dict[str, np.ndarray]:
    return {
        item_id: np.asarray(values, dtype=np.float64)
        for item_id, values in payload["embeddings"]["vectors"].items()
    }


def build_fixture(path: Path = FIXTURE_PATH, refresh: bool = False) -> dict:
    """Generate any missing texts, embed them, and write the fixture."""
    if path.exists() and not refresh:
        print(f"using {path}")
        return load_fixture(path)
    reference = reference_submission()
    corpus = _load_or_generate_corpus(reference)
    anchor = corpus[0]
    cases = _generate_cases(reference, anchor)
    ordered = [reference, *corpus, *cases.values()]
    matrix = embed_texts([item.text for item in ordered])
    payload = {
        "reference": reference.to_dict(),
        "corpus": [item.to_dict() for item in corpus],
        "cases": {name: item.to_dict() for name, item in cases.items()},
        "embeddings": {
            "model": active_embed_model(),
            "dimensions": int(matrix.shape[1]),
            "vectors": {
                item.id: [float(value) for value in matrix[index]]
                for index, item in enumerate(ordered)
            },
        },
        "generation_model": active_generate_model(),
        "paraphrase_of": anchor.id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {path} ({len(corpus)} corpus submissions, {len(cases)} cases)")
    return payload


def _load_or_generate_corpus(reference: Submission) -> list[Submission]:
    if LEGACY_CORPUS_PATH.exists():
        legacy = json.loads(LEGACY_CORPUS_PATH.read_text(encoding="utf-8"))
        unlabeled = [
            (item["id"], item["text"])
            for item in legacy["submissions"]
        ]
        if len(unlabeled) < 50:
            raise RuntimeError(f"legacy corpus has {len(unlabeled)} submissions; expected 50")
        return _label_topics(unlabeled[:50])
    return _generate_corpus(reference)


def _label_topics(rows: list[tuple[str, str]]) -> list[Submission]:
    """Ask Gemini for a topic tag per submission. Tags are not filled in locally."""
    listing = "\n".join(f"{item_id}: {text}" for item_id, text in rows)
    prompt = f"""Return a JSON object mapping each submission id to a short topic tag.
Use the submission's own content. Vary the tags. Do not give every id the same tag.

{listing}
"""
    parsed = _parse_json(generate_text(prompt, max_output_tokens=4096))
    if not isinstance(parsed, dict):
        raise RuntimeError("topic labeling did not return a JSON object")
    labeled: list[Submission] = []
    for item_id, text in rows:
        topic = parsed.get(item_id)
        if not isinstance(topic, str) or not topic.strip():
            raise RuntimeError(f"missing topic for {item_id}")
        labeled.append(Submission(id=item_id, text=text, topic=topic.strip()))
    return labeled


def _generate_corpus(reference: Submission) -> list[Submission]:
    prompt = f"""Generate a JSON array of 50 short first-person submissions responding to this prompt.

Prompt:
{reference.text}

Each object has keys id, text, topic.
- id is prior_001 through prior_050.
- text is 40 to 80 words, a concrete story, and stays on the prompt.
- topic is a short tag such as story, harvest, pollinator, or community.
- Vary the situations, phrasing, and level of overlap with the prompt.
- Do not repeat a sentence frame with only a name or number changed.
"""
    raw = generate_text(prompt, max_output_tokens=8192)
    parsed = _parse_json(raw)
    if isinstance(parsed, dict):
        parsed = parsed.get("submissions") or parsed.get("items")
    if not isinstance(parsed, list) or len(parsed) < 50:
        raise RuntimeError("corpus generation did not return 50 submissions")
    return [
        Submission(
            id=f"prior_{index:03d}",
            text=str(item["text"]),
            topic=str(item["topic"]),
        )
        for index, item in enumerate(parsed[:50], start=1)
    ]


def _generate_cases(reference: Submission, anchor: Submission) -> dict[str, Submission]:
    prompt = f"""Write three short submissions as one JSON object with keys novel_relevant, non_novel, and novel_irrelevant.
Each value is an object with keys text and topic.

Reference prompt:
{reference.text}

1. novel_relevant: a genuinely original take on the reference that stays on topic.
   30 to 70 words. Choose a topic tag from the text you write.

2. non_novel: a close paraphrase of the submission below. Keep the same facts
   and actions. Change only wording. 30 to 70 words. Choose a topic tag from the text.
   Submission:
   {anchor.text}

3. novel_irrelevant: something creative and original that has nothing to do with
   the reference. 30 to 70 words. Choose a topic tag from the text.

Return only the JSON object.
"""
    last_error = "no attempt"
    for _attempt in range(3):
        raw = generate_text(prompt, max_output_tokens=2048)
        try:
            parsed = _parse_json(raw)
            return {
                "novel_relevant": _case_from_model("case_novel_relevant", parsed["novel_relevant"]),
                "non_novel": _case_from_model("case_non_novel", parsed["non_novel"]),
                "novel_irrelevant": _case_from_model("case_novel_irrelevant", parsed["novel_irrelevant"]),
            }
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            last_error = str(exc)
    raise RuntimeError(f"could not generate test cases: {last_error}")


def _case_from_model(submission_id: str, payload: dict) -> Submission:
    if not isinstance(payload, dict):
        raise ValueError(f"{submission_id} was not an object")
    topic = payload.get("topic")
    if not isinstance(topic, str) or not topic.strip():
        raise ValueError(f"{submission_id} is missing a topic")
    return Submission(id=submission_id, text=str(payload["text"]), topic=topic.strip())


def _parse_json(raw: str):
    stripped = raw.strip()
    stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
    stripped = re.sub(r"\s*```$", "", stripped)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"(\[.*\]|\{.*\})", stripped, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and cache the novelty fixture.")
    parser.add_argument("--refresh", action="store_true", help="Regenerate cases and embeddings.")
    parser.add_argument("--path", type=Path, default=FIXTURE_PATH)
    args = parser.parse_args()
    build_fixture(args.path, refresh=args.refresh)


if __name__ == "__main__":
    main()
