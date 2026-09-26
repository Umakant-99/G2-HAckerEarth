"""Score one submission from the command line.

Cached fixture texts are scored from stored vectors. Any other text is embedded
with Gemini and then scored against the cached reference and corpus.

    python -m novelty "Your submission text here"
"""

from __future__ import annotations

import argparse
import json

from novelty.service import score_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a submission for relevance and novelty.")
    parser.add_argument("text", help="Submission text, at most 100 words.")
    parser.add_argument("--id", default="submission")
    parser.add_argument("--topic", default="adhoc")
    parser.add_argument(
        "--reduction",
        choices=("max", "top3_mean"),
        default="max",
        help="Nearest-neighbor reduction. Default is max.",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            score_text(args.text, topic=args.topic, submission_id=args.id, reduction=args.reduction),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
