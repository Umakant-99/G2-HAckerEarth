# G2-HAckerEarth

Scores a submission against a fixed reference and a pool of prior submissions.

```
relevance   = cosine(submission, reference)
novelty     = 1 - max cosine to any prior submission
final_score = max(relevance, 0) * novelty
```

`max` is the default. `top3_mean` averages the three closest prior submissions instead. Either way the score is the product of the two signals, so a near-duplicate or an off-topic submission cannot score well on the other signal alone.

The reference is a `Submission` with `id="reference"`. Every submission has `id`, `text` (at most 100 words), and `topic`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export GEMINI_API_KEY="your-key"
```

Do not put the key in source files.

## Data

`python -m novelty.synth_data` writes `tests/fixtures/corpus.json`.

- The 50 background submissions are the Gemini texts already stored in `data/corpus.json`. Topic tags for those texts are requested from Gemini when the fixture is built.
- The three cases (novel and relevant, close paraphrase, novel and off-topic) are generated in that same run.
- Embeddings use `text-embedding-004` when that model is served, otherwise `gemini-embedding-001`. Vectors are cached. The fixture does not contain scores.

Generation tries `gemini-2.5-flash` first. This key cannot call that model, so the client continues with `gemini-3.5-flash`.

`--refresh` rebuilds the cases and embeddings.

## Score a submission

```bash
python -m novelty.server
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). The page loads the reference from the fixture. Scoring posts the text you type to `/api/score`, which runs the same function as the command line. A new sentence needs `GEMINI_API_KEY` so it can be embedded. Text already in the fixture is scored from the cached vector.

```bash
python -m novelty "Your submission text here"
pytest
```

`python -m novelty` prints `relevance`, `novelty`, and `final_score` from that call. `pytest` loads the fixture and computes the same function. Neither path prints a stored score.
