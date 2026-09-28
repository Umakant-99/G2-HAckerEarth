# G2-HAckerEarth

Scores a submission against a fixed reference and a pool of prior submissions.

```
relevance   = cosine(submission, reference)
novelty     = 1 - max cosine to any prior submission
final_score = max(relevance, 0) * novelty
```

`max` is the default. `top3_mean` averages the three closest prior submissions instead. Either way the score is the product of the two signals, so a near-duplicate or an off-topic submission cannot score well on the other signal alone.

The reference is a `Submission` with `id="reference"`. Every submission has `id`, `text` (at most 100 words), and `topic`.

## Run on another machine

### Requirements

| Need | Detail |
| --- | --- |
| OS | macOS, Linux, or Windows |
| Python | 3.9 or newer (`python3 --version`) |
| Pip packages | `numpy>=2.0`, `pytest>=8.0` (from `requirements.txt`) |
| API key | `GEMINI_API_KEY` — only for **new** text embeddings or rebuilding the fixture |
| Network | Needed for `git clone`, `pip install`, and Gemini calls; not needed for offline `pytest` or scoring text already in the fixture |

The repo already includes `tests/fixtures/corpus.json`, so you do **not** need to regenerate data to run tests or score known texts.

### 1. Clone and install

```bash
git clone https://github.com/Umakant-99/G2-HAckerEarth.git
cd G2-HAckerEarth

python3 -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows (PowerShell)
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### 2. Set the API key (optional for offline checks)

```bash
# macOS / Linux
export GEMINI_API_KEY="your-key"

# Windows (PowerShell)
# $env:GEMINI_API_KEY="your-key"
```

Or create a local `.env` file (gitignored) with `GEMINI_API_KEY=your-key`, then load it in the same shell before starting the server:

```bash
set -a && source .env && set +a   # macOS / Linux
```

Do not commit the key or put it in source files.

### 3. Verify (offline — no API key)

```bash
pytest -v
# expect: 5 passed
```

### 4. Run the local UI

```bash
python -m novelty.server
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

- Text already in the fixture scores offline from cached vectors.
- New text needs `GEMINI_API_KEY` in the **same shell** that started the server.

### 5. Run the CLI

```bash
python -m novelty "Your submission text here"
```

### Quick checks

| Goal | Command |
| --- | --- |
| Automated tests | `pytest -v` |
| Web UI | `python -m novelty.server` → http://127.0.0.1:8000 |
| One-off score | `python -m novelty "…"` |
| Rebuild fixture | `export GEMINI_API_KEY=…` then `python -m novelty.synth_data --refresh` |

More detail: [`docs/TESTING.md`](docs/TESTING.md), [`docs/PROJECT.md`](docs/PROJECT.md).

## Setup (short)

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
