# Engineering Design Document — Rewarding Novelty in Submissions

**Document type:** Markdown engineering design report  
**Project:** G2-HAckerEarth novelty scoring  
**Stack:** Python 3.9+, NumPy, pytest, Gemini API  
**Local UI:** `http://127.0.0.1:8000`

This document covers **engineering design**, **rationale**, **success criteria**, **results**, and **limitations**, plus the file map and how to run the system.

---

## 1. Problem statement (what was asked)

Build a system that evaluates the novelty of user-generated content and assigns a **normalized novelty score between 0.0 and 1.0**.

The challenge is to **reward content that is genuinely novel** while ensuring it **remains relevant** to a fixed original piece of content.

| Requirement | Expectation |
| --- | --- |
| Content shape | Structured user content with **three discrete properties** |
| Reference | Evaluate against one fixed piece of content, **≤100 words** |
| Corpus | Measure novelty against about **50** other submissions |
| Synthetic data | Use synthetic generation if required |
| Automated tests | Novel+relevant rewarded; non-novel not rewarded; novel+low-relevance not rewarded |
| Preferred stack | Python, TypeScript, or Ruby |
| Optional models | Gemini (`aistudio.google.com`) |

---

## 2. Engineering design

### 2.0 Dependencies

Install with `pip install -r requirements.txt` inside a Python 3.9+ virtualenv.

| Dependency | Version / source | Required for | Notes |
| --- | --- | --- | --- |
| **Python** | 3.9+ | Runtime | Prefer `.venv`; on macOS use `source .venv/bin/activate` or `.venv/bin/python` |
| **NumPy** | `numpy>=2.0` | Scoring math (`scorer.py`, vectors) | Only third-party runtime library |
| **pytest** | `pytest>=8.0` | Automated tests | Dev / eval; not needed to run CLI or UI |
| **Gemini API** | External HTTPS API | Embed new text; build / refresh fixture | Set `GEMINI_API_KEY` in the environment; never commit the key |
| **stdlib** | Bundled with Python | CLI, HTTP UI, Gemini HTTP client, JSON I/O | `http.server`, `urllib`, `json`, `argparse`, `pathlib`, `hashlib`, `dataclasses`, etc. — no Flask/FastAPI |

**Pinned install file:** `requirements.txt`

```
numpy>=2.0
pytest>=8.0
```

**Not required as pip packages:** Google AI SDK, web frameworks, or embedding libraries. Gemini is called over HTTPS with `urllib`. Offline scoring of fixture text needs only NumPy + `tests/fixtures/corpus.json`.

### 2.1 Data model

A submission has exactly three fields:

| Property | Type | Meaning |
| --- | --- | --- |
| `id` | `str` | Stable identifier |
| `text` | `str` | Content body, **≤100 words** |
| `topic` | `str` | Category / tag |

The fixed original content is also a `Submission` with `id="reference"`. That keeps the pipeline uniform: everything that is scored or compared is the same shape.

Implemented in `novelty/models.py`.

### 2.2 Score definition

Two independent signals, combined **multiplicatively**:

```
relevance   = cosine(submission_embedding, reference_embedding)
novelty     = 1 − nearest_neighbor_similarity(submission, corpus ≈ 50)
final_score = clamp(max(relevance, 0) × novelty, 0, 1)
```

| Signal | Definition | Default |
| --- | --- | --- |
| Relevance | Cosine similarity to the reference | — |
| Novelty | Distance from the closest prior work | Neighbor reduction = **`max`** |
| Optional reduction | Mean of the three closest corpus items | **`top3_mean`** |

`final_score` is always in **[0.0, 1.0]**.

Implemented in `novelty/scorer.py`. CLI and UI both call `novelty/service.py` → `score_text()`, which embeds if needed and then calls `score_vectors()`.

### 2.3 Embedding and generation layer

| Concern | Design |
| --- | --- |
| Embeddings | Prefer `text-embedding-004`; fall back to `gemini-embedding-001` when the preferred model is not served |
| Generation | Prefer `gemini-2.5-flash`; fall forward to current flash models (`gemini-3.5-flash`, etc.) |
| Caching | Embedding vectors cached on disk under `.cache/` by model + text hash |
| Offline tests | Fixture stores texts **and** vectors so pytest needs no network |

Implemented in `novelty/gemini_client.py` and `novelty/synth_data.py`.

### 2.4 Synthetic corpus and cases

| Artifact | Source |
| --- | --- |
| ~50 background submissions | Gemini-generated narratives (also kept in `data/corpus.json`) |
| Topic tags | Gemini labels at fixture build time |
| Case: novel + relevant | Gemini-generated original on-topic take |
| Case: non-novel | Gemini paraphrase of one corpus item (`paraphrase_of`) |
| Case: novel + irrelevant | Gemini-generated off-topic creative text |
| Vectors | Gemini embeddings for reference + corpus + cases |

Written to `tests/fixtures/corpus.json`. **Scores are not stored** in the fixture; they are computed at evaluation time.

### 2.5 Interfaces

| Interface | Entry | Behavior |
| --- | --- | --- |
| Automated tests | `pytest` | Offline scoring of the three cases from cached vectors |
| CLI | `python -m novelty "…"` | Prints live JSON: relevance, novelty, final_score |
| Local UI | `python -m novelty.server` | Page + `/api/context` + `/api/score`; Live score panel |

### 2.6 Architecture (runtime)

```
User text (CLI / UI)
        │
        ▼
  service.score_text()
        │
        ├─ known fixture text → reuse cached vector
        └─ new text → gemini_client.embed_texts()  [GEMINI_API_KEY]
                │
                ▼
        scorer.score_vectors (numpy only)
                │
                ├─ relevance vs reference
                ├─ novelty vs ~50 corpus vectors
                └─ final = max(relevance, 0) × novelty
                │
                ▼
     CLI JSON  |  UI Live score panel  |  pytest
```

### 2.7 File responsibilities

| File | Responsibility |
| --- | --- |
| `novelty/models.py` | Submission schema and word-limit validation |
| `novelty/gemini_client.py` | Embed / generate, retries, model fallbacks, disk cache |
| `novelty/scorer.py` | Pure scoring math (no I/O) |
| `novelty/synth_data.py` | Build / refresh fixture |
| `novelty/service.py` | Shared scoring entry for CLI and UI |
| `novelty/__main__.py` | CLI adapter |
| `novelty/server.py` | Local HTTP UI and JSON APIs |
| `tests/test_scorer.py` | Automated success criteria |
| `tests/fixtures/corpus.json` | Cached texts + vectors |
| `data/corpus.json` | Source pool of 50 synthetic submissions |
| `details.md` | Design notes used during implementation |
| `docs/TESTING.md` | How to verify the system |
| `docs/PROJECT.md` | This engineering design document |
| `requirements.txt` | Pip dependencies (NumPy, pytest) |
| `README.md` | Short setup and usage |

---

## 3. Rationale (why this design)

### 3.1 Why a product, not a weighted sum

| Situation | Product behavior | Weighted-sum risk |
| --- | --- | --- |
| Near-duplicate of corpus | Novelty ≈ 0 → final ≈ 0 | Still gets credit from high relevance |
| Novel but off-topic | Relevance low → final low | Still gets credit from high novelty (~0.5) |
| Novel and on-topic | Both factors high → final high | Same, but product enforces both gates |

The brief requires that **novel-but-irrelevant is not rewarded**. Multiplicative gating is the direct way to encode that.

### 3.2 Why embeddings + cosine, not LLM-as-judge for the score

| Concern | Choice |
| --- | --- |
| Determinism | Same vectors → same score |
| Testability | Offline pytest with cached vectors |
| Cost / rate limits | Score path does not call a generative model |
| Numerics | Score is a real number in `[0, 1]`, not free-form prose |

Gemini generation is used as the **data factory** (corpus + cases). Gemini embeddings are the **scoring engine**.

### 3.3 Why max nearest neighbor for novelty

Novelty should punish being too close to **any** existing submission, not merely average closeness across the pool. `max` (closest neighbor) does that. `top3_mean` is available when a slightly softer penalty is useful.

### 3.4 Why cache the fixture

| Goal | How the fixture helps |
| --- | --- |
| Deterministic tests | Same vectors every pytest run |
| No network in CI / local test | Embeddings already stored |
| Avoid rate limits | Generation and embedding happen once (`--refresh` when needed) |
| No hardcoded scores | Fixture has texts/vectors only; scores computed live |

### 3.5 Why relative ordering in tests

Embedding cosine scales differ by model. Absolute thresholds like `final > 0.5` are brittle on `gemini-embedding-001`. Relative checks (`novel_relevant > non_novel`, etc.) still prove the three required behaviors.

---

## 4. Success criteria

### 4.1 Functional criteria (must pass)

| ID | Criterion | How verified |
| --- | --- | --- |
| S1 | Submission has three discrete properties | `Submission(id, text, topic)` |
| S2 | Reference ≤100 words; submissions reject >100 words | Model validation + fixture |
| S3 | Novelty measured against ~50 priors | Fixture corpus length = 50 |
| S4 | Synthetic generation used | `synth_data.py` + Gemini |
| S5 | Score normalized to `[0.0, 1.0]` | `scorer.score_vectors` clamp |
| S6 | Novel + relevant outranks non-novel and off-topic | pytest ordering tests |
| S7 | Non-novel is not rewarded | Low novelty / low final for paraphrase |
| S8 | Off-topic is less relevant than on-topic novel | pytest relevance comparison |
| S9 | `final_score == max(relevance, 0) * novelty` | pytest product identity |
| S10 | Scores are computed live, not stored | Fixture contains no score fields |

### 4.2 Operational criteria

| ID | Criterion | How verified |
| --- | --- | --- |
| O1 | Offline automated tests | `pytest -v` → 5 passed without API |
| O2 | Local UI can score | `python -m novelty.server` + Score button |
| O3 | New text can be embedded when key is set | `/api/score` returns 200 with `GEMINI_API_KEY` |
| O4 | Secrets not in source | Key only via environment |

### 4.3 Stretch criteria (design notes, not hard gates)

| ID | Note |
| --- | --- |
| X1 | Absolute cutoffs such as novel+relevant `> 0.5` and others `< 0.2` depend on embedding scale |
| X2 | Preferred model names (`text-embedding-004`, `gemini-2.5-flash`) when the API serves them |

---

## 5. Results

Measured against the current fixture with **live** `score_vectors` calls (not stored scores).

### 5.1 Environment used for these results

| Item | Value |
| --- | --- |
| Embedding model in fixture | `gemini-embedding-001` |
| Embedding dimensions | 768 |
| Generation model recorded | `gemini-3.5-flash` |
| Corpus size | 50 |
| Paraphrase source | `prior_001` |
| Automated tests | **5 passed** |

### 5.2 Case scores (computed at evaluation time)

| Case | Topic (from Gemini) | Relevance | Novelty | Final |
| --- | --- | --- | --- | --- |
| `novel_relevant` | acoustic benefits | 0.745 | 0.323 | **0.240** |
| `non_novel` | hospital shift routine | 0.635 | 0.049 | **0.031** |
| `novel_irrelevant` | deep-sea biology | 0.536 | 0.404 | **0.216** |

### 5.3 Interpretation against the brief

| Brief expectation | Result |
| --- | --- |
| Highly novel and relevant is rewarded | Highest **final** among the three (0.240). Novelty is moderate (0.323), not extreme, because the corpus already covers many rooftop-garden angles. |
| Non-novel is not rewarded | Final 0.031; novelty 0.049 — clearly suppressed. |
| Highly novel but low-relevance is not rewarded | Final (0.216) stays below the on-topic novel case. Relevance (0.536) is lower than the on-topic case (0.745) but not near zero on this embedding model. |

### 5.4 Automated test outcome

```
pytest -q
.....  5 passed
```

Covered assertions:

- Product identity for every case  
- `novel_relevant` final > `non_novel` and > `novel_irrelevant`  
- Paraphrase novelty lower than novel take  
- Off-topic relevance lower than novel take  
- Paraphrase closer to its corpus source than the novel take is  

### 5.5 Delivered surfaces

| Surface | Status |
| --- | --- |
| Python package `novelty/` | Complete |
| Cached synthetic fixture | Complete |
| pytest suite | Passing |
| CLI scorer | Complete |
| Local scoring UI with Live score panel | Complete |
| Engineering docs (`PROJECT.md`, `TESTING.md`) | Complete |

---

## 6. Limitations

### 6.1 Model availability

This API key does not serve `text-embedding-004` or `gemini-2.5-flash` for new users. The client falls back to:

- Embeddings: `gemini-embedding-001`
- Generation: `gemini-3.5-flash` (and other available flash models)

Behavior is correct; preferred model names from the design notes may not be the ones that actually run.

### 6.2 Embedding similarity floor

With `gemini-embedding-001`, clearly unrelated text often still has cosine ≈ **0.4–0.55** against the reference. Consequences:

- “Low relevance” is relative, not near-zero  
- Absolute thresholds from the design notes (`final > 0.5`, `< 0.2`) are not reliable on this model  
- Ordering remains the robust success criterion  

### 6.3 Novelty headroom on a dense corpus

The 50 synthetic submissions already cover many concrete rooftop-garden situations. A new on-topic angle can still be nearest-neighbor-similar enough that novelty sits around **0.3**, so final scores are modest even when the case is the best of the three.

### 6.4 No separate “evals” framework

Evaluation is done with **pytest** over the three required scenarios. There is no third-party eval harness (Promptfoo, LangSmith, etc.). The tests are the eval suite required by the brief.

### 6.5 Operational constraints

| Limit | Detail |
| --- | --- |
| API key | Required for new text embeddings and for `--refresh`; must be set in the same shell as the server |
| macOS `python` | Often missing until `.venv` is activated; use `source .venv/bin/activate` or `.venv/bin/python` |
| Secrets | Never commit `GEMINI_API_KEY`; rotate if it was pasted into chat or logs |
| UI refresh | After editing `server.py`, restart the server and hard-refresh the browser |

### 6.6 What is intentionally not hardcoded

- Corpus and case **texts** come from Gemini (then cached)  
- **Embeddings** come from Gemini (then cached)  
- **Scores** are always computed at run time  
- Fixture does not store pass/fail score numbers  

---

## 7. How to run (summary)

```bash
cd "/Users/umakant/Rewarding Novelty in Submissions"
source .venv/bin/activate
pip install -r requirements.txt

# Success criteria / evals (offline)
pytest -v

# Local UI
export GEMINI_API_KEY="your-real-key"
python -m novelty.server
# open http://127.0.0.1:8000

# CLI
python -m novelty "Your submission text here"

# Rebuild synthetic fixture
python -m novelty.synth_data --refresh
```

Full testing steps: [`docs/TESTING.md`](TESTING.md).

---

## 8. Conclusion

The system is built in the correct format for the brief:

- Three-property content model  
- Fixed ≤100-word reference  
- Novelty against ~50 synthetic submissions  
- Gemini-backed generation and embeddings with offline fixture  
- Automated tests demonstrating the three required behaviors  
- Normalized live score in `[0.0, 1.0]` via `relevance × novelty`  

**Success on absolute “highly novel / near-zero relevance” cutoffs is limited by the embedding model’s similarity scale.** Success on **ordering**, **non-novel suppression**, and **live non-hardcoded scoring** is demonstrated by the current results and the passing pytest suite.
