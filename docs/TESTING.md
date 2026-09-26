# Testing Guide — Novelty Scoring

How to verify the application locally. Scores are always computed at run time from embeddings. The fixture stores texts and vectors only.

## Prerequisites

```bash
cd "/Users/umakant/Rewarding Novelty in Submissions"
source .venv/bin/activate
pip install -r requirements.txt
```

Optional for new text (not already in the fixture):

```bash
export GEMINI_API_KEY="your-key"
```

Do not put the key in source files.

Confirm the fixture exists:

```bash
ls tests/fixtures/corpus.json
```

If it is missing:

```bash
export GEMINI_API_KEY="your-key"
python -m novelty.synth_data
```

---

## 1. Automated tests

```bash
pytest -v
```

Expected: **5 passed**.

| Test | What it proves |
| --- | --- |
| `test_final_score_is_the_product_of_the_two_signals` | `final_score` equals `max(relevance, 0) * novelty` for every case |
| `test_novel_and_relevant_outranks_the_other_cases` | Novel + relevant scores higher than paraphrase and off-topic |
| `test_non_novel_is_closer_to_the_corpus` | Paraphrase has lower novelty than the original take |
| `test_irrelevant_case_is_less_relevant` | Off-topic case has lower relevance than the on-topic take |
| `test_paraphrase_matches_its_source_more_than_the_original_take` | Paraphrase embedding is closer to its source than the novel take is |

These tests load `tests/fixtures/corpus.json` and call `score_vectors`. They do not call the network and do not assert pre-written score numbers.

---

## 2. Command-line scoring

Score a sentence that is already in the fixture (offline):

```bash
python -m novelty "Before my 12-hour shift at the hospital begins, I climb up to my paint-bucket containers in the dark. Plucking fresh mint under the glow of the skyline, I slip the leaves into my pocket. Slipping them into my water bottle later keeps me awake through the longest nights."
```

Expected shape of the JSON:

```json
{
  "id": "submission",
  "topic": "adhoc",
  "embedding_model": "gemini-embedding-001",
  "relevance": 0.641,
  "novelty": 0.0,
  "final_score": 0.0
}
```

Novelty near `0` and final score near `0` is correct: the text matches a prior submission.

Score a new sentence (needs `GEMINI_API_KEY`):

```bash
python -m novelty "An infrared scan of the block shows the planted roof radiating less heat than the bare membranes next door."
```

Expect non-zero relevance and novelty, both computed from a fresh embedding.

Reject over-long text:

```bash
python -m novelty "$(python -c 'print("word "*101)')"
```

Expected: `ValueError` that the text is over 100 words.

---

## 3. Local web UI

```bash
python -m novelty.server
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

### Checklist

1. The page shows the reference passage and “50 prior submissions” with the embedding model name.
2. Paste a corpus duplicate → Score → novelty near 0, final score near 0.
3. Paste an on-topic original angle → Score → non-zero relevance and novelty; final score is their product.
4. Paste an off-topic sentence (math, sports, recipes) → Score → relevance lower than an on-topic take; final score stays modest.
5. Submit more than 100 words → error message about the word limit.
6. Toggle reduction from `max` to `top3_mean` and score the same text → novelty may change; relevance should not.

### API smoke checks

```bash
curl -s http://127.0.0.1:8000/api/context | python -m json.tool
curl -s -X POST http://127.0.0.1:8000/api/score \
  -H 'Content-Type: application/json' \
  -d '{"text":"Soil and plants cool the rooms under a planted membrane.","topic":"infra","reduction":"max"}' \
  | python -m json.tool
```

---

## 4. Regenerating synthetic data

Only when you intentionally want new cases or embeddings:

```bash
export GEMINI_API_KEY="your-key"
python -m novelty.synth_data --refresh
pytest -v
```

After refresh, re-check that ordering still holds: novel+relevant > paraphrase and > off-topic.

---

## 5. Pass / fail summary

| Area | Pass condition |
| --- | --- |
| `pytest` | 5 passed |
| Duplicate submission | `novelty ≈ 0`, `final_score ≈ 0` |
| Product rule | `final_score ≈ max(relevance, 0) * novelty` |
| Word limit | >100 words rejected |
| UI | Reference loads; Score returns live numbers |
| Fixture | No stored `final_score` fields in `cases` |

## Known model limits

This API key does not serve `text-embedding-004` or `gemini-2.5-flash`. Embeddings use `gemini-embedding-001`. Generation uses `gemini-3.5-flash`. Absolute cutoffs such as “score > 0.5” from the design notes may not hold on that embedding scale; relative ordering is the reliable check.
