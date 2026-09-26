Good — that's exactly the stack you need. Here's how I'd structure the build, calling out where Gemini plugs in.

1. Data model (3 discrete properties)
python
@dataclass
class Submission:
    id: str
    text: str          # ≤100 words, the actual content
    topic: str          # category/tag, e.g. "product-idea", "slogan", "story"

Keep it minimal and generic so the same pipeline works whatever the "content" domain actually is (hackathon pitches, marketing taglines, story premises, etc). The reference content is just a Submission with id="reference".

2. Core scoring idea

Two independent signals, combined multiplicatively so neither can be gamed alone:

final_score = relevance(sub, reference) * novelty(sub, corpus)
Relevance = cosine similarity between sub embedding and reference embedding (Gemini text-embedding-004, or the newer gemini-embedding-001).
Novelty = 1 - max_similarity_to_corpus, where you compare against the ~50 other submissions. Use max (or mean of top-3) rather than average — you want to punish being too close to any one existing submission, not just close on average.

Why multiplicative, not weighted sum:

Non-novel (near-duplicate of corpus): novelty≈0 → score≈0 regardless of relevance. ✅
Novel but irrelevant (random tangent): relevance≈0 → score≈0 regardless of novelty. ✅
Novel AND relevant: both high → score high. ✅

A weighted sum (0.5*rel + 0.5*nov) would let a wildly irrelevant-but-novel submission still score ~0.5, which fails your third test case. Multiplicative gating is the key design decision here.

3. Where Gemini comes in

Two uses, pick one or both:

A. Embeddings (cheap, deterministic, scalable) — call gemini-embedding-001 once per submission, cache vectors, do cosine similarity in numpy. This is the main scoring engine — fast, no rate-limit worries for 50 items.

B. LLM-as-judge (optional second opinion) — prompt Gemini (gemini-2.5-flash) with the reference + submission and ask for a structured JSON verdict: {"relevance": 0-1, "reasoning": "..."}. Useful as:

a sanity-check / calibration layer on top of embedding scores
the generator of your synthetic test corpus (see below)

I'd make embeddings the scoring mechanism (numeric, testable, cheap) and Gemini generation the data factory for synthetic submissions.

4. Synthetic data generation

Use Gemini to generate three batches:

~50 "background corpus" submissions — plausible, on-topic, varying quality — prompt: "generate 50 short submissions responding to this prompt: {reference}, varying in creativity and phrasing."
A "novel + relevant" test case — prompt Gemini explicitly: "give a genuinely original take on {reference} that no one else would think of, but stays on-topic."
A "non-novel" test case — near-paraphrase of an existing corpus item (or just reuse one from the corpus).
A "novel but irrelevant" test case — prompt: "write something creative and original that has nothing to do with {reference}."

Cache all of this to a JSON fixture file after first generation, so tests don't re-hit the API every run (flaky/rate-limited/non-deterministic otherwise).

5. Project layout
novelty/
  models.py        # Submission dataclass
  gemini_client.py # embed_text(), generate_text() wrappers w/ retry
  scorer.py        # relevance(), novelty(), score()
  synth_data.py     # one-time generation script → fixtures/corpus.json
tests/
  fixtures/corpus.json
  test_scorer.py    # the 3 required scenarios
6. Test thresholds

Pick thresholds empirically after first run (embeddings won't give clean 0/1), something like:

novel+relevant → score > 0.5
non-novel → score < 0.2
novel+irrelevant → score < 0.2

Assert relative ordering too (score(novel_relevant) > score(non_novel) and > score(novel_irrelevant)) — more robust than fixed thresholds since embedding similarity scales vary by model.

7. Build order
Models + Gemini client wrapper (embed + generate) with local disk caching.
synth_data.py → generate & cache the 50-item corpus + 3 test submissions once.
scorer.py → relevance/novelty/score functions, pure numpy, fully unit-testable without network calls once vectors are cached.
tests/test_scorer.py → the three scenarios, loading cached fixtures (fast, no live API needed in CI).