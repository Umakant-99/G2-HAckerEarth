"""Local page for scoring a submission.

    python -m novelty.server

Then open http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from novelty.gemini_client import GeminiError
from novelty.service import context, score_text


PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Novelty score</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --ink: #0f1a14;
      --panel: #16231c;
      --panel-2: #1d2d24;
      --mist: #e8efe4;
      --mute: #8a9a8e;
      --leaf: #b8d45a;
      --line: #2a3b31;
      --danger: #f0a0a0;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: "IBM Plex Sans", sans-serif;
      background: radial-gradient(1200px 600px at 10% -10%, #24382c 0%, var(--ink) 55%);
      color: var(--mist);
    }
    main {
      max-width: 56rem;
      margin: 0 auto;
      padding: 2.5rem 1.25rem 4rem;
      display: grid;
      gap: 1.5rem;
    }
    @media (min-width: 900px) {
      main { grid-template-columns: 1.4fr 0.9fr; align-items: start; }
      .brand, .reference { grid-column: 1 / -1; }
    }
    .brand h1 {
      margin: 0;
      font-family: Fraunces, Georgia, serif;
      font-size: clamp(2.4rem, 5vw, 3.4rem);
      font-weight: 700;
      letter-spacing: -0.03em;
    }
    .meta { color: var(--mute); margin: 0.4rem 0 0; }
    .reference, .compose, .score-panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 1.25rem 1.35rem;
    }
    .compose { background: var(--panel-2); }
    .kicker {
      margin: 0 0 0.65rem;
      font-size: 0.72rem;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--leaf);
      font-weight: 500;
    }
    #reference {
      margin: 0;
      font-family: Fraunces, Georgia, serif;
      font-size: 1.15rem;
      line-height: 1.45;
    }
    label {
      display: block;
      margin-top: 0.9rem;
      font-size: 0.8rem;
      color: var(--mute);
      font-weight: 500;
    }
    textarea, input, select, button { font: inherit; width: 100%; color: var(--mist); }
    textarea, input, select {
      margin-top: 0.35rem;
      background: var(--ink);
      border: 1px solid var(--line);
      border-radius: 4px;
      padding: 0.7rem 0.8rem;
    }
    textarea { min-height: 10rem; resize: vertical; }
    .row {
      display: grid;
      gap: 0.75rem;
      margin-top: 0.25rem;
    }
    @media (min-width: 640px) {
      .row { grid-template-columns: 1fr 1fr auto; align-items: end; }
      .row button { width: auto; min-width: 7rem; }
    }
    button {
      margin-top: 0.9rem;
      background: var(--leaf);
      color: var(--ink);
      border: 0;
      border-radius: 4px;
      padding: 0.75rem 1.2rem;
      font-weight: 600;
      cursor: pointer;
    }
    button:disabled { opacity: 0.6; cursor: wait; }
    .score-panel { position: sticky; top: 1rem; min-height: 16rem; }
    .final {
      font-family: Fraunces, Georgia, serif;
      font-size: 3.5rem;
      font-weight: 700;
      letter-spacing: -0.03em;
      line-height: 1;
      margin: 0.2rem 0 0.35rem;
    }
    .hint { color: var(--mute); font-size: 0.85rem; margin: 0 0 1.2rem; }
    .metric { margin-top: 1rem; }
    .metric-top {
      display: flex;
      justify-content: space-between;
      font-size: 0.92rem;
      margin-bottom: 0.4rem;
    }
    .metric-top strong { color: var(--leaf); font-weight: 600; }
    .track {
      height: 7px;
      border-radius: 999px;
      background: var(--ink);
      overflow: hidden;
    }
    .fill {
      height: 100%;
      width: 0%;
      background: var(--leaf);
      border-radius: 999px;
      transition: width 0.25s ease;
    }
    .error { color: var(--danger); margin: 0; white-space: pre-wrap; }
    .empty { color: var(--mute); margin: 0; }
  </style>
</head>
<body>
  <main>
    <header class="brand">
      <h1>NOVELTY</h1>
      <p class="meta" id="context">Loading the reference…</p>
    </header>
    <section class="reference">
      <p class="kicker">Reference</p>
      <blockquote id="reference"></blockquote>
    </section>
    <section class="compose">
      <p class="kicker">Your submission</p>
      <form id="form">
        <label for="text">Text (100 words or fewer)</label>
        <textarea id="text" name="text" required placeholder="Write something on-topic but not already in the corpus…"></textarea>
        <div class="row">
          <div>
            <label for="topic">Topic</label>
            <input id="topic" name="topic" value="adhoc">
          </div>
          <div>
            <label for="reduction">Neighbor reduction</label>
            <select id="reduction" name="reduction">
              <option value="max" selected>max</option>
              <option value="top3_mean">top3_mean</option>
            </select>
          </div>
          <button type="submit" id="score-btn">Score</button>
        </div>
      </form>
    </section>
    <aside class="score-panel" id="results" aria-live="polite">
      <p class="kicker">Live score</p>
      <p class="empty">Press Score to compute relevance, novelty, and final score.</p>
    </aside>
  </main>
  <script>
    const results = document.getElementById("results");
    const scoreBtn = document.getElementById("score-btn");
    const clamp01 = (value) => Math.max(0, Math.min(1, Number(value) || 0));
    const pct = (value) => (clamp01(value) * 100).toFixed(1) + "%";

    function renderScore(data) {
      results.innerHTML =
        '<p class="kicker">Live score</p>' +
        '<p class="final" id="final"></p>' +
        '<p class="hint">final = max(relevance, 0) × novelty</p>' +
        '<div class="metric"><div class="metric-top"><span>Relevance</span><strong id="rel"></strong></div><div class="track"><div class="fill" id="rel-fill"></div></div></div>' +
        '<div class="metric"><div class="metric-top"><span>Novelty</span><strong id="nov"></strong></div><div class="track"><div class="fill" id="nov-fill"></div></div></div>';
      document.getElementById("final").textContent = Number(data.final_score).toFixed(3);
      document.getElementById("rel").textContent = Number(data.relevance).toFixed(3);
      document.getElementById("nov").textContent = Number(data.novelty).toFixed(3);
      document.getElementById("rel-fill").style.width = pct(data.relevance);
      document.getElementById("nov-fill").style.width = pct(data.novelty);
    }

    fetch("/api/context").then((response) => response.json()).then((data) => {
      document.getElementById("context").textContent =
        data.corpus_size + " prior submissions · " + data.embedding_model;
      document.getElementById("reference").textContent = data.reference.text;
    }).catch(() => {
      document.getElementById("context").textContent = "Could not load reference context.";
    });

    document.getElementById("form").addEventListener("submit", async (event) => {
      event.preventDefault();
      results.innerHTML = '<p class="kicker">Live score</p><p class="empty">Scoring…</p>';
      scoreBtn.disabled = true;
      try {
        const response = await fetch("/api/score", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            text: document.getElementById("text").value,
            topic: document.getElementById("topic").value,
            reduction: document.getElementById("reduction").value
          })
        });
        const data = await response.json();
        if (!response.ok) {
          results.innerHTML = '<p class="kicker">Live score</p><p class="error"></p>';
          results.querySelector(".error").textContent = data.error || "Scoring failed.";
          return;
        }
        renderScore(data);
      } catch (error) {
        results.innerHTML = '<p class="kicker">Live score</p><p class="error"></p>';
        results.querySelector(".error").textContent = String(error);
      } finally {
        scoreBtn.disabled = false;
      }
    });
  </script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/":
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
            return
        if self.path == "/api/context":
            self._send_json(200, context())
            return
        self._send_json(404, {"error": "Not found."})

    def do_POST(self) -> None:
        if self.path != "/api/score":
            self._send_json(404, {"error": "Not found."})
            return
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8")
        try:
            body = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            self._send_json(400, {"error": "Request body must be JSON."})
            return
        text = str(body.get("text", ""))
        topic = str(body.get("topic") or "adhoc")
        reduction = str(body.get("reduction") or "max")
        if reduction not in ("max", "top3_mean"):
            self._send_json(400, {"error": "reduction must be max or top3_mean."})
            return
        try:
            self._send_json(200, score_text(text, topic=topic, reduction=reduction))
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
        except GeminiError as exc:
            self._send_json(502, {"error": str(exc)})

    def log_message(self, fmt: str, *args) -> None:
        print(f"{self.address_string()} {fmt % args}")

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: int, payload: dict) -> None:
        self._send(status, json.dumps(payload).encode("utf-8"), "application/json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the local novelty scoring page.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Open http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
