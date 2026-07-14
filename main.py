"""
VeriFi-Lite: a minimal RAG agent for the Maritime builder challenge.

A lightweight, serverless-friendly take on the grounded-retrieval idea behind
VeriFi (my team's in-progress C++ vector store) -- rebuilt fresh in Python to
fit Maritime's message-triggered, sleep/wake deployment model.

HTTP contract (mirrors Maritime's documented /health + /run agent shape):
  - HTTP server on the Maritime-injected PORT (default 8080 locally)
  - GET  /       -> browser UI for asking questions (demo frontend)
  - GET  /health -> {"status": "ok"}
  - POST /run    -> {"task": "..."} -> {"result": "...", "sources": [...], "mode": "..."}

Implementation notes:
  - Python standard library ONLY (http.server + urllib). No pip dependencies.
    This matches Maritime's own custom-repo sample (maritime-hello-web), which
    is the reference image proven to boot on their Firecracker micro-VMs.
  - OpenAI is called over raw HTTPS. Maritime injects OPENAI_API_KEY and
    OPENAI_BASE_URL at runtime (their managed LLM proxy); both are honored.
  - Graceful degradation: without OPENAI_API_KEY, /run falls back to keyword
    retrieval and returns the raw top chunks, so the deployment is
    demonstrably alive even before secrets are configured.

Flow per request (the RAG loop):
  1. Embed the incoming question            (text-embedding-3-small)
  2. Cosine-similarity top-k over the corpus (pure Python, exact search --
     the same exact-kNN-first philosophy as VeriFi)
  3. Ask the LLM to answer ONLY from the retrieved chunks, with citations
  4. Return the grounded answer + which sources were used
"""

import json
import math
import os
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# ----------------------------------------------------------------------------
# Tiny embedded corpus. Finance/market-structure flavored, echoing VeriFi's
# domain (grounding LLM answers in regulatory/market documents).
# Each entry: (source_id, text). Small on purpose -- this is a 1-hour demo.
# ----------------------------------------------------------------------------
CORPUS = [
    ("mkt-001",
     "A limit order is an instruction to buy or sell a security at a specified "
     "price or better. A buy limit order executes at the limit price or lower; "
     "a sell limit order executes at the limit price or higher."),
    ("mkt-002",
     "Price-time priority means resting orders are matched first by best price, "
     "then by arrival time among orders at the same price. It is the standard "
     "matching rule in most electronic limit order books."),
    ("mkt-003",
     "Slippage is the difference between the expected price of a trade and the "
     "price at which it actually executes. It is most severe in thin or fast "
     "markets, such as during a flash crash."),
    ("mkt-004",
     "VWAP, the volume-weighted average price, is total traded value divided by "
     "total volume over a window. Large orders are often benchmarked against "
     "VWAP because sweeping a thin book raises the blended average fill price."),
    ("mkt-005",
     "Regulation NMS requires trading centers to prevent trade-throughs: "
     "executions at prices worse than protected quotations displayed by other "
     "venues, subject to defined exceptions."),
    ("mkt-006",
     "A retrieval-augmented generation (RAG) system grounds a language model's "
     "answer in retrieved source passages, reducing hallucination by making the "
     "model cite documents rather than rely on parametric memory alone."),
]

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
_corpus_vecs = None  # lazily computed once per wake, then cached in memory


# ----------------------------------------------------------------------------
# OpenAI over raw HTTPS (stdlib only). Maritime injects OPENAI_API_KEY and
# OPENAI_BASE_URL (its managed proxy); default to the public endpoint locally.
# ----------------------------------------------------------------------------
def _openai_post(path: str, payload: dict) -> dict:
    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    req = urllib.request.Request(
        f"{base}{path}",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def _embed(texts: list[str]) -> list[list[float]]:
    resp = _openai_post("/embeddings", {"model": EMBED_MODEL, "input": texts})
    return [d["embedding"] for d in resp["data"]]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _retrieve(question: str, k: int = 3):
    """Exact cosine top-k over the corpus (VeriFi's baseline philosophy)."""
    global _corpus_vecs
    if _corpus_vecs is None:
        _corpus_vecs = _embed([text for _, text in CORPUS])
    qv = _embed([question])[0]
    scored = [
        (_cosine(qv, cv), sid, text)
        for cv, (sid, text) in zip(_corpus_vecs, CORPUS)
    ]
    scored.sort(reverse=True)
    return scored[:k]


def _keyword_fallback(question: str, k: int = 3):
    """Keyless fallback: crude keyword overlap so the demo never 500s."""
    qwords = set(question.lower().split())
    scored = [
        (len(qwords & set(text.lower().split())), sid, text)
        for sid, text in CORPUS
    ]
    scored.sort(reverse=True)
    return scored[:k]


def _answer(question: str, hits) -> str:
    context = "\n\n".join(f"[{sid}] {text}" for _, sid, text in hits)
    prompt = (
        "Answer the question using ONLY the sources below. "
        "Cite source ids in brackets like [mkt-002]. "
        "If the sources do not contain the answer, say so plainly.\n\n"
        f"Sources:\n{context}\n\nQuestion: {question}"
    )
    resp = _openai_post("/chat/completions", {
        "model": CHAT_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 300,
    })
    return resp["choices"][0]["message"]["content"]


def run_task(question: str) -> dict:
    question = question.strip()
    if not question:
        return {"result": "Send a question in the 'task' field."}

    if os.environ.get("OPENAI_API_KEY"):
        try:
            hits = _retrieve(question)
            answer = _answer(question, hits)
            mode = "rag"
        except (urllib.error.URLError, KeyError, OSError) as exc:
            # LLM backend unavailable: stay demoable instead of 500ing.
            hits = _keyword_fallback(question)
            answer = (
                f"LLM call failed ({exc}); returning raw top-matching sources "
                "(keyword mode):\n\n"
                + "\n\n".join(f"[{sid}] {text}" for _, sid, text in hits)
            )
            mode = "keyword-fallback"
    else:
        hits = _keyword_fallback(question)
        answer = (
            "OPENAI_API_KEY not set -- returning raw top-matching sources "
            "(keyword mode):\n\n"
            + "\n\n".join(f"[{sid}] {text}" for _, sid, text in hits)
        )
        mode = "keyword-fallback"

    return {
        "result": answer,
        "sources": [sid for _, sid, _ in hits],
        "mode": mode,
    }


# ----------------------------------------------------------------------------
# Browser UI: a single self-contained page served at GET / (no build step,
# no external assets). It calls POST /run on the same origin.
# ----------------------------------------------------------------------------
INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>VeriFi-Lite &mdash; grounded answers</title>
<style>
  :root {
    --bg: #0b0e14; --panel: #121722; --panel2: #0e131d;
    --text: #e6e9f0; --muted: #8b93a7; --accent: #4f8cff;
    --accent2: #7c5cff; --ok: #2ecc8f; --warn: #f0b429;
    --border: #232b3b;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; min-height: 100vh; background:
      radial-gradient(1200px 600px at 80% -10%, #16203a 0%, transparent 60%),
      radial-gradient(900px 500px at -10% 110%, #131b2e 0%, transparent 55%),
      var(--bg);
    color: var(--text);
    font: 16px/1.55 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
    display: flex; align-items: flex-start; justify-content: center;
    padding: 48px 16px;
  }
  .app { width: 100%; max-width: 780px; }
  .badge {
    display: inline-flex; align-items: center; gap: 8px;
    font-size: 12px; letter-spacing: .12em; text-transform: uppercase;
    color: var(--muted); border: 1px solid var(--border);
    padding: 6px 12px; border-radius: 999px; background: var(--panel2);
  }
  .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--ok); }
  h1 { font-size: 34px; margin: 18px 0 6px; letter-spacing: -0.02em; }
  h1 .grad {
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    -webkit-background-clip: text; background-clip: text; color: transparent;
  }
  .sub { color: var(--muted); margin: 0 0 28px; max-width: 60ch; }
  .card {
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 16px; padding: 20px;
  }
  .row { display: flex; gap: 10px; }
  textarea {
    flex: 1; resize: vertical; min-height: 64px; padding: 14px 16px;
    border-radius: 12px; border: 1px solid var(--border);
    background: var(--panel2); color: var(--text); font: inherit;
    outline: none;
  }
  textarea:focus { border-color: var(--accent); }
  button {
    align-self: flex-end; padding: 13px 22px; border: 0; border-radius: 12px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    color: white; font: inherit; font-weight: 600; cursor: pointer;
    transition: opacity .15s ease;
  }
  button:disabled { opacity: .45; cursor: wait; }
  .examples { margin-top: 14px; display: flex; flex-wrap: wrap; gap: 8px; }
  .chip {
    font-size: 13px; color: var(--muted); background: var(--panel2);
    border: 1px solid var(--border); border-radius: 999px;
    padding: 6px 12px; cursor: pointer; transition: all .15s ease;
  }
  .chip:hover { color: var(--text); border-color: var(--accent); }
  .answer { margin-top: 18px; display: none; }
  .answer.show { display: block; }
  .meta { display: flex; gap: 8px; align-items: center; margin-bottom: 12px; }
  .tag {
    font-size: 12px; font-weight: 600; padding: 4px 10px; border-radius: 999px;
  }
  .tag.rag { background: rgba(46,204,143,.12); color: var(--ok); }
  .tag.fallback { background: rgba(240,180,41,.12); color: var(--warn); }
  .tag.src { background: rgba(79,140,255,.12); color: var(--accent); }
  #result { white-space: pre-wrap; }
  .spinner {
    display: none; margin: 18px auto 0; width: 26px; height: 26px;
    border: 3px solid var(--border); border-top-color: var(--accent);
    border-radius: 50%; animation: spin .8s linear infinite;
  }
  .spinner.show { display: block; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .foot { margin-top: 22px; color: var(--muted); font-size: 13px; }
  .foot code {
    background: var(--panel2); border: 1px solid var(--border);
    padding: 2px 6px; border-radius: 6px;
  }
</style>
</head>
<body>
<div class="app">
  <span class="badge"><span class="dot"></span>Live on Maritime &mdash; sleeps when idle, wakes on request</span>
  <h1>VeriFi<span class="grad">-Lite</span></h1>
  <p class="sub">A minimal grounded-retrieval (RAG) agent. Ask a market-structure
  question; it retrieves the most relevant passages from its embedded corpus and
  answers <em>only</em> from them, with citations. Off-corpus questions are
  declined rather than hallucinated.</p>

  <div class="card">
    <div class="row">
      <textarea id="q" placeholder="Ask about limit orders, slippage, VWAP, price-time priority, Reg NMS, or RAG itself&hellip;"></textarea>
      <button id="ask">Ask</button>
    </div>
    <div class="examples">
      <span class="chip">Why can a large limit order in a thin book fill worse than expected?</span>
      <span class="chip">What is price-time priority?</span>
      <span class="chip">How does RAG reduce hallucination?</span>
      <span class="chip">What does Regulation NMS require?</span>
    </div>
    <div class="spinner" id="spin"></div>
    <div class="answer" id="answer">
      <div class="meta" id="meta"></div>
      <div id="result"></div>
    </div>
  </div>

  <p class="foot">Also available as JSON: <code>GET /health</code> and
  <code>POST /run {"task": "&hellip;"}</code></p>
</div>
<script>
  const q = document.getElementById("q");
  const btn = document.getElementById("ask");
  const spin = document.getElementById("spin");
  const answer = document.getElementById("answer");
  const meta = document.getElementById("meta");
  const result = document.getElementById("result");

  document.querySelectorAll(".chip").forEach(c =>
    c.addEventListener("click", () => { q.value = c.textContent; ask(); }));

  async function ask() {
    const task = q.value.trim();
    if (!task || btn.disabled) return;
    btn.disabled = true;
    spin.classList.add("show");
    answer.classList.remove("show");
    // Resolve /run relative to the page URL so it works both locally ("/run")
    // and behind Maritime's gateway prefix ("/a/{agent-id}/run").
    const runUrl = new URL("run",
      location.href.endsWith("/") ? location.href : location.href + "/");
    try {
      const r = await fetch(runUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ task }),
      });
      const data = await r.json();
      const isRag = data.mode === "rag";
      meta.innerHTML =
        '<span class="tag ' + (isRag ? "rag" : "fallback") + '">' +
        (isRag ? "grounded &middot; rag" : "keyword fallback") + "</span>" +
        (data.sources || []).map(s => '<span class="tag src">' + s + "</span>").join("");
      result.textContent = data.result || JSON.stringify(data);
      answer.classList.add("show");
    } catch (e) {
      meta.innerHTML = '<span class="tag fallback">error</span>';
      result.textContent = "Request failed: " + e +
        "\\n(If the agent was asleep, it may just be waking up -- try again.)";
      answer.classList.add("show");
    } finally {
      btn.disabled = false;
      spin.classList.remove("show");
    }
  }
  btn.addEventListener("click", ask);
  q.addEventListener("keydown", e => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); ask(); }
  });
</script>
</body>
</html>
"""


# ----------------------------------------------------------------------------
# HTTP server (stdlib): /, /health, and /run on the Maritime-injected PORT.
# ----------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    def _send_json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", ""):
            body = INDEX_HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/health":
            self._send_json(200, {"status": "ok"})
        else:
            self._send_json(404, {"detail": "Not found. Try GET /, GET /health, or POST /run."})

    def do_POST(self):
        if self.path != "/run":
            self._send_json(404, {"detail": "Not found. Try POST /run."})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length) or b"{}")
            task = str(body.get("task", ""))
        except (ValueError, json.JSONDecodeError):
            self._send_json(400, {"detail": "Body must be JSON: {\"task\": \"...\"}"})
            return
        self._send_json(200, run_task(task))

    def log_message(self, format, *args):  # noqa: A002 -- stdlib signature
        pass  # keep container logs quiet


if __name__ == "__main__":
    # Maritime injects PORT for custom-repo deployments; 8080 is the local default.
    port = int(os.environ.get("PORT", "8080"))
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
