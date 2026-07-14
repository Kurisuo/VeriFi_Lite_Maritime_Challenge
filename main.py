"""
VeriFi-Lite: a minimal RAG agent for the Maritime builder challenge.

A lightweight, serverless-friendly take on the grounded-retrieval idea behind
VeriFi (my team's in-progress C++ vector store) -- rebuilt fresh in Python to
fit Maritime's message-triggered, sleep/wake deployment model.

HTTP contract (mirrors Maritime's documented /health + /run agent shape):
  - HTTP server on the Maritime-injected PORT (default 8080 locally)
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
# HTTP server (stdlib): /health and /run on the Maritime-injected PORT.
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
        if self.path.rstrip("/") in ("", "/health".rstrip("/")) or self.path == "/health":
            self._send_json(200, {"status": "ok"})
        else:
            self._send_json(404, {"detail": "Not found. Try GET /health or POST /run."})

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
