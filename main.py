"""
VeriFi-Lite: a minimal RAG agent for the Maritime builder challenge.

A lightweight, serverless-friendly take on the grounded-retrieval idea behind
VeriFi (my team's in-progress C++ vector store) -- rebuilt fresh in Python to
fit Maritime's message-triggered, sleep/wake deployment model.

HTTP contract follows Maritime's LangGraph template exactly, per
https://maritime.sh/docs/frameworks/langgraph :
  - FastAPI server on port 8080
  - GET  /health -> {"status": "ok"}
  - POST /run    -> {"task": "..."} -> {"result": "..."}
  - OPENAI_API_KEY provided via Maritime environment variables
    (set with `maritime env set OPENAI_API_KEY=sk-...`, per
     https://maritime.sh/blog/deploying-crewai-agents-to-production)

Flow per request (the RAG loop):
  1. Embed the incoming question            (OpenAI text-embedding-3-small)
  2. Cosine-similarity top-k over the corpus (pure Python, exact search --
     the same exact-kNN-first philosophy as VeriFi)
  3. Ask the LLM to answer ONLY from the retrieved chunks, with citations
  4. Return the grounded answer + which sources were used

Graceful degradation: if OPENAI_API_KEY is missing, /run still works --
it falls back to keyword retrieval and returns the raw top chunks, so the
deployment is demonstrably alive even before secrets are configured.
"""

import os
import math
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="VeriFi-Lite", version="0.1.0")

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

# ----------------------------------------------------------------------------
# Embedding + retrieval
# ----------------------------------------------------------------------------
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
_corpus_vecs = None  # lazily computed once per wake, then cached in memory


def _client():
    from openai import OpenAI  # imported lazily so /health works keyless
    return OpenAI()  # reads OPENAI_API_KEY from env (Maritime-injected)


def _embed(texts: list[str]) -> list[list[float]]:
    resp = _client().embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


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
    resp = _client().chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
    )
    return resp.choices[0].message.content


# ----------------------------------------------------------------------------
# Maritime LangGraph-template HTTP contract
# (port 8080, /health, /run {"task"} -> {"result"})
# per https://maritime.sh/docs/frameworks/langgraph
# ----------------------------------------------------------------------------

class Task(BaseModel):
    task: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/run")
def run(body: Task):
    question = body.task.strip()
    if not question:
        return {"result": "Send a question in the 'task' field."}

    if os.environ.get("OPENAI_API_KEY"):
        hits = _retrieve(question)
        answer = _answer(question, hits)
        mode = "rag"
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


if __name__ == "__main__":
    import uvicorn
    # Maritime injects PORT for custom-repo deployments; 8080 remains the
    # documented LangGraph-compatible default for local runs.
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))