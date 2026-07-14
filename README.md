# VeriFi-Lite

VeriFi-Lite is a tiny grounded-retrieval agent built for Maritime's builder
challenge. A question sent to `POST /run` is embedded, matched by exact cosine
similarity against six market-structure passages, and answered by
`gpt-4o-mini` using only the retrieved context. Answers include source IDs such
as `[mkt-003]`.

This is a lightweight Python rebuild of the grounded-retrieval idea behind my
team's in-progress C++ vector store, VeriFi. It is not the VeriFi codebase and
should not be described as "VeriFi deployed."

## API

- `GET /health` returns `{"status":"ok"}`.
- `POST /run` accepts `{"task":"..."}`.
- With `OPENAI_API_KEY`, `/run` returns a grounded answer with `mode: "rag"`.
- Without the key, it stays demoable by returning keyword-ranked passages with
  `mode: "keyword-fallback"`.

## Run locally

```bash
python -m pip install -r requirements.txt
python main.py
```

```bash
curl http://localhost:8080/health
curl -X POST http://localhost:8080/run \
  -H "Content-Type: application/json" \
  -d '{"task":"Why can a large limit order in a thin book fill worse than expected?"}'
```

## Deploy on Maritime

The production template list did not expose a LangGraph template when this
project was deployed, so it uses Maritime's documented public GitHub-repo flow:

```bash
npm install -g maritime-cli
maritime signup
maritime guide --json
maritime templates
maritime create verifi-lite \
  --repo https://github.com/Kurisuo/VeriFi_Lite_Maritime_Challenge \
  --branch main --public --port 8080
```

No credentials or secrets are committed to this repository. After using the
agent:

```bash
maritime sleep verifi-lite
```

## Platform feedback

- The current public templates endpoint omitted LangGraph even though a
  LangGraph framework page was still discoverable through search; the direct
  docs URL returned 404.
- The custom-repo requirement for a `Dockerfile` is clear in the current
  quickstart, but differs from older template-based guidance.
- The first deployment built successfully but became `active` while its public
  URL returned HTTP 502, with `All connection attempts failed` from
  `fc-manager`.
- Recreating the dead-on-arrival agent reused the deleted agent's packaged
  rootfs tag, then failed with `pull access denied` for the new tag. Changing
  the image identity bypassed that cache issue, but the fresh micro-VM still
  failed before its exec server or application became reachable.
