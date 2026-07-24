---
title: Chat with PDF
emoji: 📄
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 6.20.0
app_file: app.py
pinned: true
short_description: Ask a PDF questions; answers cite the source page
---

# 📄 Chat with PDF

Upload a PDF, ask questions in plain language, and get answers **with the source
page cited**. This is the
[Stage 1 project](https://golamrasul97.github.io/ai-engineering-roadmap/stage-1-rag/)
of my AI Engineering Roadmap: a minimal but complete
**Retrieval-Augmented Generation (RAG)** application.

**▶️ Live demo:** https://huggingface.co/spaces/mdgolamrasul/chat-with-pdf

## The RAG loop

```
ingest → chunk → embed → store → retrieve → augment → generate
```

Each phase lives in its own small module so the pattern is visible:

| Phase | File | What happens |
|-------|------|--------------|
| ingest + chunk | `rag/ingest.py` | Load the PDF page by page, split into overlapping passages, keep each passage's page number |
| embed + store + retrieve | `rag/index.py` | Embed passages with `all-MiniLM-L6-v2`, store vectors in an in-memory FAISS index, fetch the top-k nearest a question |
| augment + generate | `rag/chat.py` | Build a page-tagged prompt from the retrieved passages and ask the LLM to answer **and cite pages** |
| config | `rag/config.py` | All settings, read from the environment |
| UI | `app.py` | Gradio front end + per-session state (the Hugging Face Spaces entrypoint) |

## Diagnose a bad answer: retrieval vs generation

Every answer shows the exact chunks it was built from (page + FAISS distance +
text) in the right-hand panel. That splits every failure cleanly in two:

- **The answer's fact isn't in any retrieved chunk → a _retrieval_ problem.**
  Tune `CHUNK_SIZE` / `CHUNK_OVERLAP` / `TOP_K`, or swap the embedding model.
- **The fact is right there in a chunk but the answer is wrong → a _generation_
  problem.** Tune the prompt in `rag/chat.py` or use a stronger model.

Distance is the first clue: a good match sits near `0.5`; if the closest chunk
is up around `1.5+`, retrieval found nothing relevant.

## Run it locally (against Ollama — free, no API key)

Local dev talks to [Ollama](https://ollama.com) through its OpenAI-compatible
endpoint. Requires **Python 3.10+** (this repo was built on 3.13).

```bash
# 1. Model backend
ollama serve                 # start the local server (if not already running)
ollama pull llama3.2         # one-time model download

# 2. App
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env         # defaults already point at Ollama — no key needed
python app.py                # open the printed http://127.0.0.1:7860 URL
```

### Configuration (only env vars change between local and hosted)

| Variable | Local (Ollama) | Hosted (Groq) |
|----------|----------------|----------------|
| `LLM_BASE_URL` | `http://localhost:11434/v1` | `https://api.groq.com/openai/v1` |
| `LLM_MODEL` | `llama3.2` | `llama-3.3-70b-versatile` |
| API key | not needed (dummy) | `GROQ_API_KEY` (a real free key) |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` / `TOP_K` | `1000` / `150` / `4` | same |

The app reads the key from `LLM_API_KEY`, then `GROQ_API_KEY`, then falls back to
a dummy value Ollama ignores — so local dev needs no key and the Space uses its
`GROQ_API_KEY` Secret unchanged.

## Ship it: GitHub (source) + Hugging Face Space (live demo)

The same repo pushes to two remotes. GitHub is the source of truth; the Space is
the running demo.

### 1. Push to GitHub (`origin` is already set)

```bash
git add -A
git commit -m "Chat with PDF: minimal RAG app with page citations"
git push -u origin main
```

### 2. Create the Hugging Face Space and push to it

1. On https://huggingface.co → **New → Space**. Name it `chat-with-pdf`, choose
   **SDK: Gradio**, hardware **CPU basic (free)**.
2. Add the Space as a second remote and push (replace `<HF_USERNAME>`):

   ```bash
   git remote add space https://huggingface.co/spaces/<HF_USERNAME>/chat-with-pdf
   git push space main
   ```

### 3. Configure the Space (Groq backend)

In the Space → **Settings**:

- **Secrets** → add `GROQ_API_KEY` = a free key from https://console.groq.com
  (this is why it's never committed).
- **Variables** → add `LLM_BASE_URL` = `https://api.groq.com/openai/v1` and
  `LLM_MODEL` = `llama-3.3-70b-versatile`.

The Space rebuilds and runs `app.py` automatically. Ollama can't run on the free
CPU Space, which is exactly why the hosted demo points at Groq.

### 4. Auto-sync from GitHub (enabled)

`.github/workflows/sync-to-hf.yml` force-pushes `main` to the Space on every
push to GitHub, so **`git push origin` is all you ever need** — the live demo
redeploys itself.

It's wired up with an `HF_TOKEN` GitHub Actions secret (a Hugging Face **write**
token from https://huggingface.co/settings/tokens). To reuse this repo under a
different account, swap the two `mdgolamrasul` occurrences in the workflow's push
URL for your HF username and set your own `HF_TOKEN` secret. (To push to the
Space manually instead, delete that file.)

## Project layout

```
chat-with-pdf/
├── app.py                 # Gradio UI + per-session state (HF Spaces entrypoint)
├── rag/
│   ├── config.py          # settings from env
│   ├── ingest.py          # ingest + chunk (keeps page numbers)
│   ├── index.py           # embed + store + retrieve
│   └── chat.py            # augment + generate (with page citations)
├── requirements.txt       # pinned, CPU-only
├── .env.example
└── .github/workflows/sync-to-hf.yml
```
