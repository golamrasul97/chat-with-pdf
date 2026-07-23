"""Gradio entrypoint — the browser UI that wires the RAG pipeline together.

Hugging Face Spaces auto-runs this file. The flow is deliberately thin: this
file handles the UI and per-session state; all the real work lives in the `rag`
package, one module per phase of the loop.

The UI has two panes on purpose: the chat answer, and a "Retrieved context"
panel showing the exact chunks that produced it. That second pane is what
distinguishes a retrieval failure from a generation failure at a glance.
"""

import gradio as gr

from rag.chat import Answer, answer_question
from rag.index import build_index
from rag.ingest import load_and_chunk


def on_upload(pdf_path: str | None):
    """Build a fresh per-session index from the uploaded PDF.

    Returns (index, status_message). The index is stashed in gr.State so it
    stays scoped to this browser session — never shared across visitors.
    """
    if not pdf_path:
        return None, "Upload a PDF to begin."

    chunks = load_and_chunk(pdf_path)
    if not chunks:
        return None, "No extractable text found — the PDF may be scanned or image-only."

    index = build_index(chunks)
    pages = {c.metadata.get("page_number") for c in chunks}
    return index, f"Indexed {len(chunks)} chunks across {len(pages)} pages."


def _render_sources(answer: Answer) -> str:
    """Format the retrieved chunks as Markdown for the diagnosis panel."""
    lines = ["### Retrieved context — passages behind this answer"]
    for i, s in enumerate(answer.sources, start=1):
        # Lower score = closer match. Seeing the raw chunk text here is what
        # makes it possible to judge whether retrieval found the right passage.
        preview = s.snippet.strip().replace("\n", " ")
        lines.append(
            f"**{i}. Page {s.page_number}** · distance `{s.score:.3f}`\n\n> {preview}"
        )
    return "\n\n".join(lines)


def on_ask(question: str, history: list[dict], index):
    """Answer one question against the session's index and update both panes."""
    question = (question or "").strip()
    if not question:
        return history, "", gr.update()

    # Guard: no index yet means no PDF was uploaded. Fail loudly, not silently.
    if index is None:
        history = history + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": "Upload a PDF before asking a question."},
        ]
        return history, "", gr.update()

    answer = answer_question(index, question)
    history = history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer.text},
    ]
    # Clear the textbox and refresh the retrieved-context panel.
    return history, "", _render_sources(answer)


with gr.Blocks(title="Chat with PDF") as demo:
    gr.Markdown(
        "# 📄 Chat with PDF\n"
        "A Retrieval-Augmented Generation (RAG): upload a PDF, ask a "
        "question, and get an answer **with page-level citations**. The panel "
        "on the right shows the passages retrieved for each answer, separating "
        "retrieval quality from generation quality."
    )

    # Per-session vector index. gr.State is unique per browser session.
    index_state = gr.State(None)

    with gr.Row():
        with gr.Column(scale=3):
            pdf = gr.File(label="PDF", file_types=[".pdf"], type="filepath")
            status = gr.Markdown("Upload a PDF to begin.")
            chatbot = gr.Chatbot(height=420, label="Conversation")
            question = gr.Textbox(
                placeholder="Ask a question about the document…",
                label="Question",
                submit_btn=True,
            )

        with gr.Column(scale=2):
            sources = gr.Markdown("Retrieved context will appear here.")

    # Wire the events. Uploading (re)builds the index; asking runs the pipeline.
    pdf.change(on_upload, inputs=pdf, outputs=[index_state, status])
    question.submit(
        on_ask,
        inputs=[question, chatbot, index_state],
        outputs=[chatbot, question, sources],
    )


if __name__ == "__main__":
    demo.launch()
