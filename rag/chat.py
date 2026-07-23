"""Phase 6 + 7 of the loop: augment -> generate.

Take the retrieved chunks, staple them into a prompt (that's "augment"), and ask
the LLM to answer using only that context and to cite its pages (that's
"generate"). We return the answer *and* the exact chunks it was given, so a
wrong answer can always be traced back to one of two causes.
"""

from dataclasses import dataclass
from functools import lru_cache

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .config import settings
from .index import retrieve

# Keep the model honest: answer only from the passages, cite pages, and admit
# when the answer isn't there rather than inventing one. This instruction is the
# difference between a citing assistant and a confident hallucinator.
SYSTEM_PROMPT = (
    "You answer questions about a single PDF. Use ONLY the context passages "
    "below — do not use outside knowledge. Each passage is tagged with its page "
    "number. Cite the page(s) you used inline, like (p. 3). If the answer is not "
    'in the context, reply exactly: "I couldn\'t find that in the document."'
)

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)


@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    """Build the chat model once. Works unchanged against Ollama or Groq —
    only the base_url / model / key differ, and those come from config."""
    return ChatOpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        temperature=0,  # factual Q&A — we want the same answer every time
    )


@dataclass
class Source:
    """One retrieved chunk, exposed to the UI so retrieval is inspectable."""

    page_number: int
    score: float          # FAISS L2 distance — smaller is more similar
    snippet: str


@dataclass
class Answer:
    text: str
    sources: list[Source]


def _format_context(hits: list[tuple[Document, float]]) -> str:
    """Lay the chunks out as page-tagged blocks the model can cite from."""
    blocks = [
        f"[page {doc.metadata.get('page_number', '?')}]\n{doc.page_content}"
        for doc, _score in hits
    ]
    return "\n\n---\n\n".join(blocks)


def answer_question(index, question: str) -> Answer:
    """Run retrieve -> augment -> generate and return the answer with its sources.

    Returning the sources alongside the answer is what makes a bad answer
    diagnosable: compare the answer against the very chunks it was handed.
      - answer's fact isn't in any chunk  -> RETRIEVAL problem (tune chunking/top_k)
      - fact is right there but answer is wrong -> GENERATION problem (tune prompt/model)
    """
    hits = retrieve(index, question)

    # A simple LangChain chain: fill the prompt, then send it to the model.
    chain = PROMPT | get_llm()
    response = chain.invoke(
        {"context": _format_context(hits), "question": question}
    )

    sources = [
        Source(
            page_number=doc.metadata.get("page_number", 0),
            score=float(score),
            snippet=doc.page_content,
        )
        for doc, score in hits
    ]
    return Answer(text=response.content, sources=sources)
