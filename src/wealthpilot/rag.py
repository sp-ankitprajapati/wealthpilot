"""Policy chunking and retrieval with injectable embeddings.

Callers supply embeddings and indexes. Tests use controlled lexical
representations; live semantic retrieval requires separate evaluation.
"""

import json
from uuid import uuid4
from typing import Annotated
from pydantic import Field

from langchain_core.tools import tool
from langchain_chroma import Chroma

from .policy_docs import POLICY_DOCS


def _simple_recursive_split(
    text: str, chunk_size: int, chunk_overlap: int, separators: list[str]
) -> list[str]:
    """Minimal recursive character splitter -- try each separator in
    priority order, keep pieces under chunk_size, fall back to the next
    separator only for pieces still too big.

    Replaces langchain_text_splitters.RecursiveCharacterTextSplitter,
    used here only as a rare fallback for a single oversized clause.
    That package pulls in sentence_transformers -> transformers -> torch
    as a TRANSITIVE import even though none of it is ever used --
    confirmed directly: a fresh `from langchain_text_splitters import
    RecursiveCharacterTextSplitter` cost ~7 seconds on every process
    start (not a cold-cache artifact -- reproduced warm too), even
    importing the specific submodule instead of the package root. For
    one fallback code path on a rare oversized clause, that's not a
    reasonable trade. This does the one thing actually needed here in
    about 20 lines with zero heavy dependencies.
    """
    if len(text) <= chunk_size:
        return [text]
    if not separators:
        step = max(chunk_size - chunk_overlap, 1)
        return [text[i : i + chunk_size] for i in range(0, len(text), step)]

    sep, *rest = separators
    parts = text.split(sep) if sep else list(text)
    chunks, current = [], ""
    for part in parts:
        candidate = f"{current}{sep}{part}" if current else part
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            if len(part) > chunk_size:
                chunks.extend(
                    _simple_recursive_split(part, chunk_size, chunk_overlap, rest)
                )
                current = ""
            else:
                current = part
    if current:
        chunks.append(current)
    return chunks


def chunk_policy_docs(
    docs: list[dict], max_clause_chars: int = 800
) -> tuple[list[str], list[dict]]:
    """Returns (texts, metadatas) — kept as parallel lists because that's
    the exact shape Chroma.from_texts() wants, not because it's the
    prettiest data structure.

    Splits on the "## " clause boundary EXPLICITLY, as a plain string
    split — not via a recursive splitter at the top level. First version
    of this function used a recursive splitter directly with
    chunk_size=700, and it silently MERGED clause 3.1 and 3.2 into one
    chunk (tagged only "3.1") because their combined length fit under
    700 — confirmed by printing the actual chunk text, not assumed. A
    generic splitter optimizes for filling each chunk close to
    chunk_size; it has no concept of "these are two distinct citable
    clauses." Explicit string-splitting on the clause boundary guarantees
    one clause per chunk, with an unambiguous clause_id, every time. The
    recursive splitter is still useful — just only as a FALLBACK for a
    single clause too long to fit in one chunk, where every sub-chunk
    correctly keeps sharing that one clause's own id.
    """
    texts, metadatas = [], []
    for doc in docs:
        for raw in doc["text"].split("\n## "):
            raw = raw.strip()
            if not raw:
                continue
            clause_text = raw if raw.startswith("##") else "## " + raw
            clause_id = clause_text[3:].split("\n", 1)[0].split(" ", 1)[0]

            if len(clause_text) <= max_clause_chars:
                texts.append(clause_text)
                metadatas.append({"source": doc["source"], "clause_id": clause_id})
            else:
                for sub in _simple_recursive_split(
                    clause_text,
                    chunk_size=max_clause_chars,
                    chunk_overlap=60,
                    separators=["\n\n", "\n", ". "],
                ):
                    texts.append(sub)
                    metadatas.append({"source": doc["source"], "clause_id": clause_id})
    return texts, metadatas


def build_policy_index(docs: list[dict], embeddings) -> Chroma:
    texts, metadatas = chunk_policy_docs(docs)
    return Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        collection_name="wealthpilot-" + uuid4().hex,
    )


def make_retrieve_policy_tool(vectorstore: Chroma):
    """Factory, not a module-level tool — same reason build_graph() takes
    a model instead of importing one: the tool needs to close over
    WHICHEVER vectorstore you built (real index in production, a tiny
    fake-embedded index in tests), and a bare module-level @tool can't do
    that."""

    @tool
    def retrieve_policy(
        query: Annotated[str, Field(min_length=1)],
        k: Annotated[int, Field(ge=1, le=20)] = 4,
    ) -> str:
        """Retrieve relevant SME underwriting policy clauses for a given
        risk question (e.g. "is high leverage a decline reason?" or "what
        happens when DSCR is below 1.0?"). Returns a JSON array of objects
        with 'text', 'source', and 'clause' — cite the clause number in
        any policy-based claim, and if nothing relevant comes back, say so
        rather than inferring policy that wasn't retrieved.
        """
        results = vectorstore.similarity_search(query, k=k)
        payload = [
            {
                "text": r.page_content,
                "source": r.metadata["source"],
                "clause": r.metadata["clause_id"],
            }
            for r in results
        ]
        # Explicit JSON serialization, as in tools.py: an
        # empty result list specifically must not be returned raw
        return json.dumps(payload)

    return retrieve_policy


if __name__ == "__main__":
    # Live check — needs real embeddings reachable over the network
    # (HuggingFace Hub for the local model, or your provider's hosted
    # embeddings). Won't run in a network-sandboxed environment; run this
    # on your own machine.
    from langchain_huggingface import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    index = build_policy_index(POLICY_DOCS, embeddings)
    tool_fn = make_retrieve_policy_tool(index)

    for query in [
        "is high leverage a decline reason?",
        "what if DSCR is below 1.0?",
        "revenue dropped this year, is that a problem?",
    ]:
        print(f"\nQ: {query}")
        results = json.loads(tool_fn.invoke({"query": query, "k": 2}))
        for r in results:
            print(f"  [{r['source']} / {r['clause']}] {r['text'][:100]}...")
