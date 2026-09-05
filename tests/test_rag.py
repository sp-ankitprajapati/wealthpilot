"""
Week 3 — Policy RAG.

Run with: pytest tests/test_rag.py -v

Same split as test_agent.py: PLUMBING is testable now, deterministically,
with no network call. QUALITY of real semantic retrieval needs real
embeddings, which need huggingface.co (for the local model) or a
provider's hosted embeddings API — neither reachable from a network-
sandboxed environment. Run rag.py's __main__ on your own machine, with
real internet access, to see real retrieval quality.

FakeEmbeddings below is a bag-of-words hash embedding — deterministic,
dependency-free, good enough to prove chunking, indexing, retrieval, and
metadata round-tripping all work, NOT a stand-in for real semantic
similarity.
"""

import hashlib
import json

import pytest
from langchain_core.embeddings import Embeddings

from wealthpilot.policy_docs import POLICY_DOCS
from wealthpilot.rag import chunk_policy_docs, build_policy_index, make_retrieve_policy_tool


class FakeEmbeddings(Embeddings):
    """Bag-of-words hashed into a fixed-size vector. Two texts sharing
    more words get more similar vectors — enough signal to prove
    retrieval mechanics work, nowhere near enough to judge real semantic
    quality."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for word in text.lower().split():
            idx = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim
            vec[idx] += 1.0
        norm = sum(v * v for v in vec) ** 0.5
        return [v / norm for v in vec] if norm else vec

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


@pytest.fixture(scope="module")
def fake_index():
    return build_policy_index(POLICY_DOCS, FakeEmbeddings())


class TestChunking:
    def test_exactly_one_chunk_per_clause_no_merging_no_orphans(self):
        """Regression test for the actual bug found while building this:
        an earlier version merged clauses 3.1+3.2 into one chunk (tagged
        only '3.1'), and a version before that cut clause 3.4 in half
        into an orphan 'n/a'-tagged chunk. Both are wrong for the same
        reason — a chunk must correspond to exactly one citable clause."""
        texts, metadatas = chunk_policy_docs(POLICY_DOCS)
        assert len(texts) == 8  # 5 clauses in doc 1, 3 in doc 2
        clause_ids = [m["clause_id"] for m in metadatas]
        assert "n/a" not in clause_ids
        assert len(clause_ids) == len(set(clause_ids))  # no duplicates = no merges

    def test_every_chunk_contains_its_own_clause_header(self):
        texts, metadatas = chunk_policy_docs(POLICY_DOCS)
        for text, meta in zip(texts, metadatas):
            assert f"## {meta['clause_id']}" in text


class TestRetrieval:
    def test_retrieval_returns_k_results(self, fake_index):
        tool_fn = make_retrieve_policy_tool(fake_index)
        results = json.loads(tool_fn.invoke({"query": "debt service coverage", "k": 3}))
        assert len(results) == 3

    def test_retrieved_results_have_required_fields(self, fake_index):
        tool_fn = make_retrieve_policy_tool(fake_index)
        results = json.loads(tool_fn.invoke({"query": "leverage limits", "k": 2}))
        for r in results:
            assert set(r.keys()) == {"text", "source", "clause"}

    def test_leverage_query_surfaces_the_leverage_clause(self, fake_index):
        """A crude but real check: even hash-bag-of-words retrieval should
        rank clause 3.2 (Leverage Limits — mentions 'leverage', 'debt',
        'equity' repeatedly) above an unrelated clause like 5.3
        (Escalation Path) for a query about leverage."""
        tool_fn = make_retrieve_policy_tool(fake_index)
        results = json.loads(tool_fn.invoke({"query": "leverage debt equity ratio", "k": 1}))
        assert results[0]["clause"] == "3.2"

    def test_tool_output_is_valid_json_string_not_a_raw_list(self, fake_index):
        """Same lesson as Week 2's detect_red_flags: don't trust the
        framework to serialize consistently — verify content really is a
        JSON string, since a LangGraph ToolNode would otherwise be the
        one calling this, and Week 2 already found a case where it
        silently didn't serialize an empty list."""
        tool_fn = make_retrieve_policy_tool(fake_index)
        raw_output = tool_fn.invoke({"query": "anything", "k": 2})
        assert isinstance(raw_output, str)
        json.loads(raw_output)  # must not raise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
