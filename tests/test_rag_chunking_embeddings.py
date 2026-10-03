"""Chunking, embeddings and the vector store."""

import sys
import types

import numpy as np

from claimflow.extract import Page, read_pdf
from claimflow.rag import chunker, embeddings, store
from claimflow.rag.chunker import chunk_policy, is_heading
from claimflow.rag.embeddings import AutoEmbedder, FastEmbedEmbedder, HashingEmbedder, build_embedder
from claimflow.rag.store import VectorStore
from tests.helpers import SAMPLE_POLICY, Skip


def norm(text):
    return " ".join(text.split())


# ---------------------------------------------------------------- headings and chunks

def test_headings_are_recognised_by_shape():
    for line in ("Section 4. Excess", "Section 7", "SECTION 3 - EXCLUSIONS", "4.2 Exclusions", "2.1 Collision and accidental damage",
                 "GENERAL CONDITIONS", "Article 5: Claims", "Part B. Cover"):
        assert is_heading(line), line
    for line in ("Section 4 applies to all claims.", "25,000 of each claim", "7 days after the incident.",
                 "We pay for accidental damage to the insured vehicle.", "10 days", "x" * 120, "1 collision with a vehicle."):
        assert not is_heading(line), line


def pages(*texts):
    return [Page(i + 1, t) for i, t in enumerate(texts)]


def test_chunks_carry_page_section_source_and_a_stable_id():
    chunks = chunk_policy("d7", "policy.pdf", pages("Section 3. Cover\n3.1 Collision\nWe pay for collision damage.\n3.2 Theft\nWe pay for theft."))
    assert [c.id for c in chunks] == ["d7-p1-0", "d7-p1-1"]
    assert [c.section for c in chunks] == ["Section 3. Cover / 3.1 Collision", "Section 3. Cover / 3.2 Theft"]
    assert all(c.source == "policy.pdf" and c.doc_ref == "d7" and c.page == 1 for c in chunks)


def test_a_chunk_never_crosses_a_heading_or_a_page():
    chunks = chunk_policy("d1", "p.pdf", pages("Section 1. A\nAlpha text here.\nSection 2. B\nBeta text here.", "Section 3. C\nGamma text here."))
    assert len(chunks) == 3
    assert not any("Alpha" in c.text and "Beta" in c.text for c in chunks)
    assert [c.page for c in chunks] == [1, 1, 2]


def test_a_section_carries_over_to_the_next_page():
    chunks = chunk_policy("d1", "p.pdf", pages("Section 5. Exclusions\n5.1 Racing\nWe do not cover racing.", "We also do not cover rallying."))
    assert chunks[-1].page == 2 and chunks[-1].section == "Section 5. Exclusions / 5.1 Racing"


def test_a_heading_directly_under_a_heading_is_kept_with_its_text():
    chunks = chunk_policy("d1", "p.pdf", pages("Section 6. Making a claim\n6.1 Telling us\nTell us within 7 days."))
    assert len(chunks) == 1 and chunks[0].text.startswith("Section 6.") and "Tell us within 7 days." in chunks[0].text
    assert chunks[0].section == "Section 6. Making a claim / 6.1 Telling us"


def test_chunk_text_is_the_pages_own_words_so_quotes_can_be_checked():
    text = "Section 4. Excess\nYou must pay the first\n25,000 of every claim.\n\nWe pay the rest."
    for chunk in chunk_policy("d1", "p.pdf", pages(text)):
        assert chunk.text in norm(text)


def test_long_text_is_split_with_nothing_lost_and_blank_pages_make_no_chunks():
    body = " ".join(f"Sentence number {i} says something useful." for i in range(60))
    chunks = chunk_policy("d1", "p.pdf", pages("Section 1. Long\n" + body, "", "   \n  "), max_chars=300)
    assert len(chunks) > 3 and all(len(c.text) <= 300 for c in chunks) and {c.page for c in chunks} == {1}
    assert sum(c.text.count("Sentence number") for c in chunks) == 60
    assert chunk_policy("d1", "p.pdf", pages("", " ")) == []


def test_the_search_text_adds_the_section_so_a_short_clause_finds_its_topic():
    chunk = chunk_policy("d1", "p.pdf", pages("Section 5. Exclusions\n5.2 Wear and tear\nWe do not cover rust."))[0]
    assert chunk.search_text == chunk.text                                    # the heading is already in the text
    later = chunk_policy("d1", "p.pdf", pages("Section 5. Exclusions", "We do not cover rust."))[-1]
    assert later.search_text.startswith("Section 5. Exclusions.") and later.text == "We do not cover rust."


def test_the_sample_policy_is_split_into_sections_with_hierarchical_labels():
    pdf = read_pdf(SAMPLE_POLICY.read_bytes())
    chunks = chunk_policy("d1", "sample.pdf", pdf.pages)
    labels = {c.section for c in chunks}
    assert len(chunks) > 20 and {c.page for c in chunks} == set(range(1, 7))
    assert "Section 5. What we do not cover / 5.1 Racing and speed testing" in labels
    assert "Section 4. Excess" in labels and any("3.3 Theft" in l for l in labels)


# ---------------------------------------------------------------- embeddings

def test_hashing_embeddings_are_deterministic_normalised_and_wording_sensitive():
    a, b = HashingEmbedder(), HashingEmbedder()
    texts = ["The excess for a collision claim is 25,000.", "We do not cover racing.", "excess collision claim amount"]
    va, vb = a.embed(texts), b.embed(texts)
    assert np.array_equal(va, vb) and va.shape == (3, 1024) and va.dtype == np.float32
    assert np.allclose(np.linalg.norm(va, axis=1), 1.0, atol=1e-5)
    assert va[0] @ va[2] > va[1] @ va[2] + 0.1                                 # shares wording with the excess clause only
    assert abs(float(va[0] @ va[0]) - 1.0) < 1e-5


def test_hashing_handles_empty_and_punctuation_only_text():
    vectors = HashingEmbedder().embed(["", "!!! ???", "real words here"])
    assert np.linalg.norm(vectors[0]) == 0 and np.linalg.norm(vectors[1]) == 0 and abs(np.linalg.norm(vectors[2]) - 1) < 1e-5
    assert HashingEmbedder().key == "hashing:signed-hashing-1gram-2gram:1024"


def test_the_backend_can_be_chosen_and_hashing_is_always_available():
    assert isinstance(build_embedder("hashing", "any"), HashingEmbedder)
    assert isinstance(build_embedder("auto", "BAAI/bge-small-en-v1.5"), AutoEmbedder)


def with_fastembed_class(fake, test):
    original = embeddings.FastEmbedEmbedder
    embeddings.FastEmbedEmbedder = fake
    try:
        return test()
    finally:
        embeddings.FastEmbedEmbedder = original


def test_auto_embedder_falls_back_to_hashing_when_fastembed_cannot_start_and_records_why():
    def broken(model):
        raise ImportError("no module named fastembed")

    def check():
        auto = AutoEmbedder("auto", "some-model")
        assert "not loaded yet" in auto.describe()                           # nothing happens at start-up
        vectors = auto.embed(["hello world"])
        assert auto.name == "hashing" and vectors.shape == (1, 1024) and "ImportError" in auto.fallback_reason
        assert "hashing" in auto.describe()
        return auto
    with_fastembed_class(broken, check)


def test_auto_embedder_uses_fastembed_when_it_starts_and_only_loads_it_once():
    created = []

    class Working(embeddings.Embedder):
        name, dim, floor = "fastembed", 4, 0.45

        def __init__(self, model):
            self.model = model
            created.append(model)

        def embed(self, texts):
            return np.ones((len(texts), 4), dtype=np.float32) / 2

    def check():
        auto = AutoEmbedder("auto", "the-model")
        assert created == []                                                  # lazy
        auto.embed(["a"]); auto.embed_query("b"); auto.key
        assert created == ["the-model"] and auto.key == "fastembed:the-model:4" and auto.floor == 0.45
        assert auto.fallback_reason is None
    with_fastembed_class(Working, check)


def test_an_explicit_hashing_backend_never_tries_fastembed():
    def forbidden(model):
        raise AssertionError("fastembed must not be started")
    with_fastembed_class(forbidden, lambda: build_embedder("hashing", "m").embed(["x"]))


def test_fastembed_wrapper_normalises_vectors_and_uses_the_query_prefix_when_available():
    class FakeEngine:
        def __init__(self, model_name):
            self.model_name = model_name

        def embed(self, texts):
            for t in texts:
                yield np.array([3.0, 4.0, 0.0]) if t != "dimension probe" else np.array([1.0, 0.0, 0.0])

        def query_embed(self, text):
            yield np.array([0.0, 0.0, 5.0])

    sys.modules["fastembed"] = types.SimpleNamespace(TextEmbedding=FakeEngine)
    try:
        embedder = FastEmbedEmbedder("fake-bge")
        assert embedder.dim == 3 and embedder.key == "fastembed:fake-bge:3"
        assert np.allclose(embedder.embed(["x"])[0], [0.6, 0.8, 0.0])
        assert np.allclose(embedder.embed_query("q"), [0.0, 0.0, 1.0])       # came from query_embed, normalised

        FakeEngine.query_embed = property(lambda self: (_ for _ in ()).throw(AttributeError()))
        del FakeEngine.query_embed
        assert np.allclose(FastEmbedEmbedder("fake-bge").embed_query("q"), [0.6, 0.8, 0.0])   # older versions: plain embed
    finally:
        del sys.modules["fastembed"]


def test_the_real_fastembed_model_loads_and_embeds():
    try:
        import fastembed  # noqa: F401
    except ImportError:
        raise Skip("fastembed is not installed on this machine")
    try:
        embedder = FastEmbedEmbedder("BAAI/bge-small-en-v1.5")
    except Exception as exc:
        raise Skip(f"the model could not be downloaded here ({type(exc).__name__})")
    vectors = embedder.embed(["collision excess", "rental car cover"])
    assert vectors.shape == (2, embedder.dim) and np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4)


# ---------------------------------------------------------------- vector store

def unit(*rows):
    m = np.array(rows, dtype=np.float32)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def test_numpy_search_returns_the_closest_vectors_best_first():
    vectors = unit([1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1])
    results = VectorStore(vectors, force_numpy=True).search(unit([1, 0.2, 0])[0], 3)
    assert [i for i, _ in results] == [0, 2, 1] and results[0][1] > results[1][1] > results[2][1]
    assert abs(results[0][1] - float(vectors[0] @ unit([1, 0.2, 0])[0])) < 1e-6


def test_store_edge_cases():
    empty = VectorStore(np.zeros((0, 8), dtype=np.float32), force_numpy=True)
    assert empty.search(np.zeros(8, dtype=np.float32), 5) == [] and empty.count == 0
    one = VectorStore(unit([1, 0]), force_numpy=True)
    assert len(one.search(np.array([1, 0], dtype=np.float32), 10)) == 1          # k larger than the store
    assert one.search(np.array([1, 0], dtype=np.float32), 0) == [] and one.backend == "numpy"


def test_the_faiss_code_path_returns_what_numpy_returns():
    calls = {}

    class FakeIndexFlatIP:
        def __init__(self, dim):
            calls["dim"] = dim

        def add(self, vectors):
            self.v = vectors

        def search(self, q, k):
            sims = self.v @ q[0]
            order = np.argsort(-sims)[:k]
            return np.array([sims[order]]), np.array([order])

    original = store.faiss
    store.faiss = types.SimpleNamespace(IndexFlatIP=FakeIndexFlatIP)
    try:
        vectors = unit([1, 0, 0], [0, 1, 0], [1, 1, 0])
        faiss_store = VectorStore(vectors)
        assert faiss_store.backend == "faiss" and calls["dim"] == 3
        query = unit([1, 0.1, 0])[0]
        assert faiss_store.search(query, 2) == VectorStore(vectors, force_numpy=True).search(query, 2)
    finally:
        store.faiss = original


def test_real_faiss_agrees_with_numpy():
    if store.faiss is None:
        raise Skip("faiss-cpu is not installed on this machine")
    rng = np.random.default_rng(3)
    vectors = rng.normal(size=(40, 16)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    query = vectors[7] + 0.05
    real = VectorStore(vectors)
    assert real.backend == "faiss"
    assert [i for i, _ in real.search(query, 5)] == [i for i, _ in VectorStore(vectors, force_numpy=True).search(query, 5)]
