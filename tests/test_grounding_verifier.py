from database.grounding_verifier import GroundingVerifier


class FakeEmbeddingService:
    """Deterministic fake: tracks call counts so tests can assert batching behavior."""

    def __init__(self):
        self.embed_text_calls = 0
        self.embed_texts_calls = 0

    def embed_text(self, text):
        self.embed_text_calls += 1
        return [float(len(text) % 7), 1.0]

    def embed_texts(self, texts):
        self.embed_texts_calls += 1
        return [[float(len(t) % 7), 1.0] for t in texts]

    def cosine_similarity(self, a, b):
        return 1.0 if a == b else 0.5


def test_verify_answer_batches_source_sentence_embeddings():
    """
    Regression test: semantic matching used to re-embed every source sentence
    individually for every claim (O(claims x sources x sentences) model calls).
    It must now batch-embed source sentences exactly once per verify_answer()
    call, regardless of how many claims are in the answer.
    """
    fake = FakeEmbeddingService()
    verifier = GroundingVerifier(embedding_service=fake)

    answer = "Paris is the capital of France. The Eiffel Tower is in Paris. It was built in 1889."
    sources = [
        {"url": "https://a.com", "content": "Paris is the capital of France. The Eiffel Tower is a famous landmark."},
        {"url": "https://b.com", "content": "It was built in 1889 for the World Fair. Many tourists visit each year."},
    ]

    result = verifier.verify_answer(answer=answer, sources=sources, query="Tell me about Paris")

    assert fake.embed_texts_calls == 1  # source sentences embedded in a single batch
    assert result.total_claims == 3


def test_verify_answer_with_no_embedding_service_skips_semantic_match():
    verifier = GroundingVerifier(embedding_service=None)
    result = verifier.verify_answer(
        answer="Something not in any source.",
        sources=[{"url": "https://a.com", "content": "Unrelated content."}],
        query="q",
    )
    assert result.total_claims == 1


def test_verify_answer_handles_sources_with_no_content():
    fake = FakeEmbeddingService()
    verifier = GroundingVerifier(embedding_service=fake)
    result = verifier.verify_answer(answer="A claim here.", sources=[{"url": "https://a.com"}], query="q")
    assert result.total_claims == 1
    assert fake.embed_texts_calls == 0  # no source sentences to embed
