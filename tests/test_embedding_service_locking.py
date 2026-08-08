import threading

import database.embedding_service as embedding_service


class FakeSentenceTransformer:
    """Deterministic stand-in that counts how many times it's constructed."""

    instances_created = 0

    def __init__(self, model_name: str):
        FakeSentenceTransformer.instances_created += 1
        self.model_name = model_name


def test_get_shared_model_loads_only_once_under_concurrency(monkeypatch):
    FakeSentenceTransformer.instances_created = 0
    monkeypatch.setattr(embedding_service, "HAS_SENTENCE_TRANSFORMERS", True)
    monkeypatch.setattr(embedding_service, "SentenceTransformer", FakeSentenceTransformer)
    monkeypatch.setattr(embedding_service, "_shared_model", None)

    barrier = threading.Barrier(8)
    results = []

    def worker():
        barrier.wait()  # maximize the chance of a real race
        results.append(embedding_service.get_shared_model())

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert FakeSentenceTransformer.instances_created == 1
    assert all(r is results[0] for r in results)


def test_get_shared_model_returns_none_when_library_missing(monkeypatch):
    monkeypatch.setattr(embedding_service, "HAS_SENTENCE_TRANSFORMERS", False)
    assert embedding_service.get_shared_model() is None
