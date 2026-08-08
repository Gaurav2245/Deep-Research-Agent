import uuid
from datetime import datetime
from types import SimpleNamespace

from database import Conversation, ConversationState, Message


def make_conversation(**overrides) -> Conversation:
    defaults = dict(
        id=uuid.uuid4(),
        title="Test Chat",
        user_id="default_user",
        message_count=0,
        research_count=0,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    defaults.update(overrides)
    conv = Conversation(**defaults)
    conv.messages = []
    return conv


def test_create_conversation(client, fake_session):
    r = client.post("/api/v1/conversations", json={"title": "My Chat"})

    assert r.status_code == 200
    body = r.json()
    assert body["title"] == "My Chat"
    assert ConversationState in fake_session.data
    assert len(fake_session.data[ConversationState]) == 1


def test_list_conversations(client, fake_session):
    fake_session.data[Conversation] = [make_conversation(), make_conversation()]
    r = client.get("/api/v1/conversations")
    assert r.status_code == 200
    assert len(r.json()) == 2


def test_get_conversation_not_found(client, fake_session):
    fake_session.data[Conversation] = []
    r = client.get(f"/api/v1/conversations/{uuid.uuid4()}")
    assert r.status_code == 404


def test_get_conversation_with_messages(client, fake_session):
    conv = make_conversation()
    conv.messages = [
        Message(id=uuid.uuid4(), conversation_id=conv.id, role="user", content="hi", created_at=datetime.utcnow()),
    ]
    fake_session.data[Conversation] = [conv]

    r = client.get(f"/api/v1/conversations/{conv.id}")

    assert r.status_code == 200
    body = r.json()
    assert len(body["messages"]) == 1
    assert body["messages"][0]["content"] == "hi"


def test_query_conversation_runs_research_and_stores_messages(client, fake_session, monkeypatch):
    conv = make_conversation()
    fake_session.data[Conversation] = [conv]

    fake_result = SimpleNamespace(error=None, final_answer="Paris is the capital of France.")
    monkeypatch.setattr("main.run_research", lambda query: fake_result)

    r = client.post(
        f"/api/v1/conversations/{conv.id}/query",
        json={"query": "capital of France?", "conversation_id": str(conv.id)},
    )

    assert r.status_code == 200
    body = r.json()
    assert body["content"] == "Paris is the capital of France."
    assert conv.message_count == 2
    assert conv.research_count == 1
    # Both user and assistant messages were persisted.
    assert len(fake_session.data.get(Message, [])) == 2


def test_query_conversation_not_found(client, fake_session):
    fake_session.data[Conversation] = []
    r = client.post(
        f"/api/v1/conversations/{uuid.uuid4()}/query",
        json={"query": "hi", "conversation_id": str(uuid.uuid4())},
    )
    assert r.status_code == 404


def test_add_message(client, fake_session):
    conv = make_conversation()
    fake_session.data[Conversation] = [conv]

    r = client.post(
        f"/api/v1/conversations/{conv.id}/messages",
        json={"role": "user", "content": "hello"},
    )

    assert r.status_code == 200
    assert r.json()["content"] == "hello"


def test_get_messages(client, fake_session):
    conv = make_conversation()
    fake_session.data[Conversation] = [conv]
    fake_session.data[Message] = [
        Message(id=uuid.uuid4(), conversation_id=conv.id, role="user", content="hi", created_at=datetime.utcnow()),
    ]

    r = client.get(f"/api/v1/conversations/{conv.id}/messages")

    assert r.status_code == 200
    assert len(r.json()) == 1


def test_update_conversation_title(client, fake_session):
    conv = make_conversation()
    fake_session.data[Conversation] = [conv]

    r = client.put(f"/api/v1/conversations/{conv.id}", params={"title": "Renamed"})

    assert r.status_code == 200
    assert r.json()["title"] == "Renamed"


def test_delete_conversation(client, fake_session):
    conv = make_conversation()
    fake_session.data[Conversation] = [conv]

    r = client.delete(f"/api/v1/conversations/{conv.id}")

    assert r.status_code == 200
    assert r.json()["status"] == "deleted"
