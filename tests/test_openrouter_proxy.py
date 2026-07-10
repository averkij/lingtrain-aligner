"""Tests for OpenRouter embedding request proxy handling."""

from lingtrain_aligner import aligner


class _Response:
    def __init__(self, data):
        self._data = data

    def json(self):
        return self._data


def _embedding_response():
    return _Response(
        {
            "data": [
                {"index": 0, "embedding": [3.0, 4.0]},
                {"index": 1, "embedding": [0.0, 2.0]},
            ]
        }
    )


def test_openrouter_embeddings_direct_when_proxy_unset(monkeypatch):
    calls = {}

    def fake_post(url, *, headers, json, timeout):
        calls["url"] = url
        calls["headers"] = headers
        calls["json"] = json
        calls["timeout"] = timeout
        return _embedding_response()

    class ForbiddenSession:
        def __init__(self, *args, **kwargs):
            raise AssertionError("requests.Session should not be used without proxy")

    monkeypatch.delenv("OPENROUTER_PROXY_URL", raising=False)
    monkeypatch.setattr("requests.post", fake_post)
    monkeypatch.setattr("requests.Session", ForbiddenSession)

    embeddings = aligner._openrouter_embed_batched(
        ["one", "two"], "test/model", "sk-test"
    )

    assert calls["url"] == "https://openrouter.ai/api/v1/embeddings"
    assert calls["headers"]["Authorization"] == "Bearer sk-test"
    assert calls["json"] == {"model": "test/model", "input": ["one", "two"]}
    assert embeddings == [[3.0, 4.0], [0.0, 2.0]]


def test_openrouter_embeddings_use_configured_proxy(monkeypatch):
    calls = {}

    class FakeSession:
        def __init__(self):
            self.trust_env = True

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, *, headers, json, timeout, proxies):
            calls["url"] = url
            calls["headers"] = headers
            calls["json"] = json
            calls["timeout"] = timeout
            calls["proxies"] = proxies
            calls["trust_env"] = self.trust_env
            return _embedding_response()

    def forbidden_post(*args, **kwargs):
        raise AssertionError("requests.post should not be used with explicit proxy")

    monkeypatch.setenv("OPENROUTER_PROXY_URL", "  http://eu-vps:3128  ")
    monkeypatch.setattr("requests.post", forbidden_post)
    monkeypatch.setattr("requests.Session", FakeSession)

    embeddings = aligner._openrouter_embed_batched(
        ["one", "two"], "test/model", "sk-test"
    )

    assert calls["url"] == "https://openrouter.ai/api/v1/embeddings"
    assert calls["headers"]["Authorization"] == "Bearer sk-test"
    assert calls["json"] == {"model": "test/model", "input": ["one", "two"]}
    assert calls["proxies"] == {
        "http": "http://eu-vps:3128",
        "https": "http://eu-vps:3128",
    }
    assert calls["trust_env"] is False
    assert embeddings == [[3.0, 4.0], [0.0, 2.0]]
