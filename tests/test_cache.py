from bench.cache import Cache, request_key


def test_key_is_order_independent_but_content_sensitive():
    msgs = [{"role": "user", "content": "hi"}]
    a = request_key("m", msgs, temperature=0.2, max_tokens=10)
    b = request_key("m", msgs, max_tokens=10, temperature=0.2)
    assert a == b, "kwarg order must not change the key"

    assert request_key("m", msgs, temperature=0.3) != a
    assert request_key("other", msgs, temperature=0.2, max_tokens=10) != a
    assert request_key("m", [{"role": "user", "content": "ho"}],
                       temperature=0.2, max_tokens=10) != a


def test_roundtrip_and_stats():
    c = Cache()
    assert c.get("missing") is None
    c.put("k1", "model-a", {"q": 1}, {"choices": [{"message": {"content": "x"}}]})
    assert c.get("k1")["choices"][0]["message"]["content"] == "x"
    c.put("k2", "model-a", {}, {})
    c.put("k3", "model-b", {}, {})
    assert c.stats() == {"model-a": 2, "model-b": 1}


def test_put_is_idempotent():
    c = Cache()
    c.put("k", "m", {}, {"v": 1})
    c.put("k", "m", {}, {"v": 2})
    assert c.get("k") == {"v": 2}
    assert c.stats() == {"m": 1}
