from __future__ import annotations

import rag.embed as embed


def test_generate_can_request_short_non_thinking_structured_json(monkeypatch):
    calls = []
    schema = {"type": "object", "required": ["queries"]}

    def fake_post(host, path, payload, timeout):
        calls.append((host, path, payload, timeout))
        return {"response": '{"queries": []}'}

    monkeypatch.setattr(embed, "_post", fake_post)

    result = embed.generate(
        "http://127.0.0.1:11434",
        "qwen3.5:0.8b",
        "prompt",
        timeout=12.0,
        json_schema=schema,
        think=False,
        num_predict=320,
    )

    assert result == '{"queries": []}'
    assert calls == [
        (
            "http://127.0.0.1:11434",
            "/api/generate",
            {
                "model": "qwen3.5:0.8b",
                "prompt": "prompt",
                "stream": False,
                "format": schema,
                "think": False,
                "options": {"temperature": 0, "num_predict": 320},
            },
            12.0,
        )
    ]


def test_embed_texts_keeps_order_with_parallel_batches(monkeypatch):
    import threading
    import time

    seen_threads = set()

    def fake_post(host, path, payload, timeout):
        seen_threads.add(threading.get_ident())
        # Reverse completion order: the first batch finishes last.
        time.sleep(0.02 if payload["input"][0] == "t0" else 0.0)
        return {"embeddings": [[float(t[1:])] for t in payload["input"]]}

    monkeypatch.setattr(embed, "_post", fake_post)
    texts = [f"t{i}" for i in range(10)]

    vectors = embed.embed_texts("h", "m", texts, batch=3, workers=4)

    assert vectors == [[float(i)] for i in range(10)]
    assert len(seen_threads) > 1
