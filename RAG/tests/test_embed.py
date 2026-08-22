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
