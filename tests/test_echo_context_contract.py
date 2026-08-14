"""Structured context contract used by Echo and recall evaluation."""

from remnant.context import compile_context_details


def test_compiled_context_reports_exact_rendered_and_omitted_ids():
    compiled = compile_context_details(
        [
            {"id": "m-current", "content": "Sven likes tea", "visibility": "private"},
            {"id": "m-long", "content": "x" * 1000, "visibility": "private"},
        ],
        token_budget=80,
    )

    rendered_ids = {item.memory_id for item in compiled.items}
    assert rendered_ids
    assert rendered_ids.isdisjoint(compiled.omitted_ids)
    assert set(rendered_ids) | set(compiled.omitted_ids) == {"m-current", "m-long"}
    assert compiled.token_count <= 80
    assert all(item.rendered_hash for item in compiled.items)


def test_compiled_context_preserves_pending_source_turn_identity():
    compiled = compile_context_details(
        [
            {
                "id": "pending-42",
                "content": "The printer is in the office.",
                "pending": True,
                "visibility": "private",
            }
        ]
    )

    assert len(compiled.items) == 1
    item = compiled.items[0]
    assert item.item_kind == "pending"
    assert item.source_turn_id == 42
    assert item.memory_id == "pending-42"
