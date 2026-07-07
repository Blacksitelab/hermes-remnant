"""Issue #22 tests: entity extraction quality and over-extraction guardrails.

These run without a live Ollama and cover the upstream extraction filters used
by both the regex fallback path and the LLM typed-entity path.
"""

from __future__ import annotations

from remnant.entity import (
    _COMMON_NOUNS,
    _FUNCTION_WORDS,
    _STOPLIST,
    extract_entities,
)
from remnant.extract import filter_typed_entities

# ===========================================================================
# filter_typed_entities (LLM typed-entity path)
# ===========================================================================


def test_filter_typed_entities_drops_noise_words():
    typed = [
        {"name": "Sven", "type": "person", "aliases": []},
        {"name": "the", "type": None, "aliases": []},  # article
        {"name": "And", "type": None, "aliases": []},  # function word
        {"name": "people", "type": "concept", "aliases": []},  # common noun
        {"name": "Monday", "type": "place", "aliases": []},  # stoplist date
        {"name": "API", "type": "service", "aliases": []},  # stoplist tech noun
        {"name": "no", "type": None, "aliases": []},  # short noise
    ]
    out = filter_typed_entities(typed)
    names = {e["name"] for e in out}
    assert names == {"Sven"}


def test_filter_typed_entities_keeps_explicit_remnant():
    """The system's own name is kept when the LLM explicitly names it."""
    out = filter_typed_entities(
        [{"name": "Remnant", "type": "service", "aliases": []}]
    )
    assert [e["name"] for e in out] == ["Remnant"]


def test_filter_typed_entities_deduplicates_case_insensitive():
    out = filter_typed_entities(
        [
            {"name": "Sven", "type": "person", "aliases": []},
            {"name": "sven", "type": "person", "aliases": []},
            {"name": "SVEN", "type": "person", "aliases": []},
        ]
    )
    assert len(out) == 1
    assert out[0]["name"] == "Sven"


def test_filter_typed_entities_caps_at_15_and_keeps_salient():
    typed = [
        {"name": f"Entity{i}", "type": "concept", "aliases": []}
        for i in range(20)
    ]
    # Give a couple of entries extra salience via multi-word names and aliases.
    typed[5]["name"] = "Project Alpha"
    typed[5]["aliases"] = ["Alpha Project"]
    typed[12]["name"] = "BlacksiteLab Homelab"
    typed[12]["aliases"] = ["BSL"]
    out = filter_typed_entities(typed)
    assert len(out) == 15
    names = {e["name"] for e in out}
    assert "Project Alpha" in names
    assert "BlacksiteLab Homelab" in names


# ===========================================================================
# extract_entities (regex fallback path)
# ===========================================================================


def test_extract_entities_finds_expected_proper_nouns():
    ents = extract_entities("Sven prefers dark mode for the Proxmox homelab")
    names = {e["name"] for e in ents}
    assert "Sven" in names
    assert "Proxmox" in names


def test_extract_entities_drops_common_nouns():
    ents = extract_entities(
        "People say that Time is money. Year after year, Work continues."
    )
    names = {e["name"] for e in ents}
    for noise in ("People", "Time", "Money", "Year", "Work"):
        assert noise not in names, f"{noise!r} is a common noun and must be dropped"


def test_extract_entities_drops_capitalized_function_words():
    ents = extract_entities("And Or But For To Of In are not real entities")
    names = {e["name"] for e in ents}
    for noise in ("And", "Or", "But", "For", "To", "Of", "In"):
        assert noise not in names, f"{noise!r} is a function word and must be dropped"


def test_extract_entities_drops_stoplist_items():
    ents = extract_entities("Sven visited New Zealand on Monday morning")
    names = {e["name"] for e in ents}
    assert "Sven" in names
    assert "New Zealand" not in names
    assert "Monday" not in names


def test_extract_entities_keeps_multiword_system_names():
    ents = extract_entities("The Proxmox Server runs in the homelab")
    names = {e["name"] for e in ents}
    assert "Proxmox Server" in names
    assert "Server" not in names  # bare generic tech noun


def test_extract_entities_suppresses_substring_entities():
    ents = extract_entities("Project Alpha is the focus. Alpha is just a codename.")
    names = {e["name"] for e in ents}
    assert "Project Alpha" in names
    assert "Alpha" not in names, "bare substring of a longer entity should be dropped"


def test_extract_entities_ranks_by_salience():
    ents = extract_entities(
        "Sven works with Proxmox. Sven also likes Proxmox. Alice visited once."
    )
    names = [e["name"] for e in ents]
    assert "Sven" in names[:2]
    assert "Proxmox" in names[:2]
    assert "Alice" in names


def test_extract_entities_default_cap_is_15():
    text = "; ".join(
        f"{name} is a project member"
        for name in [
            "Alpha", "Bravo", "Charlie", "Delta", "Echo",
            "Foxtrot", "Golf", "Hotel", "Ivan", "Juliet",
            "Kilo", "Lima", "Mike", "Nash", "Oscar",
            "Papa", "Quebec", "Romeo", "Sierra", "Tango",
        ]
    )
    ents = extract_entities(text)
    assert 0 < len(ents) <= 15
    # All 20 names are proper nouns and survive filtering, so the cap is the
    # only reason we do not see 20.
    assert len(ents) == 15


def test_extract_entities_max_entities_zero_returns_all():
    text = "; ".join(
        f"{name} is a project member"
        for name in [
            "Alpha", "Bravo", "Charlie", "Delta", "Echo",
            "Foxtrot", "Golf", "Hotel", "Ivan", "Juliet",
            "Kilo", "Lima", "Mike", "Nash", "Oscar",
            "Papa", "Quebec", "Romeo", "Sierra", "Tango",
        ]
    )
    ents = extract_entities(text, max_entities=0)
    assert len(ents) == 20


def test_extract_entities_respects_custom_max_entities():
    text = "; ".join(
        f"{name} is a project member"
        for name in [
            "Alpha", "Bravo", "Charlie", "Delta", "Echo",
            "Foxtrot", "Golf", "Hotel", "Ivan", "Juliet",
        ]
    )
    ents = extract_entities(text, max_entities=3)
    assert len(ents) == 3


def test_entity_stoplists_have_no_overlap_with_real_names():
    """Sanity check that our stoplists do not swallow the canonical test
    entities used throughout the suite."""
    real_names = {"sven", "proxmox", "alice smith", "project alpha", "homelab"}
    for name in real_names:
        assert name not in _STOPLIST, f"{name!r} must not be in _STOPLIST"
        assert name not in _COMMON_NOUNS, f"{name!r} must not be in _COMMON_NOUNS"
        assert name not in _FUNCTION_WORDS, f"{name!r} must not be in _FUNCTION_WORDS"


def test_filter_typed_entities_drops_generic_response_words():
    """Capitalized response/sentence-starter words are not durable entities."""
    out = filter_typed_entities([
        {"name": "Sure", "type": None, "aliases": []},
        {"name": "Let", "type": None, "aliases": []},
        {"name": "Great", "type": "concept", "aliases": []},
        {"name": "Proxmox", "type": "service", "aliases": []},
    ])
    assert [e["name"] for e in out] == ["Proxmox"]


def test_extract_entities_drops_capitalized_response_words():
    """Regex path must drop sentence-initial generic words like Sure/Let."""
    ents = extract_entities("Sure, I can help. Let me check Proxmox.")
    names = {e["name"] for e in ents}
    assert "Sure" not in names
    assert "Let" not in names
    assert "Proxmox" in names
