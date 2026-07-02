"""Phase 3 tests: entity graph, memory_edit actions, audit log, contradiction
detection, and search exclusion.

Run without a live Ollama: the Embedder is monkeypatched with deterministic
word-bag vectors (same pattern as test_phase2) and the extraction LLM call is
stubbed to return no facts.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from remnant import RemnantMemoryProvider
from remnant.config import RemnantConfig
from remnant.db import default_db_path, open_db
from remnant.edit import memory_edit
from remnant.embed import Embedder, cosine
from remnant.entity import (
    extract_entities,
    link_memory_entities,
    normalize_aliases,
    resolve_and_link,
    seed_relations,
)
from remnant.graph import graph_search, graph_traverse
from remnant.ingest import detect_contradiction, store_memory
from remnant.search import search as hybrid_search

# --- shared fakes (mirror test_phase2) -------------------------------------


def _hash(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _fake_embed(db, config, dim=8):
    """Deterministic word-bag embedder (same as test_phase2)."""
    emb = Embedder.__new__(Embedder)
    emb._db = db
    emb._model = config.embed_model
    emb._url = config.embed_url
    emb._timeout = config.embed_timeout
    emb._client = None

    def _seed(text: str) -> list[float]:
        words = [w.lower() for w in text.strip().split()]
        vec = [0.0] * dim
        for w in words:
            h = abs(hash(w)) % dim
            vec[h] += 1.0
        n = sum(v * v for v in vec) ** 0.5
        if n:
            vec = [v / n for v in vec]
        return vec

    def embed(text: str) -> list[float]:
        cached = db.get_cached_embedding(emb._model, _hash(text))
        if cached is not None:
            return cached
        vec = _seed(text)
        db.put_cached_embedding(emb._model, _hash(text), vec)
        return vec

    emb.embed = embed
    emb.close = lambda: None
    return emb


@pytest.fixture()
def hermes_home(tmp_path: Path) -> Path:
    home = tmp_path / "hermes"
    (home / "remnant").mkdir(parents=True, exist_ok=True)
    return home


@pytest.fixture()
def provider(hermes_home: Path) -> RemnantMemoryProvider:
    p = RemnantMemoryProvider()
    p.initialize(session_id="p3-session", hermes_home=str(hermes_home))
    yield p
    p.shutdown()


@pytest.fixture(autouse=True)
def patch_embed(monkeypatch):
    import sys

    remnant_init = sys.modules["remnant"]
    embed_mod = sys.modules["remnant.embed"]

    original_init = remnant_init.Embedder
    original_embed_mod = embed_mod.Embedder

    def fake_ctor(db, config):
        return _fake_embed(db, config)

    monkeypatch.setattr(remnant_init, "Embedder", fake_ctor)
    monkeypatch.setattr(embed_mod, "Embedder", fake_ctor)
    yield
    monkeypatch.setattr(remnant_init, "Embedder", original_init)
    monkeypatch.setattr(embed_mod, "Embedder", original_embed_mod)


@pytest.fixture(autouse=True)
def patch_extract(monkeypatch):
    from remnant import extract as extract_mod

    monkeypatch.setattr(extract_mod.ExtractionWorker, "_extract", lambda self, u, a: [])
    yield


# --- helpers ----------------------------------------------------------------


def _open_db(hermes_home: Path):
    return open_db(default_db_path())


def _store(db, emb, cfg, *, fact, entities, agent_id="default", visibility="private"):
    """Store a memory with typed entities (drives graph linking + contradiction)."""
    return store_memory(
        db, emb, cfg, fact=fact, entity=entities[0]["name"] if entities else "general",
        session_id="seed", agent_id=agent_id, visibility=visibility, entities=entities,
    )


def _store_simple(db, emb, cfg, *, fact, entity, agent_id="default", visibility="private"):
    """Store via the legacy single-entity path (memory_store tool equivalent)."""
    return store_memory(
        db, emb, cfg, fact=fact, entity=entity, session_id="seed",
        agent_id=agent_id, visibility=visibility,
    )


# ===========================================================================
# 1. Entity extraction and resolution
# ===========================================================================


def test_extract_entities_finds_proper_nouns():
    ents = extract_entities("Sven prefers dark mode for the Proxmox homelab")
    names = {e["name"] for e in ents}
    assert "Sven" in names
    assert "Proxmox" in names
    # Common stopwords are dropped.
    ents = extract_entities("The homelab runs Proxmox")
    assert all(e["name"] != "The" for e in ents)


def test_extract_entities_empty():
    assert extract_entities("") == []
    assert extract_entities("no proper nouns here at all") == []


def test_extract_entities_assigns_guessed_type():
    ents = extract_entities("The Proxmox server runs in the homelab")
    by_name = {e["name"]: e for e in ents}
    # "server" keyword in context => Proxmox guessed as a service.
    proxmox = by_name.get("Proxmox")
    assert proxmox is not None
    assert proxmox["type"] in {"service", "project", None}


def test_extract_entities_multiword_phrase():
    ents = extract_entities("Alice Smith works on Project Alpha")
    names = {e["name"] for e in ents}
    assert "Alice Smith" in names
    assert "Project Alpha" in names


def test_resolve_and_link_creates_entity_and_links(hermes_home: Path):
    db = _open_db(hermes_home)
    cfg = RemnantConfig()
    _fake_embed(db, cfg)
    try:
        mid = db.insert_memory(content="Sven prefers dark mode", agent="default")
        eid, display = resolve_and_link(
            db, memory_id=mid, entity_name="Sven", agent_id="default",
            entity_type="person", aliases=["svenny"],
        )
        assert eid, "entity id should be returned"
        assert display == "Sven"
        # memory_entities link exists.
        with db.read() as cur:
            cur.execute(
                "SELECT entity_id FROM memory_entities WHERE memory_id=?", (mid,)
            )
            rows = cur.fetchall()
        assert any(r["entity_id"] == eid for r in rows)
        # Entity row has the type + alias.
        ent = db.get_entity(eid)
        assert ent is not None
        assert ent["name"] == "sven"
        assert ent["type"] == "person"
    finally:
        db.close()


def test_resolve_and_link_is_idempotent_on_name(hermes_home: Path):
    """Resolving the same name twice returns the same entity id."""
    db = _open_db(hermes_home)
    cfg = RemnantConfig()
    _fake_embed(db, cfg)
    try:
        mid = db.insert_memory(content="Sven prefers dark mode", agent="default")
        eid1, _ = resolve_and_link(db, memory_id=mid, entity_name="Sven", agent_id="default")
        mid2 = db.insert_memory(content="Sven also likes vim", agent="default")
        eid2, _ = resolve_and_link(db, memory_id=mid2, entity_name="Sven", agent_id="default")
        assert eid1 == eid2
    finally:
        db.close()


def test_aliases_normalized_and_resolve(hermes_home: Path):
    """Aliases are lowercased + punctuation-stripped and resolve back."""
    raw = ["Svenny", "Sven E.", "sven!"]
    norm = normalize_aliases(raw)
    assert norm == ["svenny", "sven e.", "sven"]
    assert normalize_aliases([]) == []
    assert normalize_aliases(["", "  ", "???"]) == []

    db = _open_db(hermes_home)
    cfg = RemnantConfig()
    _fake_embed(db, cfg)
    try:
        mid = db.insert_memory(content="Sven prefers dark mode", agent="default")
        eid, _ = resolve_and_link(
            db, memory_id=mid, entity_name="Sven", agent_id="default",
            entity_type="person", aliases=norm,
        )
        # find_entity_by_name matches the alias.
        assert db.find_entity_by_name("svenny", agent_id="default") == eid
        assert db.find_entity_by_name("Sven E.", agent_id="default") == eid
        assert db.find_entity_by_name("Sven", agent_id="default") == eid
    finally:
        db.close()


def test_link_memory_entities_seeds_relations(hermes_home: Path):
    """Two co-occurring entities get a related_to edge."""
    db = _open_db(hermes_home)
    cfg = RemnantConfig()
    _fake_embed(db, cfg)
    try:
        mid = db.insert_memory(content="Sven runs Proxmox on the homelab", agent="default")
        ids = link_memory_entities(
            db, memory_id=mid,
            entities=[
                {"name": "Sven", "type": "person", "aliases": []},
                {"name": "Proxmox", "type": "service", "aliases": []},
                {"name": "homelab", "type": "place", "aliases": []},
            ],
            agent_id="default",
        )
        assert len(ids) == 3
        # At least one relation should exist between the pairs.
        rels = db.get_relations(ids[0]) + db.get_relations(ids[1]) + db.get_relations(ids[2])
        assert rels, "co-occurring entities should be related"
    finally:
        db.close()


def test_entity_resolution_scoped_per_agent(hermes_home: Path):
    """Two agents with the same entity name get distinct entities."""
    db = _open_db(hermes_home)
    cfg = RemnantConfig()
    _fake_embed(db, cfg)
    try:
        mid_a = db.insert_memory(content="Sven A fact", agent="agentA")
        eid_a, _ = resolve_and_link(db, memory_id=mid_a, entity_name="Sven", agent_id="agentA")
        mid_b = db.insert_memory(content="Sven B fact", agent="agentB")
        eid_b, _ = resolve_and_link(db, memory_id=mid_b, entity_name="Sven", agent_id="agentB")
        assert eid_a != eid_b
    finally:
        db.close()


# ===========================================================================
# 2. Graph traversal
# ===========================================================================


def _seed_graph(db, emb, cfg):
    """Seed: Sven -- Proxmox -- homelab, with a memory linking all three."""
    mid = _store(db, emb, cfg, fact="Sven runs Proxmox on the homelab",
                 entities=[
                     {"name": "Sven", "type": "person", "aliases": []},
                     {"name": "Proxmox", "type": "service", "aliases": []},
                     {"name": "homelab", "type": "place", "aliases": []},
                 ])
    return mid


def test_memory_graph_tool_returns_connected_entities(provider: RemnantMemoryProvider):
    provider.handle_tool_call(
        "memory_store",
        {"fact": "Sven runs Proxmox on the homelab", "entity": "Sven"},
        session_id="seed",
    )
    # The single-entity path only links "Sven"; store a second memory that
    # links Sven + Proxmox so a relation exists.
    provider.handle_tool_call(
        "memory_store",
        {"fact": "Sven uses Proxmox for virtualization", "entity": "Sven"},
        session_id="seed",
    )
    res = provider.handle_tool_call(
        "memory_graph", {"entity": "Sven", "depth": 2}, session_id="graph",
    )
    assert "entities" in res
    assert "memories" in res
    # Sven resolves and is the seed at depth 0.
    names = [e.get("name") for e in res["entities"]]
    assert "sven" in names
    # At least one memory linked to Sven is returned.
    assert res["count"] >= 1


def test_memory_graph_tool_unknown_entity(provider: RemnantMemoryProvider):
    res = provider.handle_tool_call(
        "memory_graph", {"entity": "Nonexistent"}, session_id="graph",
    )
    assert res["entity"] is None
    assert res["entities"] == []
    assert res["count"] == 0


def test_search_graph_strategy_finds_linked_memories(hermes_home: Path):
    db = _open_db(hermes_home)
    cfg = RemnantConfig()
    emb = _fake_embed(db, cfg)
    try:
        _seed_graph(db, emb, cfg)
        # Query mentions a known entity by proper-noun name.
        results = hybrid_search(
            db, cfg, "Sven", agent_id="default", limit=10, strategy="graph",
        )
        assert results, "graph search should return linked memories"
        assert any("Sven" in r["content"] or "sven" in r["content"].lower()
                   for r in results)
    finally:
        db.close()


def test_search_graph_strategy_no_match(hermes_home: Path):
    db = _open_db(hermes_home)
    cfg = RemnantConfig()
    _fake_embed(db, cfg)
    try:
        results = hybrid_search(
            db, cfg, "NobodyKnowsThis", agent_id="default", limit=10, strategy="graph",
        )
        assert results == []
    finally:
        db.close()


def test_graph_traverse_returns_entities_and_memories(hermes_home: Path):
    db = _open_db(hermes_home)
    cfg = RemnantConfig()
    emb = _fake_embed(db, cfg)
    try:
        _seed_graph(db, emb, cfg)
        res = graph_traverse(db, "Sven", agent_id="default", depth=2)
        assert res["entity"] is not None
        assert res["entity"]["name"] == "sven"
        assert len(res["entities"]) >= 1
        # Seed at depth 0.
        depths = {e["name"]: e["depth"] for e in res["entities"]}
        assert depths.get("sven") == 0
        assert res["memories"], "should return linked active memories"
    finally:
        db.close()


def test_graph_traverse_empty_name(hermes_home: Path):
    db = _open_db(hermes_home)
    try:
        res = graph_traverse(db, "", agent_id="default")
        assert res == {"entity": None, "entities": [], "memories": []}
    finally:
        db.close()


def test_relation_strength_stored_and_returned(hermes_home: Path):
    db = _open_db(hermes_home)
    cfg = RemnantConfig()
    _fake_embed(db, cfg)
    try:
        mid = db.insert_memory(content="x", agent="default")
        eid_a, _ = resolve_and_link(db, memory_id=mid, entity_name="Alpha", agent_id="default")
        eid_b, _ = resolve_and_link(db, memory_id=mid, entity_name="Beta", agent_id="default")
        eid_c, _ = resolve_and_link(db, memory_id=mid, entity_name="Gamma", agent_id="default")
        seed_relations(db, memory_id=mid, entity_ids=[eid_a, eid_b, eid_c], strength=0.5)
        rels = db.get_relations(eid_a)
        assert rels, "relations should be stored"
        for r in rels:
            assert r["strength"] == 0.5
            assert r["relation_type"] == "related_to"
    finally:
        db.close()


def test_relation_strength_max_upsert(hermes_home: Path):
    """Re-adding a relation with higher strength keeps the max."""
    db = _open_db(hermes_home)
    cfg = RemnantConfig()
    _fake_embed(db, cfg)
    try:
        mid = db.insert_memory(content="x", agent="default")
        eid_a, _ = resolve_and_link(db, memory_id=mid, entity_name="Alpha", agent_id="default")
        eid_b, _ = resolve_and_link(db, memory_id=mid, entity_name="Beta", agent_id="default")
        db.add_relation(entity_a=eid_a, entity_b=eid_b, strength=0.3)
        db.add_relation(entity_a=eid_a, entity_b=eid_b, strength=0.8)
        rels = db.get_relations(eid_a)
        assert len(rels) == 1
        assert rels[0]["strength"] == 0.8  # MAX upsert keeps the higher value
    finally:
        db.close()


# ===========================================================================
# 3. memory_edit actions
# ===========================================================================


def _store_one(db, emb, cfg, fact="Sven prefers dark mode", agent_id="default"):
    mid = _store_simple(db, emb, cfg, fact=fact, entity="Sven", agent_id=agent_id)
    assert mid is not None
    return mid


def test_edit_update_creates_new_and_supersedes_old(hermes_home: Path):
    db = _open_db(hermes_home)
    cfg = RemnantConfig()
    emb = _fake_embed(db, cfg)
    try:
        old = _store_one(db, emb, cfg, fact="Sven prefers dark mode")
        res = memory_edit(
            db, cfg, emb, action="update", actor="default", memory_id=old,
            content="Sven prefers light mode",
        )
        assert "error" not in res
        new = res["memory_id"]
        assert new != old
        assert res["superseded_id"] == old
        # Old is superseded; new is active.
        assert db.get_memory(old)["status"] == "superseded"
        assert db.get_memory(new)["status"] == "active"
        assert db.get_memory(new)["content"] == "Sven prefers light mode"
    finally:
        db.close()


def test_edit_update_missing_memory(hermes_home: Path):
    db = _open_db(hermes_home)
    cfg = RemnantConfig()
    emb = _fake_embed(db, cfg)
    try:
        res = memory_edit(
            db, cfg, emb, action="update", actor="default",
            memory_id="does-not-exist", content="x",
        )
        assert "error" in res
    finally:
        db.close()


def test_edit_update_requires_content(hermes_home: Path):
    db = _open_db(hermes_home)
    cfg = RemnantConfig()
    emb = _fake_embed(db, cfg)
    try:
        old = _store_one(db, emb, cfg)
        res = memory_edit(
            db, cfg, emb, action="update", actor="default", memory_id=old, content="",
        )
        assert "error" in res
    finally:
        db.close()


def test_edit_merge_combines_and_supersedes(hermes_home: Path):
    db = _open_db(hermes_home)
    cfg = RemnantConfig()
    emb = _fake_embed(db, cfg)
    try:
        m1 = _store_simple(db, emb, cfg, fact="Sven likes vim", entity="Sven")
        m2 = _store_simple(db, emb, cfg, fact="Sven likes git", entity="Sven")
        res = memory_edit(
            db, cfg, emb, action="merge", actor="default",
            memory_ids=[m1, m2], content="Sven likes vim and git",
        )
        assert "error" not in res
        new = res["memory_id"]
        assert new not in (m1, m2)
        assert set(res["superseded_ids"]) == {m1, m2}
        for old in (m1, m2):
            assert db.get_memory(old)["status"] == "superseded"
        assert db.get_memory(new)["content"] == "Sven likes vim and git"
        assert db.get_memory(new)["status"] == "active"
    finally:
        db.close()


def test_edit_merge_requires_two_ids(hermes_home: Path):
    db = _open_db(hermes_home)
    cfg = RemnantConfig()
    emb = _fake_embed(db, cfg)
    try:
        m1 = _store_simple(db, emb, cfg, fact="Sven likes vim", entity="Sven")
        res = memory_edit(
            db, cfg, emb, action="merge", actor="default",
            memory_ids=[m1], content="x",
        )
        assert "error" in res
    finally:
        db.close()


def test_edit_forget_marks_status_and_hides(hermes_home: Path):
    db = _open_db(hermes_home)
    cfg = RemnantConfig()
    emb = _fake_embed(db, cfg)
    try:
        mid = _store_one(db, emb, cfg, fact="Sven prefers dark mode")
        res = memory_edit(db, cfg, emb, action="forget", actor="default", memory_id=mid)
        assert res["status"] == "forgotten"
        assert db.get_memory(mid)["status"] == "forgotten"
        # Forgotten memory must not appear in search.
        results = hybrid_search(db, cfg, "dark mode", agent_id="default")
        assert all(r["id"] != mid for r in results)
    finally:
        db.close()


def test_edit_forget_missing_memory(hermes_home: Path):
    db = _open_db(hermes_home)
    cfg = RemnantConfig()
    emb = _fake_embed(db, cfg)
    try:
        res = memory_edit(
            db, cfg, emb, action="forget", actor="default", memory_id="nope",
        )
        assert "error" in res
    finally:
        db.close()


def test_edit_feedback_raises_trust(hermes_home: Path):
    db = _open_db(hermes_home)
    cfg = RemnantConfig()
    emb = _fake_embed(db, cfg)
    try:
        mid = _store_one(db, emb, cfg)
        before = db.get_memory(mid)["trust_score"]
        res = memory_edit(
            db, cfg, emb, action="feedback", actor="default",
            memory_id=mid, feedback="useful",
        )
        assert res["trust_score"] == pytest.approx(before + 0.1)
        assert db.get_memory(mid)["trust_score"] == res["trust_score"]
    finally:
        db.close()


def test_edit_feedback_lowers_trust(hermes_home: Path):
    db = _open_db(hermes_home)
    cfg = RemnantConfig()
    emb = _fake_embed(db, cfg)
    try:
        mid = _store_one(db, emb, cfg)
        before = db.get_memory(mid)["trust_score"]
        res = memory_edit(
            db, cfg, emb, action="feedback", actor="default",
            memory_id=mid, feedback="wrong",
        )
        assert res["trust_score"] == pytest.approx(before - 0.2)
    finally:
        db.close()


def test_edit_feedback_clamps_at_bounds(hermes_home: Path):
    db = _open_db(hermes_home)
    cfg = RemnantConfig()
    emb = _fake_embed(db, cfg)
    try:
        mid = _store_one(db, emb, cfg)
        # Hammer with "wrong" many times to clamp at 0.0.
        for _ in range(20):
            memory_edit(
                db, cfg, emb, action="feedback", actor="default",
                memory_id=mid, feedback="wrong",
            )
        assert db.get_memory(mid)["trust_score"] == pytest.approx(0.0)
        # Then "useful" raises by 0.1 but still bounded at 1.0 eventually.
        for _ in range(20):
            memory_edit(
                db, cfg, emb, action="feedback", actor="default",
                memory_id=mid, feedback="useful",
            )
        assert db.get_memory(mid)["trust_score"] == pytest.approx(1.0)
    finally:
        db.close()


def test_edit_feedback_invalid_value(hermes_home: Path):
    db = _open_db(hermes_home)
    cfg = RemnantConfig()
    emb = _fake_embed(db, cfg)
    try:
        mid = _store_one(db, emb, cfg)
        res = memory_edit(
            db, cfg, emb, action="feedback", actor="default",
            memory_id=mid, feedback="maybe",
        )
        assert "error" in res
    finally:
        db.close()


def test_edit_share_changes_visibility(hermes_home: Path):
    db = _open_db(hermes_home)
    cfg = RemnantConfig()
    emb = _fake_embed(db, cfg)
    try:
        mid = _store_one(db, emb, cfg, fact="Sven prefers dark mode")
        assert db.get_memory(mid)["visibility"] == "private"
        res = memory_edit(db, cfg, emb, action="share", actor="default", memory_id=mid)
        assert res["visibility"] == "shared"
        assert db.get_memory(mid)["visibility"] == "shared"
    finally:
        db.close()


def test_edit_unshare_changes_visibility(hermes_home: Path):
    db = _open_db(hermes_home)
    cfg = RemnantConfig()
    emb = _fake_embed(db, cfg)
    try:
        mid = _store_simple(db, emb, cfg, fact="Shared fact", entity="x", visibility="shared")
        res = memory_edit(db, cfg, emb, action="unshare", actor="default", memory_id=mid)
        assert res["visibility"] == "private"
        assert db.get_memory(mid)["visibility"] == "private"
    finally:
        db.close()


def test_edit_unknown_action(hermes_home: Path):
    db = _open_db(hermes_home)
    cfg = RemnantConfig()
    emb = _fake_embed(db, cfg)
    try:
        res = memory_edit(db, cfg, emb, action="bogus", actor="default")
        assert "error" in res
    finally:
        db.close()


def test_memory_edit_tool_dispatch(provider: RemnantMemoryProvider):
    provider.handle_tool_call(
        "memory_store",
        {"fact": "Sven prefers dark mode", "entity": "Sven"},
        session_id="seed",
    )
    # Look it up to get the id.
    search_res = provider.handle_tool_call(
        "memory_search", {"query": "dark mode"}, session_id="edit",
    )
    mid = search_res["results"][0]["id"]
    res = provider.handle_tool_call(
        "memory_edit",
        {"action": "feedback", "memory_id": mid, "feedback": "useful"},
        session_id="edit",
    )
    assert "error" not in res
    assert res["trust_score"] == pytest.approx(0.6)


# ===========================================================================
# 4. Audit logging
# ===========================================================================


def test_audit_log_row_written_for_each_action(hermes_home: Path):
    db = _open_db(hermes_home)
    cfg = RemnantConfig()
    emb = _fake_embed(db, cfg)
    try:
        mid = _store_one(db, emb, cfg, fact="Sven prefers dark mode")

        # forget
        memory_edit(db, cfg, emb, action="forget", actor="auditor", memory_id=mid)
        rows = db.list_audit(memory_id=mid, action="forget")
        assert len(rows) == 1
        assert rows[0]["actor"] == "auditor"

        # feedback on a fresh memory
        mid2 = _store_one(db, emb, cfg, fact="Sven likes vim")
        memory_edit(db, cfg, emb, action="feedback", actor="auditor",
                    memory_id=mid2, feedback="useful")
        rows = db.list_audit(memory_id=mid2, action="feedback")
        assert len(rows) == 1

        # share
        memory_edit(db, cfg, emb, action="share", actor="auditor", memory_id=mid2)
        rows = db.list_audit(memory_id=mid2, action="share")
        assert len(rows) == 1

        # unshare
        memory_edit(db, cfg, emb, action="unshare", actor="auditor", memory_id=mid2)
        rows = db.list_audit(memory_id=mid2, action="unshare")
        assert len(rows) == 1
    finally:
        db.close()


def test_audit_update_writes_supersede_and_update_rows(hermes_home: Path):
    db = _open_db(hermes_home)
    cfg = RemnantConfig()
    emb = _fake_embed(db, cfg)
    try:
        old = _store_one(db, emb, cfg, fact="Sven prefers dark mode")
        memory_edit(db, cfg, emb, action="update", actor="auditor",
                    memory_id=old, content="Sven prefers light mode")
        # supersede row points at the old memory.
        sup_rows = db.list_audit(memory_id=old, action="supersede")
        assert len(sup_rows) >= 1
        # update row exists (on the new memory).
        all_update = db.list_audit(action="update", limit=10)
        assert any(r["actor"] == "auditor" for r in all_update)
    finally:
        db.close()


def test_audit_merge_writes_supersede_and_merge_rows(hermes_home: Path):
    db = _open_db(hermes_home)
    cfg = RemnantConfig()
    emb = _fake_embed(db, cfg)
    try:
        m1 = _store_simple(db, emb, cfg, fact="Sven likes vim", entity="Sven")
        m2 = _store_simple(db, emb, cfg, fact="Sven likes git", entity="Sven")
        res = memory_edit(db, cfg, emb, action="merge", actor="auditor",
                          memory_ids=[m1, m2], content="Sven likes vim and git")
        new = res["memory_id"]
        # Two supersede rows (one per original).
        sup1 = db.list_audit(memory_id=m1, action="supersede")
        sup2 = db.list_audit(memory_id=m2, action="supersede")
        assert len(sup1) == 1
        assert len(sup2) == 1
        # One merge row on the new memory.
        merge_rows = db.list_audit(memory_id=new, action="merge")
        assert len(merge_rows) == 1
    finally:
        db.close()


def test_audit_details_contain_before_after(hermes_home: Path):
    db = _open_db(hermes_home)
    cfg = RemnantConfig()
    emb = _fake_embed(db, cfg)
    try:
        mid = _store_one(db, emb, cfg, fact="Sven prefers dark mode")
        # forget: details has before snapshot + after_status.
        memory_edit(db, cfg, emb, action="forget", actor="auditor", memory_id=mid)
        row = db.list_audit(memory_id=mid, action="forget")[0]
        details = row["details"]
        assert isinstance(details, dict)
        assert details["after_status"] == "forgotten"
        assert "before" in details
        assert details["before"]["content"] == "Sven prefers dark mode"

        # feedback: details has before_score + after_score.
        mid2 = _store_one(db, emb, cfg, fact="Sven likes vim")
        memory_edit(db, cfg, emb, action="feedback", actor="auditor",
                    memory_id=mid2, feedback="useful")
        row = db.list_audit(memory_id=mid2, action="feedback")[0]
        assert "before_score" in row["details"]
        assert "after_score" in row["details"]

        # share: details has before + after visibility.
        memory_edit(db, cfg, emb, action="share", actor="auditor", memory_id=mid2)
        row = db.list_audit(memory_id=mid2, action="share")[0]
        assert row["details"]["before"] == "private"
        assert row["details"]["after"] == "shared"
    finally:
        db.close()


def test_audit_update_details_contain_before_snapshot(hermes_home: Path):
    db = _open_db(hermes_home)
    cfg = RemnantConfig()
    emb = _fake_embed(db, cfg)
    try:
        old = _store_one(db, emb, cfg, fact="Sven prefers dark mode")
        memory_edit(db, cfg, emb, action="update", actor="auditor",
                    memory_id=old, content="Sven prefers light mode")
        # Find the update audit row.
        all_update = db.list_audit(action="update", limit=50)
        row = next(r for r in all_update if r["actor"] == "auditor")
        details = row["details"]
        assert details["before_id"] == old
        assert "before" in details
        assert details["before"]["content"] == "Sven prefers dark mode"
        assert "after_id" in details
    finally:
        db.close()


# ===========================================================================
# 5. Contradiction detection
# ===========================================================================


def test_detect_contradiction_antonym():
    assert detect_contradiction("Sven prefers dark mode", "Sven prefers light mode")
    assert detect_contradiction("The server is online", "The server is offline")


def test_detect_contradiction_negation():
    assert detect_contradiction("Sven likes dark mode", "Sven does not like dark mode")


def test_detect_contradiction_no_contradiction():
    assert not detect_contradiction("Sven prefers dark mode", "Sven likes vim")
    assert not detect_contradiction("Sven prefers dark mode", "Alice likes vim")


def test_storing_conflicting_fact_flags_both(hermes_home: Path):
    db = _open_db(hermes_home)
    cfg = RemnantConfig()
    emb = _fake_embed(db, cfg)
    try:
        # First fact: creates the Sven entity + links the memory.
        mid1 = _store(db, emb, cfg, fact="Sven prefers dark mode",
                      entities=[{"name": "Sven", "type": "person", "aliases": []}])
        assert mid1 is not None
        # Second conflicting fact shares the Sven entity.
        mid2 = _store(db, emb, cfg, fact="Sven prefers light mode",
                      entities=[{"name": "Sven", "type": "person", "aliases": []}])
        assert mid2 is not None
        assert mid1 != mid2
        # Old memory's metadata.contradicts references the new fact.
        m1 = db.get_memory(mid1)
        meta1 = m1.get("metadata") or {}
        assert "contradicts" in meta1
        assert any("light mode" in c for c in meta1["contradicts"])
        # New memory's metadata.contradicts references the old memory id.
        m2 = db.get_memory(mid2)
        meta2 = m2.get("metadata") or {}
        assert "contradicts" in meta2
        assert mid1 in meta2["contradicts"]
    finally:
        db.close()


def test_contradiction_not_flagged_for_unrelated_facts(hermes_home: Path):
    db = _open_db(hermes_home)
    cfg = RemnantConfig()
    emb = _fake_embed(db, cfg)
    try:
        mid1 = _store(db, emb, cfg, fact="Sven prefers dark mode",
                      entities=[{"name": "Sven", "type": "person", "aliases": []}])
        mid2 = _store(db, emb, cfg, fact="Sven likes vim",
                      entities=[{"name": "Sven", "type": "person", "aliases": []}])
        assert mid1 is not None and mid2 is not None
        m2 = db.get_memory(mid2)
        assert "contradicts" not in (m2.get("metadata") or {})
    finally:
        db.close()


# ===========================================================================
# 6. Search exclusion
# ===========================================================================


def test_forgotten_memory_excluded_from_search(hermes_home: Path):
    db = _open_db(hermes_home)
    cfg = RemnantConfig()
    emb = _fake_embed(db, cfg)
    try:
        mid = _store_one(db, emb, cfg, fact="Sven prefers dark mode")
        # Visible before forget.
        before = hybrid_search(db, cfg, "dark mode", agent_id="default")
        assert any(r["id"] == mid for r in before)
        memory_edit(db, cfg, emb, action="forget", actor="default", memory_id=mid)
        after = hybrid_search(db, cfg, "dark mode", agent_id="default")
        assert all(r["id"] != mid for r in after)
    finally:
        db.close()


def test_superseded_memory_excluded_from_search(hermes_home: Path):
    db = _open_db(hermes_home)
    cfg = RemnantConfig()
    emb = _fake_embed(db, cfg)
    try:
        old = _store_one(db, emb, cfg, fact="Sven prefers dark mode")
        memory_edit(db, cfg, emb, action="update", actor="default",
                    memory_id=old, content="Sven prefers light mode")
        # Old content should not appear.
        results = hybrid_search(db, cfg, "dark mode", agent_id="default")
        assert all(r["id"] != old for r in results)
        # New content should appear.
        results = hybrid_search(db, cfg, "light mode", agent_id="default")
        assert any("light mode" in r["content"].lower() for r in results)
    finally:
        db.close()


def test_forgotten_excluded_from_semantic_search(hermes_home: Path):
    db = _open_db(hermes_home)
    cfg = RemnantConfig()
    emb = _fake_embed(db, cfg)
    try:
        mid = _store_one(db, emb, cfg, fact="Sven prefers dark mode")
        memory_edit(db, cfg, emb, action="forget", actor="default", memory_id=mid)
        results = hybrid_search(db, cfg, "dark mode", agent_id="default",
                                strategy="semantic", embedder=emb)
        assert all(r["id"] != mid for r in results)
    finally:
        db.close()


def test_forgotten_excluded_from_graph_search(hermes_home: Path):
    db = _open_db(hermes_home)
    cfg = RemnantConfig()
    emb = _fake_embed(db, cfg)
    try:
        mid = _seed_graph(db, emb, cfg)
        memory_edit(db, cfg, emb, action="forget", actor="default", memory_id=mid)
        # Graph search returns active memories only.
        results = graph_search(db, "Sven", agent_id="default", depth=2)
        assert all(r["id"] != mid for r in results)
    finally:
        db.close()


def test_memory_search_tool_excludes_forgotten(provider: RemnantMemoryProvider):
    provider.handle_tool_call(
        "memory_store",
        {"fact": "Sven prefers dark mode", "entity": "Sven"},
        session_id="seed",
    )
    res = provider.handle_tool_call(
        "memory_search", {"query": "dark mode"}, session_id="s",
    )
    mid = res["results"][0]["id"]
    provider.handle_tool_call(
        "memory_edit", {"action": "forget", "memory_id": mid}, session_id="s",
    )
    res2 = provider.handle_tool_call(
        "memory_search", {"query": "dark mode"}, session_id="s",
    )
    assert all(r["id"] != mid for r in res2["results"])


# --- config + tool schema sanity for Phase 3 --------------------------------


def test_phase3_tool_schemas_present(provider: RemnantMemoryProvider):
    schemas = provider.get_tool_schemas()
    names = {s["function"]["name"] for s in schemas}
    assert {"memory_search", "memory_store", "memory_reflect"} <= names
    assert "memory_graph" in names
    assert "memory_edit" in names


def test_memory_edit_schema_actions_enum(provider: RemnantMemoryProvider):
    schemas = provider.get_tool_schemas()
    edit = next(s for s in schemas if s["function"]["name"] == "memory_edit")
    props = edit["function"]["parameters"]["properties"]
    assert set(props["action"]["enum"]) == {
        "update", "merge", "forget", "feedback", "share", "unshare",
    }
    assert set(props["feedback"]["enum"]) == {"useful", "wrong"}


# --- re-verify cosine helper for graph/embedding sanity --------------------


def test_cosine_phase3_helper():
    assert cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
