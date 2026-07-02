# Remnant — Phase 3 Spec: Entity Graph + Self-Editing

## Goal

Add entity graph traversal, expose it via `memory_search(strategy="graph")` and a new `memory_graph` tool, and implement `memory_edit` for agents to update, merge, forget, score, and share memories. All mutations are audit-logged; nothing is deleted.

## In scope

- Entity extraction from facts and turns (name + type + aliases)
- `entities`, `memory_entities`, `relations` tables already exist from Phase 1 schema; now fully wired
- Entity resolution: fuzzy match on name + aliases; canonical entity per agent
- `memory_graph(entity, depth=2)` tool: traverse `relations` and return connected entities + memories
- `memory_search(strategy="graph")`: extract entities from query, traverse graph, pull linked memories
- `memory_edit` tool with actions:
  - `update`: create new version, mark old `superseded`
  - `merge`: combine multiple memories into one, supersede others
  - `forget`: mark `status='forgotten'` (still in DB, not returned by search)
  - `feedback`: adjust `trust_score` (`useful` raises, `wrong` lowers)
  - `share`: promote `private` memory to `shared`
  - `unshare`: revert `shared` memory to `private`
- `audit_log` table: record all mutations with actor, timestamp, before/after
- Contradiction detection during ingestion: when a new fact conflicts with an existing one, flag both with `metadata.contradicts`

## Out of scope

- Vault indexing (Phase 4)
- Dream loop / threads (Phase 5)
- Migration from Hindsight (cross-phase)

## File targets

| File | Changes |
|------|---------|
| `remnant/db.py` | Entity CRUD, relation CRUD, audit log, contradiction storage, `search_graph()` |
| `remnant/entity.py` | New file: entity extraction, resolution, alias normalization |
| `remnant/graph.py` | New file: graph traversal helpers |
| `remnant/edit.py` | New file: `memory_edit` actions + audit logging |
| `remnant/extract.py` | Use entity extraction to populate entities/relations |
| `remnant/ingest.py` | Contradiction detection on new facts |
| `remnant/search.py` | Add `strategy="graph"` |
| `remnant/tools.py` | Add `memory_graph` and `memory_edit` schemas + dispatch |
| `remnant/__init__.py` | Update system prompt block |
| `tests/test_phase3.py` | New tests for entity graph, edit ops, audit log, contradictions |

## Acceptance criteria

- [ ] Entity extraction populates `entities` and `memory_entities`
- [ ] `memory_graph` returns connected entities within N hops
- [ ] `memory_search(strategy="graph")` finds memories linked to query entities
- [ ] `memory_edit update` creates superseded chain
- [ ] `memory_edit merge` collapses duplicates
- [ ] `memory_edit forget` hides memory from search but preserves DB row
- [ ] `memory_edit feedback` adjusts trust_score
- [ ] `memory_edit share/unshare` changes visibility
- [ ] All mutations write to `audit_log`
- [ ] Contradictions are flagged on ingestion
- [ ] Forgotten/superseded memories excluded from search
- [ ] All prior tests still pass

## Design notes

- Entity types from spec: person, service, project, concept, place, tool
- Aliases stored as JSON array; resolution lowercases and strips punctuation
- Relations get `strength` 0-1; initial links seeded from co-occurring entities in same memory
- Contradiction detection: compare new fact against existing memories sharing an entity; use a lightweight local heuristic first (negation words, antonyms) and an LLM check for ambiguous cases
- Audit log stores JSON `details` with before/after snapshots
- All edit actions return the resulting memory_id(s) and audit log id
