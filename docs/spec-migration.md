# Remnant — Migration Spec: Hindsight + Memory Store Import + Shadow Mode

## Goal

Import existing durable facts from Hermes MEMORY.md / USER.md files across all profiles and from the Hindsight memory store, run deduplication, and operate a shadow mode that logs what Remnant WOULD inject without actually injecting it.

## In scope

### Memory store import (`memory_import(source='memory_store')`)

- Discover `MEMORY.md` and `USER.md` in all Hermes profiles under `~/.hermes/profiles/`
- Parse each entry as a fact with `source='import'`, `confidence=0.9`, `trust_score=0.9`
- Agent tag from profile name
- Visibility:
  - User profile facts (name, timezone, preferences) → `fleet`
  - Agent-specific relationship/working-style facts → `private`
  - Cross-agent project/hardware facts → `shared`
- Use heuristics to assign visibility based on content keywords
- Extract entities, generate embeddings, run dedup
- Write to audit log with `action='import'`
- Support `dry_run` mode

### Hindsight import (`memory_import(source='hindsight')`)

- Use `hindsight_recall` with broad queries to pull stored memories
- Parse each entry as `source='hindsight'`, `trust_score=0.5`
- Default visibility `private`
- Run extraction for entities and embeddings
- Run dedup against existing memories
- Write to audit log
- Support `dry_run` mode

### Shadow mode

- `memory_import` with `shadow=True` logs to `~/.hermes/remnant/shadow.log` instead of importing
- Each line: JSON with timestamp, source, proposed action, content, duplicate status, token estimate
- Shadow log is for human comparison against Hindsight's actual injection

### `memory_import` tool schema

```python
{
  "name": "memory_import",
  "description": "Import memories from existing stores. Use dry_run to preview.",
  "parameters": {
    "source": {"enum": ["memory_store", "hindsight", "vault"]},
    "profile": "string (optional)",
    "dry_run": "boolean",
    "shadow": "boolean"
  }
}
```

Vault import was already implemented in Phase 4.

## Acceptance criteria

- [ ] MEMORY.md / USER.md entries parsed across all profiles
- [ ] Visibility heuristics assign fleet/shared/private reasonably
- [ ] Deduplication collapses duplicates during import
- [ ] Hindsight recall fetches memories and imports them
- [ ] dry_run returns counts without writing
- [ ] shadow=True writes to shadow.log instead of DB
- [ ] All imports write audit_log entries
- [ ] All prior tests still pass

## Design notes

- Memory store entries are freeform markdown bullet lines or numbered items. Parse by bullet boundaries (`- `, `* `, `1. ` etc.) and treat each as a separate fact.
- For Hindsight, call `hindsight_recall` with a set of broad queries and aggregate results. Avoid duplicates by content hash.
- Visibility heuristic keywords:
  - fleet: timezone, name, email, prefers, hates, wants, language, region, location
  - shared: project, repo, hardware, server, network, decision, agreed, plan, build
  - private: relationship, style, notes, manner, habit, personal
- Store the original source path in `source_id`
- If a duplicate is found during import, increment `seen_count` on the existing memory and skip creating a new row
- Shadow log path: `~/.hermes/remnant/shadow.log`; rotate by appending; not a memory source
- Dry_run still performs parsing, extraction, and dedup simulation but does not write to memories/audit_log
