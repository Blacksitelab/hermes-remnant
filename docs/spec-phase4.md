# Remnant — Phase 4 Spec: Vault Indexing + Profile-Scoped Search

## Goal

Index the Obsidian vault into Remnant as `type='document'` memories, react to file changes, parse frontmatter, respect agent workspace exclusions and locked-note rules, and add profile-scoped search filtering.

## In scope

- Vault path: `/home/jd/obsidian-vaults/BlacksiteLabVault`
- Exclude glob patterns: `90_*`–`95_*` and `99_ARCHIVE/`
- Parse YAML frontmatter into `metadata` JSON
- Store document memories with `source='vault'`, `source_id=relative vault path`
- Hash-based change detection; re-index only changed files
- Mark memories for deleted files as `forgotten`
- Per-agent `profile_scope` config: list of allowed path prefixes; search filters `document` memories by source_id prefix
- Locked notes: parse `locked: true` frontmatter; content not returned in search results for other agents (return metadata only)
- Background re-index function (for cron/timer use) and a `memory_import(source='vault', ...)` tool path

## Out of scope

- Real-time file watcher process (Phase 4 provides re-index function; a daemon/timer wires it)
- Dream loop (Phase 5)
- Threads (Phase 5)

## File targets

| File | Changes |
|------|---------|
| `remnant/vault.py` | New: `index_vault()`, `index_file()`, `_should_index()`, `_parse_frontmatter()`, `_file_hash()`, `_relative_path()` |
| `remnant/config.py` | Add `vault_path`, `vault_exclude`, `profile_scope`, `vault_reindex_interval_s` defaults |
| `remnant/db.py` | Add `vault_files` hash table, `get_vault_hash()`, `set_vault_hash()`, document search filtering helpers |
| `remnant/search.py` | Add `profile_scope` parameter; filter document source memories by allowed prefixes |
| `remnant/tools.py` | Add `memory_import` tool schema and dispatch; add `profile_scope` to search tool schema |
| `remnant/__init__.py` | Update system prompt block; add `reindex_vault()` helper on provider |
| `tests/test_phase4.py` | Tests for exclusions, frontmatter, hashing, re-index, deleted files, profile scope, locked notes |

## Acceptance criteria

- [ ] `90_*`–`95_*` and `99_ARCHIVE/` files are excluded from indexing
- [ ] Frontmatter `type`, `tags`, `status`, `created`, `updated`, `author`, `locked` are stored in metadata
- [ ] Document memories have embeddings and entity links
- [ ] Re-index skips unchanged files (hash match)
- [ ] Deleted vault files mark their memory as `forgotten`
- [ ] `profile_scope` restricts document search to allowed prefixes
- [ ] Locked note content is hidden from search results for other agents
- [ ] All prior tests still pass

## Design notes

- Use Python stdlib `hashlib.sha256` for file content hashing; store hex digest in `vault_files` table
- `vault_files( path TEXT PRIMARY KEY, hash TEXT, memory_id TEXT, indexed_at TEXT )`
- For `locked: true`, store content for indexing/search but mark metadata `locked=True`; search results for non-owner agents omit `content` and include only title/path + metadata
- Profile scopes in config are lists of relative path strings; search builds a SQL `OR` of `source_id LIKE ?` conditions
- Markdown content is indexed as plain text; strip frontmatter before embedding
