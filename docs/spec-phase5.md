# Remnant — Phase 5 Spec: Threads + Dream Loop

## Goal

Add topic threads that capture ongoing conversations, and a bounded dream loop that finds non-obvious connections across memories via a cloud model, writing its reflections to a private diary and actionable suggestions as threads.

## In scope

- `threads` table management: create, update, resolve, list, stale sweep
- `memory_thread` tool with actions `create`, `update`, `resolve`, `list`, `stale`
- Day/night dream loop functions (not a daemon — callable from cron/systemd)
- Local candidate pre-filtering using cosine similarity (>0.6 connection, >0.7 cross-agent dedup)
- Cloud model judgment on candidate pairs (configurable endpoint, default deepseek-v4-flash:cloud via Hermes gateway or direct)
- Two-stage evaluation: generate candidate observations, then self-evaluate usefulness
- Actionable results:
  - Real connections → append to `DREAMS.md` diary and optionally promote to thread
  - Same fact across agents → merge into `shared` memory, supersede originals
  - Similar wording → discard, log in diary
- Stale thread sweep: 14 days inactive → status='stale'
- Impulse budget: day max 3, night max 5, 2-hour cooldown per topic
- Diary at `~/.hermes/remnant/DREAMS.md`, first-person, not indexed by Remnant
- Machine state in SQLite: `dream_state` table

## Out of scope

- GPU VRAM guard (extract.py already has queue; defer explicit VRAM check)
- Web dashboard
- Email/feed indexing

## File targets

| File | Changes |
|------|---------|
| `remnant/threads.py` | New: thread CRUD, stale sweep |
| `remnant/dream.py` | New: day_dream, night_dream, candidate selection, cloud judgment, diary write, budget/cooldown |
| `remnant/db.py` | Add `threads` and `dream_state` tables; thread CRUD helpers |
| `remnant/config.py` | Add dream model endpoints, budgets, cooldown, diary path |
| `remnant/tools.py` | Add `memory_thread` tool; wire `memory_import` for future sources |
| `remnant/__init__.py` | Update system prompt block; add `run_dream_loop(mode)` method |
| `tests/test_phase5.py` | Tests for thread CRUD, stale sweep, dream candidate selection, budget/cooldown, diary write, merge across agents |

## Acceptance criteria

- [ ] `memory_thread create/update/resolve/list/stale` work and return expected shapes
- [ ] Threads inactive for 14 days are marked stale
- [ ] Dream loop selects candidate pairs using local similarity only
- [ ] Cloud model is called with a bounded candidate list
- [ ] Two-stage evaluation rejects noise
- [ ] Cross-agent duplicates are merged into shared memories
- [ ] Impulse budget enforced per mode
- [ ] Cooldown prevents repeated suggestions on same topic within 2 hours
- [ ] Diary entries are appended, first-person, and not indexed
- [ ] Machine state is persisted in `dream_state`
- [ ] All prior tests still pass

## Design notes

- `threads` table fields: id, title, topic, status, importance, tags, related_entities, source, added_by, created_at, last_activity, updated_at
- `dream_state` table: key, value JSON, updated_at
- Candidate selection: for each memory created since last run, find top-5 similar active memories; also cross-agent top similar pairs
- Cloud prompt returns JSON: `{ "judgments": [{"pair_ids": [...], "judgment": "connection|same_fact|noise", "reason": "...", "thread_title": "..."}] }`
- Merge same_fact across agents via `memory_edit(action='merge', actor='system')` with combined content
- Budget counters stored in `dream_state` and reset daily/nightly based on `day_run_ts` and `night_run_ts`
- Diary format: `## 2026-07-02 02:00 (night)\n\nI noticed ...\n\n---\n`
- The diary is never read by Remnant for indexing; it's only for human review
