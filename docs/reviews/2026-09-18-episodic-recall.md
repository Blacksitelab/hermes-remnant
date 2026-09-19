# Independent review: episodic recall PR #46

Verdict: **request-changes**. The most consequential reproduced defect is cross-archive summary contamination: a worker can claim another archive's job and, with colliding source IDs and a shared reference, return that archive's private summary text to the wrong archive's caller. Independently, realistic long excerpts can produce an empty result with no continuation, and summary validation does not consistently preserve its provenance contract.

This is a completed review, not approval of the feature, authorization to merge/deploy, or final user acceptance.

## Target and scope

- Task: `t_1f1db27b`; reviewer: SENTINEL.
- PR: https://github.com/Blacksitelab/hermes-remnant/pull/46
- Baseline / merge base: `b194374bcf68c5d55f38d31f5c74eac71d921107`.
- Candidate: `7874c0b1114ec11bdfd1193701a7799a3dc92f96`.
- Plan: `docs/episodic-recall-implementation-plan.md`, task `t_062aa1ea`; SHA256 `610c44116a7b9661e4b6c24222bb8efd2497f14cc3561662db372efaaac952fb`.
- Exact head remained unchanged; tracked diff was empty at final verification. Existing untracked plan and `model-backfill-report-final2.json` were not modified. The protected report was not opened or hashed, so its byte-for-byte hash was not independently verified.
- Read all changed production code and new tests, and traced provider lifecycle, tool dispatch, archive authorization, cache/queue transactions, extraction, ordinary retrieval/prefetch, claim resolution, graph/vault/import boundaries, Echo, dream, maintenance, backup/restore and recovery. Whole tracked Python source/test inventory was AST-parsed and scanned: 85 modules / 31,017 lines. Full repository tests and lint ran. The whole-codebase audit was risk-driven, not a claim of exhaustive manual line-by-line proof of every unchanged helper.
- No production DB or live configuration was opened. Source archives, migrations, concurrency probes, model responses and security markers were synthetic. No real model endpoint was called. Report/probe files are the only review additions; implementation fixes were not authored.

## Findings introduced by the PR

Severity here indicates impact, not exploit prevalence. Every finding below includes the trigger and the evidence; no security claim depends on interpreting arbitrary model prose as infallible.

### S-001 — High — Summary worker ignores the archive partition

Locations: `remnant/db.py:1151-1212` (`claim_history_summary`, especially selection at 1176-1181); `remnant/history.py:1091-1148`; `remnant/history.py:2094-2128`.

Trigger: two provider-local archive homes share a Remnant DB and configured `agent_id` (including the default), with a colliding session ID. Claims select by owner only, although the cache primary key includes `archive_key`. The claiming service reads its own archive, not the archive on the claimed row, then writes the resulting summary to that row.

Reproduced with two isolated default-profile archives, each containing the same session/message IDs, timestamps and a common question but different private answers. B processed A's pending job. The cache row under A's archive key became ready with B's summary. A's later evidence-only recall returned the synthetic `B private detail`. Its common reference validated; the other invalid reference was dropped rather than invalidating the statement (S-003). Different explicit profile metadata can prevent this particular final cache hit, but cannot fix the wrong-archive claim itself.

Evidence: `episodic-review-probes.json`, `cross_archive_claim`: all four booleans/status demonstrate the wrong worker, wrong cache and returned foreign text. The model stub deliberately returns a structurally valid summary of B's supplied messages; it is not claimed to measure a live model.

Required correction: scope claiming, lease recovery/work checks and completion to the trusted archive key as well as owner; reject any claimed archive mismatch before loading content or calling a model. Retain owner-wide daily budget if desired. Add a shared-DB/two-archive collision test that proves no foreign work or text crosses the partition.

### S-002 — High — Output fitting can erase all evidence, lose continuation and exceed the wire cap

Locations: `remnant/history.py:2680-2713`, `1520-1547`, `1549-1570`, `2510-2631`; final serializer `remnant/__init__.py:624`.

The fitting loop removes evidence while retaining the original, oversized `synthesis` until after the removal loop. For 24 messages of 1,900 characters, evidence-only session recall returned **zero** evidence, zero usable continuation, and `has_more=false`. Coverage nevertheless reported `messages_returned=24`. This is a realistic source length, below the per-excerpt cap, not an artificial tiny-budget configuration.

The cap also measures compact internal JSON, then mutates the cursor/coverage after fitting; the provider subsequently emits JSON with spaces. In the independent wire-format probe, the final provider-equivalent serialization was **4,154 estimated tokens**, above the promised 4,000. Its compact form was 3,894. Token estimates also lag post-fitting cursor changes.

Required correction: budget the actual final serializer and all final fields; keep synthesis synchronized with retained evidence during fitting. Preserve at least a bounded usable excerpt/provenance or a working continuation for oversized pages, and calculate returned/omitted counts after fitting. Regression-test long ASCII/CJK, multi-session pages, final provider serialization and complete cursor draining, not just small excerpts.

### S-003 — High — Summary validation can cite unseen input or retain invalidated claims

Locations: `remnant/history.py:1109-1146`, `2205-2224`, `2277-2315`, `2068-2138`, `2434-2447`.

Two reproduced provenance failures:

1. A 60-message summary prompt includes first/last 20 of the **loaded** rows, but validation allows all 60. A stub citing message 30, absent from the prompt, produced a ready cache entry. Coverage said `messages_returned=60`, `sampled=false`, `content_truncated=false`, although only 40 messages were supplied. Character fitting can remove still more input without changing these facts. For longer sessions, only the first 80 messages are loaded, so their last 20 are not the actual session tail.
2. A cached statement cited a generic discussion plus `Adopt option blue.`. After editing the latter source to `Actually adopt option red.`, source count/max-ID/version stayed the same. Reference validation rejected the changed source but kept the old blue statement on the surviving generic reference. Recall returned the obsolete summary and the corrected raw text. Range filtering has the same any-surviving-reference behavior. This is not merely the unavoidable uncertainty of a model paraphrase: a known invalidated dependency remains rendered.

Summary-first deduplication can then hide the original excerpt for the surviving reference, reducing the caller's ability to inspect the unsupported paraphrase.

Required correction: carry the exact fitted model-input records/references into validation and coverage; sample true first/last session windows within bounds. Invalidate a statement/cache when a required reference is edited, hidden, rewound, deleted or excluded from the claimed range; do not silently transplant the whole statement onto a subset. Retain accessible raw fallback and its provenance. Test multi-reference corrections and mixed-date summaries as well as single-citation rejection.

### S-004 — Medium — Runtime summary reference checks authorize message IDs as session IDs

Locations: `remnant/history.py:655`, `945-960`.

`_authorized` chooses `data['id']` before `data['session_id']`. For a session row that is correct; for `validate_reference` it is the numeric **message** ID. Runtime ownership lookup consequently queries Remnant turns for session `"1"` rather than the actual session `"s"`.

Reproduced: a turn establishes exact Alice ownership of session `s`; session validation succeeds but its visible message reference fails. This breaks generation/reuse of properly owned historical summaries in runtime-identity mode. The common path fails closed; this finding is not a claim that the probe leaked content.

Required correction: pass the source session ID explicitly to the shared ownership check, avoiding row-shape inference. Test current trusted sessions and historical turn-mapped sessions, plus another owner and unknown ownership.

### S-005 — Medium — Relative continuation silently changes its time window

Locations: `remnant/history.py:401-405`, `1219`, `1254`, `1443`, `1627-1632`.

The cursor contains the original selector/fingerprint but not the captured reference clock/resolved range. Each continuation resolves `yesterday`, `today`, `last_week` or rolling durations using a new clock. Reproduced by advancing an injected clock across midnight: a valid continuation for June 14 instead searched June 15 and returned `no_evidence` with remaining June 14 messages unvisited. Rolling durations drift even without crossing midnight.

Required correction: persist and validate the first request's resolved bounds/reference clock in the cursor and reuse them on every page. Test elapsed-time and civil-day rollovers on a stable archive.

### S-006 — Medium — Topic discovery can discard confirmed matches and never combine cache/raw coverage

Locations: `remnant/history.py:727-731`, `847-867`, `1860-1888`, `1995-1997`.

Raw SQL searches full content but projects only the first 2,001 characters. Expansion tests the query again against that prefix and discards the confirmed hit if the match is later. A message with `quasar` after 2,200 characters returned `no_evidence`, `discovery_complete=true`, `has_more=false`, despite `topic_hits_returned=1`.

Separately, cache topic search runs only when the initial raw search is empty. A valid cache-only quasar session was omitted when a different session contained a literal raw quasar match; no cursor offered it. This fails the planned combined discovery rather than merely lacking semantic search.

Required correction: preserve authoritative lexical hit identity and return a bounded match-local excerpt/window without reading an unbounded message. Merge/paginate authorized cache and raw candidates so neither vetoes the other. Add late-within-message matches, cache/raw union and late-correction topic tests.

### S-007 — Medium — The request deadline and connection lifetime are not bounded as documented

Locations: `remnant/history.py:556-566`, `666-673`, `677-715`, `801-867`, `1291-1425`; README historical limits.

Each archive connection starts a fresh three-second SQLite progress deadline. No request-wide deadline includes repeated discovery/authorization/expansion or waits on the shared Remnant lock. A real temporary DB lock held for 3.4 seconds made an evidence-only request take over 3.4 seconds and return `ok`, rather than observe the three-second request ceiling. Repeated per-reference/session reads can multiply that allowance.

`with self._connect() as conn` commits/rolls back but **does not close** sqlite3 connections. Fifty archive metadata reads with cyclic GC temporarily disabled increased process descriptors from 7 to 57, returning to 7 only after GC. This demonstrates GC-dependent resource release, not a claim of a permanent leak under every CPython workload.

Topic retrieval probes FTS once for diagnostics but always executes a full-content `LOWER(...) LIKE '%term%'` scan with OFFSET. Date discovery's result LIMIT does not bound how many sessions SQLite examines. A cancelled LIKE query is swallowed as an empty hit set plus `index_incomplete`; it can still report `discovery_complete=true`. Large-corpus correctness/latency therefore needs more than a returned-row cap.

Required correction: reuse a single monotonic request deadline, including `db.deadline`, all archive queries and expansion loops; close connections deterministically with existing stdlib context management. Use bounded indexed source discovery where available, with explicit resumable/incomplete fallback. Cancellation must never assert complete discovery. Correct README promises to match tested behavior.

### S-008 — Medium — Validly typed extreme selectors raise uncaught exceptions

Locations: `remnant/history.py:363-375`, `1218-1221`, `1584-1633`.

Both `relative='999999999999999999999d'` and `start='9999-12-31'` raised uncaught `OverflowError` from `recall`, rather than return `invalid_request`. Validation catches only `HistoryRequestError`. The duration matches the exposed schema's pattern.

Required correction: bound selector size/magnitude and translate datetime arithmetic/conversion overflow to `HistoryRequestError` before archive/model work. Check extreme anchors and cursor numeric conversions at the same boundary. Add a compact invalid-input regression group.

### S-009 — Medium — A temporarily missing archive stays unavailable for the service lifetime

Locations: `remnant/history.py:536-548`, `556-569`, `1010`, `1023-1027`.

The trusted path is resolved with `strict=True` only during construction. When the file does not exist then, `_trusted_path` remains `None` forever. Reproduced by creating the correct temporary `state.db` after service construction: recall still reports `unavailable`. This can affect first-session initialization or recovery after a temporarily missing archive; it is separate from the correct behavior of returning unavailable while the file is absent.

Required correction: retry safe path resolution/containment on later access, without creating the archive or weakening symlink checks. Add absent-then-created and replacement/escaping-symlink tests.

## Pre-existing full-codebase findings — separate from PR regression attribution

These warrant follow-up; they were not introduced by episodic recall and should not be silently bundled into a large history rewrite.

### S-010 — High — Vault symlink escapes configured filesystem scope

Locations: `remnant/vault.py:47-67`, `252-287`, `429-456`.

When resolved containment fails, `_relative_path` falls back to the lexical path. `allowed/linked.md` can resolve to a file outside the vault yet pass `profile_scope=['allowed']`; the walker and reader follow it. A synthetic external Markdown document was indexed as `allowed/linked.md` in the independent probe. On real input this can import outside-vault/private content and pass it to configured embedding/entity processing.

Required correction: fail closed on resolved root escape at the common indexing boundary and recheck scope against the canonical contained path. Test file/directory symlinks and exclusions with no network calls. Preserve intentionally allowed in-vault links if product policy requires them.

### S-011 — High — Explicit `memory_graph` bypasses configured vault scope

Locations: `remnant/tools.py:539-559`; `remnant/recall.py:251-263`; `remnant/graph.py:50-76`.

The graph tool omits `config.profile_scope` even though `graph_traverse` accepts it. The subsequent candidate authorization checks owner only, so it does not repair the missing path restriction. An owned vault document at `private/blocked.md` was returned by `memory_graph` under `profile_scope=['allowed']`. The existing search tests do not exercise this tool path.

Required correction: apply configured scope consistently to graph candidates and exposed entity metadata, and make the common candidate authorization boundary enforce the relevant active/visibility/path policy rather than trusting each adapter. Regression-test explicit graph, normal search and prefetch with the same excluded owned document.

### S-012 — High data-safety risk, not reproduced corruption — WAL on an affected SQLite runtime

Location: `remnant/db.py:577-588`; dependency/runtime support in `pyproject.toml`.

The tested Python 3.12.3 runtime links SQLite 3.45.1; Remnant unconditionally enables WAL. SQLite's upstream documentation says versions through 3.51.2 are affected by the rare WAL-reset corruption race when concurrent connections write/checkpoint; fixes are 3.51.3+, with specified backports 3.44.6 / 3.50.7. This shared, multi-process DB design meets the concurrency precondition. Installed Hermes independently detected this runtime during the disposable integration check and chose DELETE journal mode; Remnant has no corresponding guard.

Primary source retrieved during review: https://sqlite.org/wal.html#walresetbug . No attempt was made to trigger corruption, and a passing integrity check does not disprove the race.

Required correction: establish a safe runtime requirement/check or coordinate a documented safe journal fallback. Account for vendor-backported fixes, concurrency and deployment compatibility. Operator/runtime changes require separate authorization; do not update production from this review.

## Verification actually executed

All checks below used the exact candidate unless explicitly labelled baseline. Probe scripts assert observed defects for reproducibility; their successful exit does **not** mean the feature passes acceptance.

| Check | Result |
|---|---|
| `.venv/bin/python -m pytest -q` | 464 passed; initial 18.20 s, final 17.74 s |
| `.venv/bin/python -m ruff check remnant tests` | Passed on both runs |
| Leadership deterministic evaluation | 240 cases; recall@5 1.0; stale exposure 0 |
| Held-out deterministic evaluation | 120 cases; recall@5 1.0; stale exposure 0; duplicate top-k occupancy 0 |
| Fresh DB / health / backup / restore | Schema 18; integrity `ok`; verified backup and restored copy |
| Genuine baseline DB migration | Created using baseline schema 17, upgraded with candidate, reopened; exact durable turns/memories/claims unchanged; cache empty; integrity `ok` |
| Wheel + sdist | `python -m build` on a `git archive` copy of the candidate; passed |
| Isolated wheel import | Fresh temporary venv; import resolved to wheel site-packages, version 0.3.2; `HistoryService` module included |
| Dependency consistency / compilation | `pip check` and `compileall` passed |
| Installed Hermes integration | Fresh temporary `SessionDB`; real create/append/close; dated recall returned the source message, status `ok` |
| Archive security | Compacted visible rows allowed; hidden/rewound/tool/system rows excluded; guessed hidden anchor rejected; SQL-injection session string and tool path/owner/profile overrides did not broaden access; escaped archive symlink rejected |
| Archive immutability | Synthetic source DB SHA256 identical before/after recall |
| Failure fallback | Malformed JSON, foreign citation and model timeout returned raw fallback; exactly one attempted model call each |
| Ordinary-turn invariant | Trap history object on provider: `sync_turn`, `prefetch`, `queue_prefetch` with date prose never touched history |
| Calendar selectors | Leap day; UTC; both Auckland DST directions; both New York DST directions; fixed-clock yesterday/last_week passed. Cross-request relative continuation fails S-005 |
| Concurrent summary budget | Two independent DB connections claimed exactly 20 unique jobs, respecting atomic owner/day reservation |
| Whole-source static scan | AST parse of tracked Python; no selected executable/deserialization primitives or private-key/token patterns found. This limited scan is not a comprehensive secret/CVE audit |
| Exact-head CI | Required Python 3.10, 3.11, 3.12 checks passed. Actions run 35295782061 reported head SHA exactly matching candidate; PR OPEN/CLEAN. Rechecked at 2026-09-18T02:05:27Z |

No type checker is configured. Editor diagnostics on deliberately duck-typed review fixtures are not a project type-check result. Several shell invocations using dynamic executable variables or `python -c` were blocked by the runner policy; equivalent absolute-path/script invocations succeeded without changing approvals. One review-script call initially used the wrong `resolve_entity` argument shape; corrected the probe and reran it successfully. No implementation failure is inferred from that harness error.

## Measured performance and storage/resource assessment

`evaluation/benchmarks/retrieval_hardening.py` ran against isolated baseline and candidate snapshots with 5,000 memories, 768-dimensional vectors and instant deterministic embedding stubs, seven probes per snapshot:

| Measurement | Baseline | Candidate |
|---|---:|---:|
| Median semantic retrieval | 293.217 ms | 293.603 ms |
| Median provider prefetch | 311.642 ms | 310.427 ms |
| Delivered contexts | 7 / 7 | 7 / 7 |
| Python scoring peak | 0.54 MiB | 0.54 MiB |

Top-100 IDs and scores were byte-for-value identical. Prefetch changed by approximately -0.39%; that is noise-level evidence of no measured regression, not a statistically established improvement or an endpoint latency guarantee. History-only no-hit lookup over 30,000 synthetic messages of roughly 1,120 characters had median 82.585 ms (five runs). This is not a million-message or worst-case performance claim; the always-LIKE plan and S-007 prevent declaring a hard corpus-independent latency bound.

The new cache has transactionally enforced row/owner queue caps, bounded stored summary/coverage JSON and an atomic daily counter; no embeddings or extra transcript table were introduced. Recovery correctly treats it as disposable. Existing metrics compaction caps telemetry rows and prunes old history day counters; audit log and durable turns/memories remain unbounded by design and can contain historical text. Do not describe the cache bound as a bound on the whole DB, SQLite freelist/WAL or audit growth. A 1,000-row cap alone also does not bound arbitrary session metadata loaded via `SELECT s.*`.

The ordinary prefetch implementation and turn ingestion are unchanged, apart from static routing text/new lifecycle wiring. The summary callback runs after extraction, at most one job per callback. It can still delay later queued extraction while its model call is running. No live endpoint or sustained mixed extraction/Echo/history load test was performed. The HTTP helper's timeout is a transport timeout, not a proven total wall-clock response deadline; it buffers response JSON before the history parser applies a size check.

## Acceptance mapping

`Failed` means at least one required condition has contrary evidence. `Unverified` names remaining tests rather than claiming absence of a bug.

| Plan criterion | Assessment / evidence |
|---|---|
| AC01 dates/ranges/relative time | Failed overall: message-time endpoints, leap/DST/fixed-clock cases pass; extreme selectors crash (S-008), continuation clock changes (S-005) |
| AC02 exhaustive bounded paging/coverage | Failed: small committed pagination tests pass, but large excerpts lose all continuation and returned counts are wrong (S-002); deadline/coverage contract fails (S-007). Exhaustive timestamp-tie/mixed-topic stress remains unverified |
| AC03 general topics, corrections, unresolved/abandoned history | Failed: topic tail/cache-union gaps S-006 and stale correction S-003. Live semantic preservation and late-correction multi-session synthesis remain unverified; heuristic kind labels are not semantic proof |
| AC04 cache/extraction/FTS/archive/model fallback | Failed overall: raw path needs no legacy extraction row; malformed/model failure fallback works; late archive availability does not recover (S-009). Missing/rebuilding FTS and every dead-letter combination are not comprehensively covered |
| AC05 provenance | Failed: unsupplied summary references and retained invalid dependencies (S-003). Wholly foreign request citations are rejected; model-derived labels exist but cannot repair known invalid sources |
| AC06 explicit-only work / boundary hooks | Core invariant passed by call-flow inspection and runtime trap; committed disabled test passes. No new ordinary-turn history call. Boundary hook enqueue-only design inspected; full live shutdown/flush ordering unverified |
| AC07 hard budgets / adversarial input | Failed: empty-budget pathology and final wire cap (S-002), input overflow (S-008), request deadlines (S-007). Bounded one-call model contract/fallback checked with stubs; no live prompt-injection robustness guarantee |
| AC08 profile/runtime/path/visibility | Failed: cross-archive cache partition S-001, runtime summary identity S-004. Raw row visibility and escaped archive link probes pass. Broader pre-existing vault/graph boundary failures S-010/S-011 remain |
| AC09 queue/cache/retry/migration/recovery | Failed: wrong-archive claims S-001, invalidation/sampling S-003. Owner queue cap committed test, concurrent daily budget, migration/reopen and disposable recovery designation pass. Exhaustive global eviction, restart/expired-lease and repeated source-version retry stress remain unverified |
| AC10 accurate health/docs/protected artifact | Failed: summary sampling health and advertised output/deadline limits do not match behavior. Health avoids raw archive access and reports unknown whole-archive coverage. Protected report left untouched; its hash deliberately not read |
| AC11 repository checks/build/import | Passed for configured suite, Ruff, both deterministic corpora, fresh health, build, isolated import and compile checks; Python matrix covered by exact-head CI, local runtime 3.12.3 |
| AC12 PR/exact CI/independent review | Passed: real open PR, exact-head required checks, this independent report; no merge/deploy. Approval is withheld |

## Next owner and limits

IMPLEMENTER owns S-001 through S-009 and the missing regression coverage needed to satisfy the plan; update the same PR only under its existing publication authorization, then request independent review of the new exact head. Coordinate S-010/S-011 and the S-012 runtime safety decision as separate baseline follow-up work; do not mix live operations into this review. ARCHITECT/user owns any decision to reduce the explicit plan's completeness or resource guarantees rather than implement them.

The deterministic leadership/held-out corpora exercise existing durable-memory recall, not `memory_history`; their perfect fixture scores must not be presented as historical-recall quality evidence. No real archive corpus, live model, production filesystem race, actual WAL corruption, crash/power-loss injection, deployment tokenizer or live backup restore was tested. Prompt framing and citations reduce risk but do not establish semantic truth or immunity to injection.

## Reproducible evidence files

All paths below are relative to this report's directory. The scripts create disposable temporary data only; their model responses are explicitly labelled stubs. They assert candidate defects and are intended to be replaced/adapted into positive regressions by IMPLEMENTER, not treated as green acceptance tests.

- `episodic-review-probes.py` and `episodic-review-probes.json`: reproduced PR defects, resource lifecycle and synthetic history timing.
- `episodic-review-verification.py` and `episodic-review-verification.json`: deterministic evaluations, security/fallback/invariant/concurrency/integration checks, baseline security reproductions.
- `episodic-review-ops.py`: wheel import, source scan, temporal and genuine migration checks, performance comparison.
- `episodic-review-performance.json`: measured baseline/candidate comparison.

Run the first two from the repository root with `PYTHONPATH=. .venv/bin/python docs/reviews/<script>`. The verification script's installed-Hermes integration uses `/home/jd/.hermes/hermes-agent` source code but writes only its explicitly created temporary Hermes home.
