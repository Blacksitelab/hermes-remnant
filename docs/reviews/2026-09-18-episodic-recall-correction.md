# PR #46 correction review — REQUEST CHANGES

The corrections make substantial progress, but pagination still silently skips evidence and model validation still preserves statements after discarding invalid citations. Those are release-blocking correctness failures. This is the completed independent review of the correction, not acceptance, merge/deploy authorization, or a new full-codebase audit.

## Exact target and verification

- Task: `t_c44a569a`; reviewer: SENTINEL.
- PR: https://github.com/Blacksitelab/hermes-remnant/pull/46
- Correction baseline: `7874c0b1114ec11bdfd1193701a7799a3dc92f96`.
- Candidate: `55d9ded51d89a52254d885492fa37d4e069b3fa1`.
- Plan: `docs/episodic-recall-implementation-plan.md`, SHA256 `610c44116a7b9661e4b6c24222bb8efd2497f14cc3561662db372efaaac952fb`.
- GitHub PR head and Actions run `35301439134` independently matched the candidate. Required Python 3.10/3.11/3.12 checks all passed; `gh pr checks 46 --required` confirmed this at the verification checkpoint `2026-09-18T03:31:33Z`.
- Independently executed `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider`: **470 passed in 21.92s**. `.venv/bin/python -m ruff check remnant tests`: passed.
- Read the three-file correction diff, affected full functions, DB deadline/queue callers, and tool/provider dispatch including the exact JSON serializer at `remnant/__init__.py:624`.
- Executed `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv/bin/python docs/reviews/episodic-correction-probes.py > docs/reviews/episodic-correction-probes.json`. These independent desired-behavior checks report individual PASS/FAIL; exit zero means the harness completed, **not** that acceptance passed.

## Per-finding acceptance matrix

| Finding | Verdict | Independently observed result |
|---|---|---|
| S-001 archive-scoped work/disclosure | **FAIL — partly fixed** | Initialized A/B archives with colliding IDs no longer cross-claim or disclose B's text. A service whose archive is absent still claims A's job through an empty archive key, consumes an attempt, and moves it to retry_wait. |
| S-002 wire budget and complete pagination | **FAIL — partly fixed** | Long ASCII/CJK single-session pages recover all 24 IDs and stay under the real serialized cap. Short-ID multi-session fixture recovers all 189. With ordinary 36-character session IDs, a complete drain returns only 136 of 180 IDs, then reports ok/no continuation. |
| S-003 exact supplied/valid provenance | **FAIL — partly fixed** | Unseen-only citation is rejected. Edited, hidden, rewound, deleted and out-of-range dependencies invalidate cached summaries. True tail sampling includes IDs 181–200 from a 200-message session. But mixed valid/unseen summary citations and mixed valid/foreign request-synthesis citations still retain the entire claim on only the valid reference. |
| S-004 runtime ownership | **PASS — resolved** | Exact historical Alice ownership authorizes the reference; Bob and unknown ownership fail closed; trusted current-session authorization works. |
| S-005 frozen relative cursor | **PASS — resolved** | Across midnight, both yesterday and rolling 2d continuations preserve identical resolved ranges and recover all 85 source IDs. |
| S-006 topic tail/cache union | **FAIL — partly fixed** | Late lexical hit now returns its message ID, but the returned excerpt still omits the matching text. One raw + one cached match works; one raw + 25 cached matches ends with an empty continuation page and missing evidence. |
| S-007 deadline/cleanup | **FAIL — partly fixed** | Fifty reads with cyclic GC disabled keep descriptors at 8. Initial lock contention returns partial at approximately 3.0015 seconds. Lock contention beginning after discovery instead lets TimeoutError escape from the subsequent authorization pass. |
| S-008 overflow validation | **FAIL — partly fixed** | Original extreme relative/date values and huge anchor return invalid_request. A bounded, otherwise valid date cursor containing a 401-digit timestamp raises uncaught OverflowError. |
| S-009 missing archive recovery | **PASS — resolved** | Absent archive returns unavailable; subsequently created synthetic archive becomes searchable; replacing it with an escaping state.db symlink again returns unavailable. |

## Required corrections, in impact order

### S-002 — High: final fitting advances past evidence it subsequently removes

Locations: `remnant/history.py:1740-1777`, `1794-1799`, `2738-2857`.

Trigger: 20 sessions, 36-character IDs, nine 1,900-character messages each, one dated request followed by every returned cursor. The first fitted page's cursor advances four message positions, but the final wire-budget refit leaves only three evidence records. Continued paging repeats the loss: 136 unique IDs returned out of 180, final status `ok`, `has_more=false`. The final serialization stays within 4,000 estimated tokens; preserving the cap alone does not preserve completeness.

At `1798`, `_fit_output` can remove more evidence after `_adjust_cursor_after_budget` has already fixed the continuation. No subsequent cursor reconciliation occurs. The fixed 128-token reserve is insufficient for a multi-session cursor. With allowed 110-character IDs, cursor encoding also fails outright: four of 180 messages returned and no continuation.

Required: derive continuation from the evidence surviving the **last** fit, include the actual encoded cursor in that fit, and reduce the selected session batch or compact cursor state when it exceeds the cursor ceiling. Do not erase continuation on overflow. Retain complete-drain regression checks for ordinary UUID-length IDs and the supported long-ID boundary.

Evidence keys: `S-002-ids-36`, `S-002-ids-60`, `S-002-long-ids`. The first-page returned IDs and cursor offset total are recorded directly.

### S-003 — High: mixed invalid references still become apparently grounded claims

Locations: `remnant/history.py:2494-2553` and `2562-2602`; callers `1320` and `2389`.

A summary model stub cites supplied message 1 plus unsupplied message 30. The worker returns success and stores `Claim depends on unseen message 30` as ready, retaining only citation 1. A request-time model stub citing a valid source plus foreign message 999 similarly returns `Unsupported joint claim` with only the valid source. These are explicit invalid dependencies being discarded, not a claim that paraphrase truth can be mechanically proved.

The new retrieval-side all-reference validation is sound for the tested mutations, but both generation validators still `continue` on bad references and accept the statement whenever any reference survives. Validating only the fitted input set fixes the unseen-only case, not this mixed case.

Required: reject the entire statement when any required citation is malformed, unsupplied or invalid, at both summary generation and request synthesis. Do not silently shorten the citation list, including references beyond the allowed count. Preserve raw fallback. Add mixed-valid/invalid regressions at both callers.

Evidence: `S-003-mixed-summary`, `S-003-mixed-synthesis`; mutation and unseen-only checks document the resolved portions.

### S-006 — Medium: discovery finds sources that expansion/pagination still cannot deliver

Locations: `remnant/history.py:1019`, `1557-1562`, `1634-1635`, `2065-2085`, `2197-2212`.

For `"x" * 2200 + " quasar"`, SQL now creates a match-local projection, but `_expand_messages` replaces it with `get_message_window`'s prefix projection. The returned ID is correct; neither its excerpt nor deterministic statement includes quasar. The committed test checks only the ID and misses the content defect.

For one raw hit plus 25 valid cache-only sessions, the first page returns source IDs 1 and 9–26; the next page returns no evidence and ends paging. Cache candidates advance independently of what survives fitting, and the general topic `message_cursor_active` check suppresses summary evidence on continuation even for newly selected cache-only sessions. Those sessions have no raw lexical fallback by construction.

Required: preserve the bounded SQL-confirmed match-local content through expansion with valid source provenance. Advance raw/cache positions only past actually delivered evidence, and validate/render newly selected cached candidates on subsequent pages. Test excerpt content and complete cache/raw union drains, not just membership in a single short page.

Evidence: `S-006-match-local-evidence`, `S-006-union-1`, `S-006-union-25` (including each page's IDs).

### S-001 — Medium remaining impact: missing archive removes the claim partition

Locations: `remnant/history.py:1265-1281`; `remnant/db.py:1163-1167`, `1317-1345`.

After constructing A with a pending job, construct another service sharing its DB/owner but with a missing archive. Its empty `archive_key` is normalized to `None`, removing the SQL partition. Calling its worker claims A's job, spends the owner-wide model reservation, detects the mismatch, and fails that foreign job into retry_wait with attempts=1. No model call or foreign text disclosure was observed in this case; the original initialized-archive disclosure reproduction is closed. The remaining defect interferes with another archive's queue and budget during the supported missing-archive condition.

Required: never claim/recover/fail another archive's work when the trusted key is empty. Resolve availability/key before claiming and make worker-facing archive scope fail closed. Keep the owner-wide daily budget if intended. Test missing-archive workers alongside pending and expired jobs for a valid archive.

Evidence: `S-001-existing-archives`, `S-001-missing-worker-partition`.

### S-007 — Medium: timeout escapes the result boundary after discovery

Locations: `remnant/history.py:1510-1542`, `2889-2893`; `remnant/tools.py:446`.

The discovery try/except ends before the second `session_rows` authorization pass. A synthetic thread takes the real shared Remnant lock immediately after the first successful session validation. The unchanged three-second request deadline expires during the second authorization; TimeoutError propagates through `recall` instead of yielding a bounded partial response. The probe synchronizes this interleaving; it does not replace the DB lock or timeout exception with a stub. The ordinary initial-lock case now passes.

Required: include all authorization/expansion stages in the deadline failure boundary and return honest partial/unknown coverage rather than throwing from the tool. Preserve deterministic connection closure.

Remaining original requirement not satisfied by inspection: FTS at `1001-1005` is still diagnostic only; discovery always executes the full-content lexical scan at `1017-1025`. On cancellation, the empty result cannot provide a resumable scan position. The new code avoids claiming complete discovery, which is an improvement, but has not implemented the requested indexed discovery/resumable bounded fallback. No new large-corpus benchmark is claimed. Implement that bounded discovery requirement or obtain an explicit ARCHITECT/user revision; a passing initial-lock test is not evidence of it.

Evidence: `S-007-close`, `S-007-initial-lock`, `S-007-reauthorization-lock`.

### S-008 — Medium: cursor conversion still bypasses invalid_request

Locations: `remnant/history.py:503-505`, `542-544`, called inside `1471-1475`.

A valid owner/request fingerprint and resolved range with date anchor `[10**400, "s"]` encodes to only 891 bytes, below the cursor limit. `float(after[0])` raises OverflowError, which neither local conversion handler catches; the outer boundary only catches HistoryRequestError. No archive/model work is needed to trigger it.

Required: bound timestamp magnitude and catch conversion overflow at both top-level and nested date-anchor validation, translating it to HistoryRequestError. Keep the passing relative/date/message-anchor guards and add the compact cursor-overflow regression.

Evidence: `S-008-relative`, `S-008-date`, `S-008-anchor`, `S-008-cursor-number`.

## Limitations, artifacts and handoff

Only synthetic temporary archives/Remnant DBs and explicitly stubbed model responses were used. No production DB/config or protected `model-backfill-report-final2.json` was read, hashed or modified. Existing untracked plans/review material were preserved. No implementation edits, commits, pushes, merge or deployment occurred. Local source remained at the exact candidate with no tracked implementation/test diff.

The correction checks reuse the fixture constructor in the existing `docs/reviews/episodic-review-probes.py` without running its old defect assertions. The first harness run hit the daily summary cap while seeding 25 historical cache rows; the fixture was corrected to use an explicit larger test-only claim allowance, with no model calls, then rerun successfully. A combined shell invocation containing `python -c` was blocked by runner policy; the ordinary script invocation succeeded without changing approvals. These were harness/tool issues, not candidate failures.

This bounded re-review did not repeat the original full audit, packaging/evaluation benchmarks, live model tests or deployment-tokenizer validation. The original report remains historical evidence, not approval of this head. Token caps here use the project's estimator on the provider's actual JSON serialization. Concurrency probes establish the stated interleavings, not every possible race.

Deliverables:
- This report: `docs/reviews/2026-09-18-episodic-recall-correction.md`.
- Runnable independent checks: `docs/reviews/episodic-correction-probes.py`.
- Actual captured results: `docs/reviews/episodic-correction-probes.json`.

Next owner: Claire/user applies this verdict to the existing implementation lane; IMPLEMENTER owns the precise corrections above. S-004, S-005 and S-009 are closed for this head. No additional agents/tasks were created. Standalone review card `t_c44a569a` is complete; this does not complete or unblock `t_3626a0c4`. Separate baseline security task `t_97af959f` remains pending; even eventual PR approval is not deployment readiness.
