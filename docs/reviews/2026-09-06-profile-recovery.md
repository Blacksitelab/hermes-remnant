# Remnant 0.3.1 recovery validation

The 0.3.0 release checked a shared database containing 6,223 memories but missed
the process-level profile database overrides, the desktop backend's open Claire
store, and three separately installed 0.2.2 plugin copies. The earlier claim of a
complete fleet rollout was not supported by those checks.

The follow-up fixes all four PR #43 findings: scoped vault deletion, global dream
state, runtime-identity memory-file imports, and legacy thread ownership. Schema
17 retains the author separately from the owner; ambiguous ownership requires an
explicit map. Recovery runs offline and preserves source snapshots.

Initial rehearsal: 12 Remnant database snapshots passed integrity and foreign-key
checks. The five current stores contain 9,104 distinct memory IDs:

| Owner | Memories |
| --- | ---: |
| Claire | 8,224 |
| Margaret | 230 |
| Sasha | 401 |
| Yuki | 249 |

The merge retains the existing Claire thread and 1,337 distinct conversation turns.
All source memory IDs, contents, owners and statuses, and all mapped conversation
contents, were checked. Keyword, semantic, graph and provider-context canaries
pass in the gateway's Python runtime with the configured embedding endpoint.
All returned memory IDs belong to the calling profile, including when a caller
supplies another profile's owner override. The default profile receives no fleet
memories. The final deployment must repeat snapshots and validation after all
writers are stopped; rehearsal counts are not a frozen production count.

453 automated tests pass, including the four regressions and recovery checks for
colliding numeric turn IDs, duplicate source snapshots, input preservation,
unknown owners, conflicting evidence, output overwrite refusal, and legacy graph
links whose owner must be inferred from their backing memory.

See [the recovery procedure](../profile-recovery.md) for operator gates and rollback.
