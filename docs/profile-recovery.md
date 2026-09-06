# Recovering split profile stores

Stop all database writers for the final snapshot, including the desktop backend,
all profile gateways, extraction scripts and scheduled jobs. Inventory each
process's `HERMES_HOME`, `REMNANT_DB_HOME`, and resolved plugin path; a matching
shared checkout alone does not prove a live rollout.

Back up every current and legacy Remnant SQLite store with SQLite's backup API.
Check integrity and foreign keys on each snapshot. Preserve the service settings,
profile plugin copies, configs and source commit for rollback.

Build an explicitly ordered manifest. Put the most current authoritative store
first. Every stored owner needs a destination owner; do not map two profiles to
one destination. For example:

```json
{
  "sources": [
    {"path": "/private/snapshots/claire.db",
     "owners": {"claire": "claire", "sasha": "sasha"}},
    {"path": "/private/snapshots/sasha.db", "owners": {"sasha": "sasha"}}
  ]
}
```

Run against snapshots, with a new output path:

```bash
python -m remnant.recover --manifest /private/manifest.json \
  --output /private/recovered.db --report /private/recovery-report.json
```

The command never changes an input or overwrites an existing output. It retains
memory IDs, content, status and owners; remaps numeric conversation IDs and
claim/receipt/queue/audit references; and retains source-to-target ID maps.
Overlapping content, status, provenance or ownership conflicts fail closed.
The first source supplies overlapping mutable projections, so put newer sources
first. Divergent claim versions map to the authoritative projection for that
memory; original snapshots remain the archive of prior versions.

Embeddings and evidence are retained. Disposable caches, telemetry and derived
Echo utility are rebuilt. No model, vault, dream or import job runs during the
merge. Schema 17 separates thread authorship from ownership and scopes dream
state. Legacy `system` threads and global state are assigned automatically only
in a single-owner store; for a mixed store, add `thread_owners` (thread ID to
source owner) and `dream_owner` to that source's manifest entry. Missing mappings
abort recovery instead of silently hiding data.

Validate source ID/content/owner/status coverage and turn provenance against all
current inputs. Run keyword, semantic, graph and provider-context recall for each
profile, plus denial checks using another profile's IDs and caller overrides.
Use copied data for mutating canaries, never production memories.

Only then install the recovered database while writers remain stopped, update
all plugin and database paths, and start services one at a time. Verify the
running process environments, plugin version, database path, profile identity,
recall and startup logs. Rollback requires the matching database, source, plugin
copies and service settings; preserve new writes before any later rollback.
