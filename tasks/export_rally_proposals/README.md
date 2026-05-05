# export_rally_proposals

Exports rally-start proposals before final acceptance/rejection so they can be reviewed and labeled.

```bash
docker compose run --rm app python tasks/export_rally_proposals/run.py
```

The output includes:

- kept and rejected proposals
- the current rule decision and reject reason
- serve/toss diagnostics
- bounce-direction and nearest-wrist diagnostics
- a flat feature row for each proposal
- an empty `label` field ready for manual annotation

This is the handoff point for the planned lightweight rally-start classifier.
