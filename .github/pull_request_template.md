## Summary

Describe the change and why it is needed.

## Validation

- [ ] `python -m ruff check .` passes.
- [ ] `python -m pytest` passes.
- [ ] I avoided heavy workflows and did not train models unless explicitly
      required by the change.

## Repository Hygiene

- [ ] No generated or raw data is committed.
- [ ] No model artifacts, checkpoints, weights, or serialized models are
      committed.
- [ ] No secrets, tokens, credentials, private keys, or private data are
      committed.
- [ ] If model behavior is touched, model predictions remain advisory-only and
      deterministic checks remain authoritative.
- [ ] Docs are updated when CLI behavior, user-facing behavior, safety
      boundaries, or workflow steps change.

