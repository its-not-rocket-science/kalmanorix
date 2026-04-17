# AGENT RULES FOR THIS REPOSITORY

## Completion gate for future Codex tasks
Before declaring any future task complete, Codex must verify CI-critical checks pass locally.

Required checks:
1. `ruff format --check .`
2. `pytest -m "not integration and not stress"`

If any required check fails, the task is not complete.
If an environment limitation prevents running a required check, Codex must explicitly report the limitation and avoid claiming the task is complete.
