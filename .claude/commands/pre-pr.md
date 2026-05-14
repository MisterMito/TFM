Run every check required before opening a Pull Request.

1. `uv run ruff check .` — fix lint errors.
2. `uv run ruff format .` — format the code.
3. `uv run pytest -q` — run the test suite.
4. Inspect the diff with `git diff --stat` and summarise the changes.
5. Verify there are no large files (>50 MB) outside Git LFS.
6. If dependencies changed, verify that `pyproject.toml` and `uv.lock`
   were updated together.
7. Propose a commit message following Conventional Commits
   (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`).
8. On user confirmation, create the commit and push the branch.
   **Do not run `gh pr create`** — the user opens the PR to `main`
   themselves (see the Git workflow in `CLAUDE.md`).
