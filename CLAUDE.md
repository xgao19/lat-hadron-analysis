@AGENTS.md

## Temporary file policy
- Avoid creating temporary helper scripts unless strictly necessary.
- Prefer inline shell/Python snippets and apply-patch over ad hoc temp files.
- If a temporary file is necessary, place it under a clearly named temp path.
- After successful execution and verification, delete any temporary files created during the task unless I explicitly asked to keep them.
- Mention any temp files you intentionally keep.