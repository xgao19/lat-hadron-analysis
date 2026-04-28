@AGENTS.md

## End-of-Task Summary
At the end of each non-trivial task, provide a short change summary.

Use this format:
- Changed: [files or notebooks changed]
- Why: [one short reason]
- Verified: [what you checked]
- Next: [next step, if any]

Rules:
- Keep it brief.
- Do not paste large code blocks.
- Do not repeat the full reasoning process.
- If no file was changed, explicitly say so.

## Temporary file policy
- Avoid creating temporary helper scripts unless strictly necessary.
- Prefer inline shell/Python snippets and apply-patch over ad hoc temp files.
- If a temporary file is necessary, place it under a clearly named temp path.
- After successful execution and verification, delete any temporary files created during the task unless I explicitly asked to keep them.
- Mention any temp files you intentionally keep.