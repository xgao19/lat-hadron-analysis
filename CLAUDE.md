# Claude Code default rules for this repo

- Use minimal context.
- Never scan or summarize the entire repository unless explicitly asked.
- Read only the files and line ranges needed for the current subtask.
- Break work into the smallest safe steps.
- For refactors, cleanups, and integrations:
  - first propose a short plan,
  - then execute only step 1,
  - then stop.
- Keep responses under 8 lines by default.
- Do not paste large code blocks unless requested.
- After each edit, provide only:
  - what changed,
  - why,
  - next smallest step.
- If the session becomes long or context-heavy, warn me and suggest /compact or a fresh session.
- Prefer fresh sessions over carrying long historical context.
- Avoid broad grep/read operations when a narrow search will do.
