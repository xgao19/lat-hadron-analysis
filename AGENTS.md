# Repository expectations

## Refactor philosophy
This repository prefers a slim implementation over compatibility preservation.

- Remove dead code aggressively.
- Keep only the current intended workflow.
- Do not preserve old entry points, aliases, or fallback branches unless explicitly requested.
- Do not add abstraction layers just to make a diff look cleaner.
- When two code paths do nearly the same thing, keep one and delete the other.
- Keep one canonical naming scheme for each concept.
- Minimize interface surface area.

## Change output expectations
When performing cleanup or refactor work, always report:
1. audit summary
2. proposed deletions / merges
3. canonical path kept
4. removed legacy logic
5. manual verification checklist

## Handoff prompt preference

When asked to generate a prompt for another agent such as Claude, prioritize:
- patterns learned from actual work already completed
- repository-specific mappings and transformation rules
- pitfalls, hidden dependencies, and target-specific checks

Do not produce a generic task restatement unless explicitly asked.
Keep process constraints brief unless the user asks for stricter control.
Always put the final handoff prompt in one large code block for easy copying.
