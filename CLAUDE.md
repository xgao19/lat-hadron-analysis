# General context-safety rules

## Context discipline
- Be efficient with context usage.
- Avoid broad repository scans unless clearly necessary.
- Prefer targeted reads, narrow searches, and small coherent steps.
- Do not repeatedly restate prior findings unless needed.

## When context becomes heavy
- If the session is becoming context-heavy, stop before failure.
- Before continuing, produce a short working summary containing:
  - current task
  - what has been completed
  - key files / notebooks / configs involved
  - important decisions or discovered rules
  - next smallest sensible step
- Then explicitly suggest either:
  - `/compact`
  - or starting a fresh session with that summary

## Self-compaction behavior
- When progress depends more on retained history than on new file reads, prefer summarizing and compressing state rather than continuing to accumulate context.
- Do not continue large reads or long explanations when the session already feels heavy.
- If a task can continue from a compact summary, switch to that mode.

## Execution style
- Complete a coherent small unit of work, then summarize briefly.
- Prefer 1–3 related low-risk actions before stopping.
- Avoid long code dumps and long prose unless explicitly requested.

## Response style
- Keep updates concise and high-information.
- After each meaningful step, report only:
  - current goal
  - what changed
  - key findings
  - next step