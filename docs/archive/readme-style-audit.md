# README style audit

A line-by-line pass over the README to remove inflated or generic wording and
replace it with specific, checkable statements. Representative changes:

| Removed / avoided | Replacement |
|-------------------|-------------|
| "modular, generator-agnostic framework … built around an inspectable evidence graph" (abstract opener) | "EG-RAG retrieves passages, splits them into atomic claims, and connects the claims with typed relations …" (says what it does) |
| Status table with check-mark decoration and "next milestones" | Plain state words: implemented / tested / not run, with a sentence naming what was actually checked |
| Implied results ("validated", "works") stated without scope | "Nothing here is a benchmark result: the numbers come from unit tests and small controlled smoke runs" |
| "robust", "comprehensive", "seamless", "powerful", "production-ready" | not used |
| "state-of-the-art" / superiority phrasing | explicit non-claim: "This is research software … not a completed benchmark evaluation" |
| "leverages / enables / provides" filler verbs | direct verbs tied to a module or command |
| Marketing framing of the 0.5B model | "not faithful enough for evaluation … only useful for exercising the adapters" |
| Vague reranker claim | "`reranked_passage_rag` currently applies an identity reranker over BM25 order" |

Other checks applied while editing:

- No decorative emojis, badges, or exclamation marks.
- No "This project…" openings; no symmetric "supports X, provides Y, enables Z"
  lists standing in for content.
- Every capability names a concrete module, command, enum, or artifact path.
- Claims that could apply to any project were cut or made specific.
- Future work is stated as pending, not promised.
- No machine-specific paths; no assistant or tooling references.

The License and Citation sections state what is actually known (MIT declared in
`pyproject.toml` but no `LICENSE` file; a placeholder BibTeX with no venue/year).
