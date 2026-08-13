# Domain docs

How the engineering skills must consume this repo's domain documentation during exploration.
The layout is **single-context**: one `CONTEXT.md` at the repo root and one `docs/adr/`
directory.

## Read these before you explore

**Bottom line: read `CONTEXT.md` and the ADRs that touch your work area. Proceed if they are absent.**

- **`CONTEXT.md`** at the repo root holds the glossary and the domain model.
- **`docs/adr/`** holds the architecture decision records. Read the ADRs that touch the area
  you are about to change.

Neither path exists yet. If a file is absent, proceed without a comment. Do not flag the
absence and do not offer to make the file at the start. The `/domain-modeling` skill makes
these files later, when a term or a decision is actually resolved.

`docs/` is gitignored in this repo, but `docs/agents/` and `docs/images/` are un-ignored.
`CONTEXT.md` sits at the repo root, so it gets version history. `docs/adr/` does **not** get
version history unless you add an un-ignore rule for it first.

## Use the vocabulary of the glossary

**Bottom line: name each domain concept with the term in `CONTEXT.md`, not a synonym.**

This applies to issue titles, refactor proposals, hypotheses and test names.

If the concept is absent from the glossary, treat this as a signal. Either you invent language
that the project does not use, which you must reconsider, or there is a real gap. Record a real
gap for `/domain-modeling`.

The workspace file `../../CLAUDE.md` and the repo file `CLAUDE.md` hold settled vocabulary
today. Examples are "marker", "baseline", "wear", "session" and "trial". Use them until
`CONTEXT.md` exists.

## Flag conflicts with an ADR

**Bottom line: state the conflict. Never override an ADR silently.**

Write the conflict in this form:

> Contradicts ADR-0007 (event-sourced orders), but worth reopening because...
