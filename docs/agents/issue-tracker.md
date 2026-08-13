# Issue tracker: local markdown

Issues and specs for this repo live as markdown files in `.scratch/`. There is no remote
tracker. The `gh` CLI is not installed on this machine.

## Conventions

**Bottom line: one directory for each feature, one file for each ticket, numbered from 01.**

- Keep one feature in one directory: `.scratch/<feature-slug>/`.
- Write the spec to `.scratch/<feature-slug>/spec.md`.
- Write each implementation issue to `.scratch/<feature-slug>/issues/<NN>-<slug>.md`, numbered
  from `01`. Never write one combined tickets file.
- Record triage state as a `Status:` line near the top of each issue file. Refer to
  `triage-labels.md` for the role strings.
- Append comments and conversation history to the bottom of the file, below a `## Comments`
  heading.

## When a skill says "publish to the issue tracker"

**Bottom line: make a new file below `.scratch/<feature-slug>/`.**

Make the directory first if it does not exist.

## When a skill says "fetch the relevant ticket"

**Bottom line: read the file at the given path.**

Guillaume usually gives the path or the issue number directly.

## Wayfinding operations

**Bottom line: `/wayfinder` uses one map file and one file for each child ticket.**

- **Map**: `.scratch/<effort>/map.md`. It holds the Notes, Decisions-so-far and Fog body.
- **Child ticket**: `.scratch/<effort>/issues/NN-<slug>.md`, numbered from `01`, with the
  question in the body. A `Type:` line records the ticket type (`research`, `prototype`,
  `grilling` or `task`). A `Status:` line records `claimed` or `resolved`.
- **Blocking**: a `Blocked by: NN, NN` line near the top. A ticket is unblocked when each file
  in that list is `resolved`.
- **Frontier**: scan `.scratch/<effort>/issues/` for files that are open, unblocked and
  unclaimed. The lowest number wins.
- **Claim**: set `Status: claimed` and save the file before you start work.
- **Resolve**: append the answer below an `## Answer` heading, set `Status: resolved`, then
  append a context pointer to Decisions-so-far in `map.md`.
