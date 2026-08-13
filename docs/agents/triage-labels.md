# Triage labels

The skills speak in five canonical triage roles. This file maps each role to the string used
in this repo. The tracker is local markdown, so the string goes in the `Status:` line of the
issue file. Refer to `issue-tracker.md`.

## Role map

**Bottom line: the defaults are kept. Each string is the same as its role name.**

| Role in mattpocock/skills | String in our tracker | Meaning |
| --- | --- | --- |
| `needs-triage` | `needs-triage` | Guillaume must evaluate this issue |
| `needs-info` | `needs-info` | Waiting for more information from the reporter |
| `ready-for-agent` | `ready-for-agent` | Fully specified, ready for an AFK agent |
| `ready-for-human` | `ready-for-human` | Guillaume must implement this |
| `wontfix` | `wontfix` | Will not be actioned |

When a skill names a role, for example "apply the AFK-ready triage label", use the matching
string from the middle column.

To change the vocabulary, edit the middle column. Do not run the setup skill again.
