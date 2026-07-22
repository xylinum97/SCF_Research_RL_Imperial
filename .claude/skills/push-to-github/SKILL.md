---
name: push-to-github
description: Push committed SCF-RL report changes to GitHub so Overleaf's git sync picks them up, safely reconciling with any concurrent Overleaf-side edits. Use after commit-report, or whenever the user asks why Overleaf/GitHub hasn't updated.
---

# Push to GitHub

Syncs the local report checkout with its GitHub remote
(`xylinum97/Research-Project-Report---Reinforcement-Learning`), which is what
the project's Overleaf project pulls from. This is the step that makes local
edits actually visible in Overleaf — a common point of confusion is editing
locally and expecting Overleaf to update without ever pushing.

## When to use this skill

- After [commit-report](../commit-report/SKILL.md), when the user wants the
  changes live on GitHub/Overleaf.
- The user asks "why hasn't Overleaf updated" or "why can't I see my changes"
  — the answer is almost always that local commits were never pushed, or
  that the remote has diverged and the push was silently rejected upstream
  of this skill running.

This is a **push to a shared remote** — a consequential, hard-to-fully-reverse
action per this project's standing git safety rules. Never force-push. Always
confirm with the user before pushing if there's any ambiguity about what's
being published.

## Report location and remote

Default: `../Report` relative to this repo's root, remote
`https://github.com/xylinum97/Research-Project-Report---Reinforcement-Learning`.

## Known account gotcha

This GitHub account setup has more than one credential identity available
locally, and only one of them has push access to `xylinum97` repos. If a push
is rejected with `403`, do **not** retry blindly — pin the working username
explicitly before retrying:
```
git config credential.username stallyargha97-png
```
(`unsupervisedlearning822-ui` is known to get `403` on `xylinum97` repos.)

## Steps

1. **Fetch before pushing, every time.**
   ```
   git fetch origin
   git log --oneline main..origin/main
   ```
   If this is empty, a plain `git push origin main` is sufficient — do it and
   confirm the ref update in the output.

2. **If the remote has diverged** (Overleaf edits landed while you were
   working locally — this has happened before in this project, since
   Overleaf's git sync commits directly to the same branch), do not blindly
   merge or rebase. First understand the shape of the divergence:
   ```
   git diff --stat main origin/main
   ```
   A small, disjoint diff (different files, or the same file but far-apart
   line ranges) will usually auto-merge cleanly. A large diff touching the
   same sections you edited (e.g. a restructure) will not — **attempting a
   plain `git merge` on a heavily-restructured file has previously produced
   silently duplicated sections** (git's line-based diff misaligning content
   that moved position, e.g. a subsection reordered earlier in the document
   gets merged as if it were new content, while the old copy also survives).
   Always inspect a conflicted or even a *clean* auto-merge of a
   heavily-changed file before trusting it:
   ```
   grep -n "^\\\\subsection{" sourcecode/<file>.tex | sort | uniq -c | awk '$1>1'
   ```
   If anything repeats, the merge is broken — abort (`git merge --abort`) and
   reconcile by hand: read the full current `origin/main` version of the
   file (`git show origin/main:sourcecode/<file>.tex`), and manually
   re-apply your local changes on top of that current structure rather than
   trusting an automatic merge.

3. **Recompile after any merge**, clean or hand-reconciled, before pushing —
   a merge that resolves without conflict markers can still produce content
   that doesn't compile or that silently duplicates a section (see above).
   ```
   latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
   ```

4. **Commit the merge** (if one was needed) with a message naming what was
   reconciled, e.g. `Merge origin/main: reconcile with Overleaf's methodology
   restructure`, plus a short note of what was carried forward from each
   side.

5. **Push.**
   ```
   git push origin main
   ```
   Confirm the output shows the expected `<old>..<new> main -> main` ref
   update, not a rejection.

6. If push is rejected again after step 1's fetch (a race — someone pushed in
   between), repeat from step 1. Do not force-push to resolve a race.

## Output

State the final commit hash pushed, and whether a merge/reconciliation was
needed. If a merge happened, summarize what was kept from each side so the
user can verify nothing was lost. Remind the user Overleaf may need a manual
sync/refresh click if its auto-detection is slow.
