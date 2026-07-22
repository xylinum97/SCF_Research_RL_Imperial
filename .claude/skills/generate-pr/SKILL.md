---
name: generate-pr
description: Open a GitHub pull request for a batch of SCF-RL report changes made on a branch, with a clear summary and test plan. Use when report edits were made on a feature branch rather than pushed straight to main, or when the user wants changes reviewed before merging.
---

# Generate PR

Opens a pull request for report changes, for the (less common, in this
project's normal workflow) case where edits were made on a branch rather than
committed straight to `main` via
[commit-report](../commit-report/SKILL.md) →
[push-to-github](../push-to-github/SKILL.md).

## When to use this skill

- The user is working on a named branch and wants review before merging into
  `main` (and therefore before it reaches Overleaf, which syncs from `main`).
- A large or risky change (e.g. a full section restructure) where a PR diff
  is a better review surface than a direct push.

For routine content/formatting edits that go straight to `main`, this skill
is unnecessary — use [push-to-github](../push-to-github/SKILL.md) directly.

## Repository

`https://github.com/xylinum97/Research-Project-Report---Reinforcement-Learning`,
checked out at `../Report` relative to this repo's root.

## Steps

1. **Confirm branch state.**
   ```
   git status
   git branch --show-current
   git log --oneline main..HEAD
   ```
   Every commit that will land in the PR should be reviewed here — not just
   the latest one.

2. **Diff against `main`, not just the latest commit.**
   ```
   git diff main...HEAD --stat
   ```
   Read the full diff for the sections changed, not only the stat summary —
   a report diff is easy to skim past a substantive change hidden inside a
   large equation reformat.

3. **Push the branch** (with the user's go-ahead — pushing a new branch is
   less risky than pushing to `main` but is still a remote-visible action):
   ```
   git push -u origin <branch-name>
   ```
   Apply the same credential gotcha as
   [push-to-github](../push-to-github/SKILL.md) if a `403` occurs
   (`git config credential.username stallyargha97-png`).

4. **Recompile before opening the PR** so the PR description can honestly
   state the report builds clean:
   ```
   latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
   ```

5. **Open the PR** with `gh pr create`, title under ~70 characters, body via
   heredoc:
   ```
   gh pr create --title "<short title>" --body "$(cat <<'EOF'
   ## Summary
   - <what changed, one bullet per logical change>

   ## Sections touched
   - sourcecode/<file>.tex — <one-line reason>

   ## Test plan
   - [ ] `latexmk -pdf` builds with no errors
   - [ ] No dangling `\ref`/`\cite` (see review-report)
   - [ ] No duplicated subsections from merge (see push-to-github)
   - [ ] Numbers consistent across abstract/results/conclusions
   EOF
   )"
   ```

6. Report the PR URL back to the user.

## Output

The PR URL, plus a one-line note on whether the report currently builds
clean on that branch. Never merge the PR yourself — that decision belongs to
the user (or their supervisor, if this report requires sign-off before
merging to `main`).
