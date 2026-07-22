---
name: commit-report
description: Recompile the SCF-RL LaTeX report, verify it builds clean, and commit the changes with a message describing what and why. Use once report edits (content, layout, or both) are ready to be saved to git history.
---

# Commit Report

Verifies and commits report changes. This skill never pushes — that's
[push-to-github](../push-to-github/SKILL.md), a separate, more consequential
step the user should approve explicitly (see the Git Safety Protocol
guidance already in effect for this project).

## When to use this skill

- After [write-section](../write-section/SKILL.md) and/or
  [format-report](../format-report/SKILL.md) have produced edits the user
  wants saved.
- The user explicitly asks to commit — never commit unprompted.

## Report location

Default: `../Report` relative to this repo's root.

## Steps

1. **Check repo state first.**
   ```
   git status
   git diff --stat
   ```
   If there are unexpected modified files beyond what this session touched,
   stop and confirm with the user before staging — don't sweep up unrelated
   in-progress work.

2. **Recompile before committing, always.** A report commit that doesn't
   build is worse than no commit.
   ```
   latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
   ```
   Check the log for `Output written on main.pdf` and no errors. If the
   build fails, fix it (or hand off to
   [format-report](../format-report/SKILL.md)) before proceeding — do not
   commit a broken build.

3. **Stage only the intended source files.** Never `git add -A` /
   `git add .` in this repo — LaTeX builds leave `.aux`, `.log`, `.out`,
   `.fls`, `.fdb_latexmk`, `.synctex.gz` files that should stay untracked
   (confirm `.gitignore` covers them; add entries if missing rather than
   hand-picking around them each time). Stage explicitly:
   ```
   git add sourcecode/03_simulation_methodology.tex sourcecode/04_results_discussion.tex
   ```

4. **Write the commit message around *why*, not just *what*.** Follow the
   style already used in this project's report history — short, specific,
   present-tense summary line, e.g.:
   - `Remove redundant Phase-3 refined-warm-start result; add BC/CQL/RL objective equations and CIRL-vs-regular-RL justification`
   - `Stack DDPG/TD3 critic/actor loss equation onto two lines, fixing column overflow`
   - `Remove trailing punctuation after every displayed equation`

   One commit per coherent change; don't bundle an unrelated layout fix into
   a content commit if they're easy to separate.

5. **Never add a `Co-Authored-By: Claude` trailer** to report commits (this
   project's standing preference — the user's commits should read as their
   own authorship).

6. **Commit.**
   ```
   git commit -m "$(cat <<'EOF'
   <summary line>
   EOF
   )"
   ```

7. **Confirm.** `git status` should show a clean tree (aside from untracked
   build artefacts) and `git log --oneline -1` should show the new commit.

## Output

Report the commit hash and message, and state clearly that it has **not**
been pushed. Ask, or wait for the user to invoke
[push-to-github](../push-to-github/SKILL.md), before touching the remote.
