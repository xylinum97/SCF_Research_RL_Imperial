---
name: review-report
description: Check the SCF-RL LaTeX report for cross-section consistency — matching numbers, dangling references, terminology drift, phase-numbering mismatches, redundant or superseded results. Use before committing a batch of report edits, or when asked to review/audit the report.
---

# Review Report

A consistency and correctness pass over the report, distinct from
[format-report](../format-report/SKILL.md) (which fixes LaTeX layout, not
content) and from a LaTeX compile (which catches syntax errors, not factual
drift).

## When to use this skill

- Before committing a batch of edits made across multiple sections.
- After merging in Overleaf-side edits, to check nothing now contradicts
  something else in the document.
- When asked to "check the report," "review it," or "does everything still
  add up."
- Periodically, if the underlying experiments have been re-run and numbers
  may have shifted.

## Report location

Default: `../Report` relative to this repo's root.

## What to check

1. **Cross-section number consistency.** The same headline results tend to
   appear in the abstract, results/discussion, and conclusions. Grep for the
   key numbers (e.g. `0.245`, `1.224`, `0.109`, `91\%`, `85\%`) across all of
   `sourcecode/*.tex` and confirm every occurrence agrees. A number that was
   updated in one place but not another is the most common real bug here.

2. **Superseded or redundant results.** If a phase's methodology changed
   (e.g. a comparison approach was dropped or replaced), search for old
   phrasing that assumed the previous approach — a common failure mode is a
   table or paragraph surviving a restructure and now contradicting the
   surrounding text (e.g. describing a refinement step that a later edit
   removed). Cross-check the abstract, results, and conclusions all describe
   the *same* current pipeline, not a mix of old and new.

3. **Dangling references.** Every `\ref{...}`, `\cite{...}`, and
   `\hyperref[...]{...}` must resolve to an existing `\label{...}` or bib key.
   ```
   grep -o "label{[^}]*}" sourcecode/*.tex | sed 's/.*label{//;s/}//' | sort -u > /tmp/labels.txt
   grep -o "ref{[^}]*}" sourcecode/*.tex | sed 's/.*ref{//;s/}//' | sort -u > /tmp/refs.txt
   comm -23 /tmp/refs.txt /tmp/labels.txt   # refs with no matching label
   ```
   Do the same for `\cite{...}` keys against `references.bib`.

4. **Duplicate labels.** Two `\label{}`s with the same name silently break
   all references to one of them. `grep -o "label{[^}]*}" sourcecode/*.tex |
   sort | uniq -d` should be empty.

5. **Duplicated sections.** If a merge or restructure went wrong, the same
   `\subsection{...}` heading can end up appearing twice (this has happened
   before during a botched git merge in this project — see
   [push-to-github](../push-to-github/SKILL.md)). `grep -n "subsection{"
   sourcecode/03_simulation_methodology.tex` (or whichever file was touched)
   and confirm every heading appears exactly once.

6. **Phase numbering and naming.** "Phase 1/2/3/4" titles must match exactly
   everywhere they're referenced in prose (not just in their own
   `\subsection` heading) — check `04_results_discussion.tex` describes the
   phases the same way `03_simulation_methodology.tex` defines them.

7. **Terminology drift.** Watch for inconsistent naming of the same concept
   across sections — e.g. "regular BC" vs "free-form BC", "CIRL" vs
   "control-informed RL" used interchangeably without ever being defined as
   the same thing, or old naming (e.g. "v1/v2 reward") surviving in one
   section after being retired elsewhere.

8. **Units and precision.** A quantity shouldn't be `\SI{1.224}{\celsius}` in
   one place and `1.22°C` (bare, non-siunitx) in another.

## Steps

1. Run the label/ref/cite consistency greps above.
2. Grep for the current headline numbers across all sections and diff them
   for consistency.
3. Read `01_header_abstract.tex`, `04_results_discussion.tex`, and
   `05_conclusions.tex` back-to-back — these three are the most likely to
   drift from each other since they each restate the same results.
4. Skim `03_simulation_methodology.tex` subsection headings for duplicates
   and confirm phase descriptions match the results section.
5. Compile (`latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`)
   as a final sanity check — an undefined reference shows up here too, as
   `??` in the PDF or an "undefined references" warning in the log.

## Output

List findings as a short punch list (file:line, what's inconsistent, what it
should probably say instead) — do not silently fix them unless the user has
also asked you to apply fixes. If nothing is wrong, say so plainly rather
than inventing minor nitpicks to report.
