---
name: format-report
description: Fix LaTeX layout problems in the SCF-RL report — equations overflowing into the adjacent column, misaligned multi-line equations, stray punctuation in display math, table/figure placement — and recompile to confirm the fix. Use when the user reports a rendering/layout issue, often from an Overleaf or PDF screenshot.
---

# Format Report

Fixes how the report *renders*, not what it *says*. Distinct from
[write-section](../write-section/SKILL.md) (content) and
[review-report](../review-report/SKILL.md) (factual/reference consistency).

## When to use this skill

- The user pastes a screenshot of the compiled PDF or Overleaf preview showing
  something visually wrong: text overlapping, an equation running into the
  next column, a table caption colliding with body text.
- Asked to "fix the layout," "make this centered," "this doesn't fit," or
  similar purely-visual requests.

## Report location

Default: `../Report` relative to this repo's root.

## Common problems and fixes in this document class

This report is a **two-column** article (`\twocolumn`), which makes single
lines that are too wide the most common failure mode. Column width is
narrow — anything with two side-by-side terms joined by `\qquad` is a strong
overflow candidate.

1. **Equation overflows into the neighbouring column.**
   Symptom: an equation's second half visually collides with text in the
   other column (e.g. a list item, a table caption). Almost always caused by
   `\qquad` or similar wide inline spacing joining two expressions that
   should be two separate lines. Fix: split into a multi-line environment.

   ```latex
   % before (overflows)
   \begin{equation}
     A = f(x), \qquad B = g(x),
   \end{equation}

   % after (fits)
   \begin{gather}
     A = f(x),\\
     B = g(x),
   \end{gather}
   ```

2. **Two-line equation looks left-shifted instead of centered.**
   Symptom: the first of two stacked lines appears shifted left of the
   second, instead of both looking centered in the column. Cause: using
   `align` with `&=` aligns the equals signs vertically rather than centering
   each line independently — correct only when the two lines are meant to
   read as a related system (e.g. an update pair like the anti-windup
   integral). If the two lines are two independent statements just being
   grouped together, use `gather` instead, which centers each row on its own:

   ```latex
   % align: two lines pinned to the same '=' column (use only when that's the intent)
   \begin{align}
     q_t      &= \mathrm{clip}(\dots),\\
     I_{e,t+1}&= I_{e,t} + \dots,
   \end{align}

   % gather: each line independently centered (use for unrelated statements)
   \begin{gather}
     [K_p, K_i, K_w]_t = \mu_\theta(\mathbf{o}_t)\\
     q_t = \mathrm{clip}(K_p e_t + K_i I_{e,t}, q_{\min}, q_{\max})
   \end{gather}
   ```

3. **Trailing punctuation inside display equations.** House style in this
   report is no comma or period directly before `\\` or before
   `\end{equation|align|gather}` inside display math — punctuate the
   surrounding prose sentence instead. Strip any such marks when found.

4. **Table/figure placement fighting the two-column layout.** Prefer `[H]`
   (via the `float` package) for single-column tables that must stay next to
   their reference in text; use `figure*`/`table*` with `[!t]` for full-width
   figures/tables that need both columns (e.g. wide result scoreboards).

5. **`\resizebox` for wide tables.** A table with many numeric columns
   (e.g. a 5-controller scoreboard) should use
   `\resizebox{\columnwidth}{!}{...}` rather than shrinking font size
   manually or letting it overflow.

## Steps

1. Locate the offending equation/table/figure from the screenshot or
   description — search for a distinctive fragment of the surrounding text
   in `sourcecode/*.tex`.
2. Apply the narrowest fix that solves the reported problem (don't restyle
   unrelated equations while you're in the file).
3. Recompile: `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`.
   Confirm exit code 0 and check the log for `Output written on main.pdf` with
   no `Overfull \hbox` warnings pointing at the fixed region.
4. If the log shows a new `Overfull \hbox` at the same line range, the fix
   didn't fully resolve the width problem — narrow the content further
   (shorter variable names in the equation, or split into more lines) rather
   than declaring it fixed.

## Output

State what was changed and confirm the recompile succeeded (page count,
"no errors"). Then hand off to [commit-report](../commit-report/SKILL.md).
Do not push automatically.
