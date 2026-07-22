---
name: write-section
description: Draft or revise a numbered section of the SCF-RL LaTeX report (sourcecode/NN_name.tex) — methodology, results, abstract, conclusions — using the project's established terminology, phase numbering, equations, and citation style. Use when asked to write, expand, or rewrite report content.
---

# Write Section

Drafts or revises content inside the report's `sourcecode/NN_name.tex` files,
keeping new prose consistent with the project's established facts, phase
structure, and notation.

## When to use this skill

- Writing a new section or subsection from scratch.
- Expanding or rewording an existing section (e.g. adding a justification
  subsection, tightening a paragraph, adding a table).
- Adding equations that describe BC/CQL/DDPG/TD3 objectives or the CIRL action
  mapping.

Do not use this skill for pure LaTeX layout fixes with no content change
(overflowing equations, punctuation, spacing) — use
[format-report](../format-report/SKILL.md) for those.

## Report location

Default: `../Report/sourcecode/` relative to this repo's root. Confirm the
path before writing if it's not there.

## Ground truth to write from

Before drafting, re-derive numbers from the actual code/notebooks in this
repo rather than trusting a prior report draft or memory — reports drift out
of date as experiments are re-run. That said, the pipeline structure is
stable and safe to assume:

**Four-phase pipeline**, each phase adding exactly one capability the
previous lacked:
1. **BC** — behavioural cloning of the anti-windup expert PI (`Kp=-0.5`,
   `Ti=300 s`, i.e. `Ki=Kp/Ti`, `Kw=Ki/Kp`), offline, on s20 (cloudy, 65 °C)
   + s21–s24 (sunny, 80 °C). Two parameterisations: Regular (predicts flow
   `q` directly, 10-D state incl. `q_prev`) and CIRL (predicts gains
   `[Kp,Ki,Kw]`, 9-D state, `Kw` pinned to `Ki/Kp` offline).
2. **CQL** — offline conservative Q-learning (CQL-H) on the same logged data,
   producing both an actor and a pessimistic critic.
3. **BC vs CQL comparison** — zero-shot and/or lightly-refined evaluation of
   both offline warm starts on four unseen sunny days (16–19 June 2026).
4. **Online fine-tuning** — DDPG/TD3, actor warm-started from BC, critic
   initialised from CQL, on the same four unseen days.

**Reward terms** (shared building blocks, combined differently per phase):
`r_track` (negative log-squared tracking error), `r_imit` (negative log-squared
distance to the expert's action), `r_smooth` (negative log-squared flow
change). State the per-phase weighting explicitly rather than a generic
formula — it differs across BC, CQL, and the online phases.

**CIRL vs regular RL justification**, if writing that argument: three legs —
safety (bounded PI structure guarantees stability even for an untrained/exploring
policy), sample efficiency and robustness (empirically: CIRL variants are far
flatter across dataset combinations than regular ones), interpretability
(gains are quantities a control engineer already reasons about). Cite
`bloor2025cirl` for the CIRL framework.

## Writing conventions (match the existing report)

- **Phase naming**: "Phase 1: Cloning the Expert Controller (BC)", "Phase 2:
  Learning a Value Function Offline (CQL)", "Phase 3: BC versus CQL on New
  Field Data", "Phase 4: Beating the Expert with DDPG and TD3" — keep these
  exact titles so cross-references stay meaningful.
- **Equations**: use `equation` for a single line, `gather` for multiple
  independently-centered lines that are not meant to align at `=` (e.g. two
  different mappings stated together), `align` only when the `=` signs should
  visually line up (e.g. a system of two related update equations). Never
  join two unrelated equations on one line with `\qquad` — it overflows the
  column in two-column layout.
- **No trailing punctuation inside display equations** — no comma or period
  immediately before `\\` or `\end{equation|align|gather}`. Punctuate in the
  surrounding prose instead.
- **Labels**: `eq:<short-name>`, `tab:<short-name>`, `fig:<short-name>`,
  `app:<short-name>` for appendix anchors. Always add a label to an equation
  you intend to reference later, even if the current draft doesn't reference
  it yet.
- **Units**: `\si{\celsius}`, `\si{L\per min}`, `\SI{value}{\celsius}` via
  `siunitx` — never bare `^\circ C` or manual unit strings.
- **Citations**: `\cite{key}`, keys defined in `references.bib`
  (`kumar2020conservative` for CQL, `kumar2022when` for the offline-RL-beats-BC
  argument, `fujimoto2018addressing` for TD3, `lillicrap2019continuous` for
  DDPG, `patterson2024empirical` for the seed-robustness justification,
  `bloor2025cirl` for CIRL, `camacho2007surveyI` for anti-windup PI).
- **Numbers**: report point estimates with method (best/worst/mean), and
  seed-robustness ranges where the repo has run multi-seed studies. Don't
  round away a number's precision inconsistently across sections — if the
  abstract says `0.245 ± 0.004`, the results section states the same figure.
- **Tone**: state results and their mechanism (why CQL beats BC, why TD3's
  safeguards can slow a critic-driven run when the critic is already
  pessimistic) rather than only reporting the number.

## Steps

1. Identify the target file (`sourcecode/NN_name.tex`) and read it in full,
   plus the sections immediately before/after it (numbers and claims must
   stay consistent across section boundaries — abstract, results, and
   conclusions in particular tend to repeat the same headline figures).
2. If the section reports a number, verify it against the source (the
   relevant `evaluate/*.ipynb` output or an existing table in the report)
   rather than copying a number from another section without checking.
3. Draft the content following the conventions above.
4. Check every new `\label` is unique in the document and every new `\ref` /
   `\cite` resolves (`grep` the file and its neighbours for the key).
5. Hand off to [format-report](../format-report/SKILL.md) to catch layout
   issues, then [review-report](../review-report/SKILL.md) for a consistency
   pass before committing.

## Output

Report which file(s) changed and a one-line summary of the content added or
revised. Don't compile or commit as part of this skill — that's
[format-report](../format-report/SKILL.md) and
[commit-report](../commit-report/SKILL.md).
