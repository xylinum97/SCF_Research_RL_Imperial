---
name: init-report
description: Scaffold a fresh checkout of the SCF-RL LaTeX report — directory structure, main.tex, numbered section stubs, references.bib — when no report checkout exists yet. Use when starting the report from scratch or re-cloning it locally.
---

# Init Report

Scaffolds a new working checkout of the report *Direct Industrial Implementation of
Reinforcement Learning based SCF Control: Leveraging Behavioral Cloning for
Actor-Critic Policy Initialization* (Argha Pradipta, CID 06058496, supervised by
Antonio Del Rio Chanona, Department of Chemical Engineering, Imperial College
London).

## When to use this skill

- No local checkout of the report exists yet (first-time setup on a new machine).
- The GitHub repository for the report exists but is empty or you're bootstrapping
  its initial structure.
- You need a disposable scratch copy of the report skeleton to experiment with
  layout changes before touching the real checkout.

Do **not** use this skill if a report checkout already exists at the expected
location — use [write-section](../write-section/SKILL.md) to edit it instead.
Re-running this skill against an existing checkout must never overwrite content;
confirm with the user first if `../Report` already has commits.

## Report location

Default: `../Report` relative to this repository's root, i.e. a sibling of
`SCF_Research_RL_Imperial/` inside the parent working directory. If the user
names a different path, use that instead and remember it for the rest of the
session.

## Steps

1. **Check before creating.** Run `git status` / `ls` on the target directory.
   If it already contains a `.git` with commits, stop and ask the user whether
   they want a fresh scaffold or to resume editing the existing one.

2. **Directory layout.** Create:
   ```
   Report/
   ├── main.tex
   ├── references.bib
   └── sourcecode/
       ├── 00_titlepage.tex
       ├── 01_header_abstract.tex
       ├── 02_introduction.tex
       ├── 03_simulation_methodology.tex
       ├── 04_results_discussion.tex
       ├── 05_conclusions.tex
       └── 09_supplementary_information.tex
   ```
   Numbering leaves gaps (06–08) for sections added later (e.g. literature
   review, nomenclature) without renumbering existing files.

3. **`main.tex` skeleton** — two-column IEEE-style article, with the title
   block living in `01_header_abstract.tex` inside a `\twocolumn[...]` block:
   ```latex
   \documentclass[10pt,twocolumn]{article}
   \usepackage[utf8]{inputenc}
   \usepackage{amsmath,amssymb}
   \usepackage{graphicx}
   \usepackage{booktabs}
   \usepackage{siunitx}
   \usepackage{hyperref}
   \usepackage[margin=0.75in]{geometry}

   \begin{document}
   \twocolumn[
     \input{sourcecode/01_header_abstract}
   ]

   \input{sourcecode/02_introduction}
   \input{sourcecode/03_simulation_methodology}
   \input{sourcecode/04_results_discussion}
   \input{sourcecode/05_conclusions}

   \bibliographystyle{unsrt}
   \bibliography{references}

   \input{sourcecode/09_supplementary_information}
   \end{document}
   ```

4. **`01_header_abstract.tex` stub** — title, author, affiliation, empty abstract:
   ```latex
   \begin{center}
     {\Large\bfseries Direct Industrial Implementation of Reinforcement Learning
      based SCF Control: Leveraging Behavioral Cloning for Actor-Critic Policy
      Initialization \par}
     \vspace{0.6em}
     {\normalsize Argha Pradipta\textsuperscript{1} \par}
     \vspace{0.3em}
     {\small\itshape \textsuperscript{1}Department of Chemical Engineering, Imperial College London, SW7 2AZ, United Kingdom. \par}
   \end{center}
   \vspace{1em}
   \noindent\textbf{Abstract}\\[0.4em]
   \noindent
   % TODO: one-paragraph summary once Phase 4 results are final
   ```
   Add co-authors/supervisor only if the user confirms they should appear on
   the report (a supervisor is not automatically a co-author).

5. **Other section stubs** — each file starts with a comment banner naming
   the section and a one-line note on scope, matching the style already used
   in this project, e.g.:
   ```latex
   %==============================================================================
   %  03 - SIMULATION METHODOLOGY
   %  Reflects the implemented mini-project (BC -> CIRL TD3/DDPG).
   %==============================================================================
   \section{Simulation Methodology}
   ```

6. **`references.bib`** — start with an empty file plus the keys already
   known to be needed from the codebase's literature basis (fill in full
   entries as citations are added): `camacho2007surveyI`, `bloor2025cirl`,
   `kumar2020conservative`, `kumar2022when`, `levine2020offline`,
   `fujimoto2018addressing`, `lillicrap2019continuous`,
   `patterson2024empirical`, `gil2026bioprocess`, `wang2025offline`.

7. **Git init.** If the target has no `.git`, run `git init`, add a `.gitignore`
   covering LaTeX build artefacts (`*.aux`, `*.log`, `*.out`, `*.fls`,
   `*.fdb_latexmk`, `*.synctex.gz`, `*.bbl`, `*.blg`), and make an initial
   commit. Do not add a remote or push without the user's explicit go-ahead —
   that's [push-to-github](../push-to-github/SKILL.md)'s job.

8. **Verify it builds.** Run `latexmk -pdf -interaction=nonstopmode
   -halt-on-error main.tex` and confirm a PDF is produced with no errors
   before reporting the scaffold as done.

## Output

Report back the created path, the section list, and whether the initial build
succeeded. Do not commit or push automatically — hand off to
[commit-report](../commit-report/SKILL.md) once the user has reviewed the
scaffold.
