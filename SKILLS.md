# Skills Index

Skills for maintaining the SCF-RL LaTeX report (Argha Pradipta, Imperial College
London — *Direct Industrial Implementation of Reinforcement Learning based SCF
Control: Leveraging Behavioral Cloning for Actor-Critic Policy Initialization*).

These skills operate on the report checkout, resolved as described in each
skill's "Report location" section (default: `../Report` relative to this repo).
See [CLAUDE.md](CLAUDE.md) for the project's technical background.

| Skill | Use it to |
|---|---|
| [init-report](.claude/skills/init-report/SKILL.md) | Scaffold a fresh report checkout (structure, `main.tex`, section stubs, `references.bib`) when no report exists yet |
| [write-section](.claude/skills/write-section/SKILL.md) | Draft or revise a numbered report section (`sourcecode/NN_name.tex`) with correct labels, citations, and terminology |
| [review-report](.claude/skills/review-report/SKILL.md) | Check the report for consistency: numbers matching across sections, dangling `\ref`s, terminology drift, phase numbering |
| [format-report](.claude/skills/format-report/SKILL.md) | Fix LaTeX layout issues — equation overflow, punctuation, table/figure placement — and recompile |
| [commit-report](.claude/skills/commit-report/SKILL.md) | Recompile, verify, and commit report changes with a clear message |
| [push-to-github](.claude/skills/push-to-github/SKILL.md) | Sync the local report checkout with GitHub, reconciling divergent Overleaf edits safely |
| [generate-pr](.claude/skills/generate-pr/SKILL.md) | Open a pull request for a batch of report changes with a proper summary and test plan |

## Typical flow

```
init-report (once)
   └─▶ write-section (per section, repeatable)
          └─▶ format-report (fix layout issues introduced by the edit)
                 └─▶ review-report (consistency pass before committing)
                        └─▶ commit-report
                               └─▶ push-to-github
                                      └─▶ generate-pr (if working on a branch)
```

Not every step is needed every time — a one-line wording fix only needs
`commit-report` → `push-to-github`; a new section needs the full chain.
