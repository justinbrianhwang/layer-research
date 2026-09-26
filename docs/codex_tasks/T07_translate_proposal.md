# T07 — English version of the research proposal

Source (Korean, do not modify): `private/vision_intervention_site_selection_proposal.ko.md` (977 lines).
Target (overwrite): `vision_intervention_site_selection_proposal.md` at the repo root, **entirely in English**.

## Requirements

- A faithful, complete translation, section by section, in the same order with the same heading
  levels, tables, bullet lists, blockquotes, LaTeX (`$...$`, `$$...$$`) and reference markers
  ([R1]–[R15]). Do not summarise, do not drop sections, do not add content or claims.
- Register: precise academic English as in an ML research proposal (NeurIPS/ICML style). Prefer the
  terminology already used in `README.md` and the code: "block", "site", "partial patching",
  "learned adapter", "budget", "regret", "observed / unseen corruptions", "fit / score / val / test
  splits", "clean-accuracy tolerance", "equivalence set", "positive control".
- The Korean title becomes: `# Does Representation Change Predict Where to Repair?` with the
  subtitle `## Reliability of Layer Selection in Vision Models under a Limited Intervention Budget`.
  Keep the existing English title line ("From Representation Change to Repair: ...") as it is.
- Header metadata lines (document type, subject, literature check date 2026-09-24) are translated too.
- Keep every reference entry (authors, venues, links) exactly as in the source; translate only the
  Korean sentences around them.
- Markdown must render on GitHub: math blocks on their own lines, tables with header separators, no
  HTML.
- Work in chunks (e.g. 150–200 lines at a time) and append; at the end verify that the number of
  `##`/`###` headings, `$$` blocks, tables and `[R..]` markers matches the source and print those counts
  for both files.

Do not touch any other file. Do not commit. Report the heading/equation/table counts for source and target.
