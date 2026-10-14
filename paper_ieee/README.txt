ivuq IEEE conference paper (target: IEEE ICRCICN 2027, max 6 pages)
===================================================================

Files
  main.tex              the paper (IEEEtran conference class, Times via mathptmx)
  IEEEtran.cls/.bst     copies of the IEEE class and bibliography style (TeX Live 2025/dev, Debian)
  references.bib        verified references, master copy (see references_verification.md)
  references_ieeeabbrv.bib  GENERATED from references.bib by similarity/make_abbrev_bib.py
                        (journal names abbreviated IEEE/ISO 4 style); main.tex uses this.
                        After editing references.bib, re-run: python3 similarity/make_abbrev_bib.py
  main_v1.tex/.pdf      version checked by Turnitin on 11 Oct 2026 (22%)
  main_diff.tex/.pdf    latexdiff v1 -> current (struck = removed, wavy = added)
  similarity/           match_inventory.md, changes.md, rewrite_v2.py, changes.json
  main.pdf              compiled paper (5 pages; v2 = N2 bias fix, see V2_CHANGES.md; Turnitin-checked text = main_v2pre.tex/.pdf)
  main.docx             Word copy made with pandoc (best effort, see below)
  tables/*.tex, *.csv   every table, generated from ../results by make_tables.py
  tables/values.tex     two inline numbers (max coverage std, fitted kappa)
  tables/SOURCES.md     which results CSV each table comes from
  figures/*.pdf, *.png  the 3 figures used (fig_p2_*); figures/data/ holds the
                        CSVs behind the slice and residual figures. The bucket
                        figure is drawn from results/uq/paper/p1p2/p1p2_summary.csv.
  citation_audit.md     every [n] -> key -> printed reference -> sections
  cover_letter.txt, author_block.txt, turnitin_notes.md, venue_note.md

Build the PDF (from this folder)
  1. (only if results changed)  ..\.venv\Scripts\python make_tables.py
                                (Linux: ../.venv/bin/python make_tables.py)
  2. pdflatex main
     bibtex main
     pdflatex main
     pdflatex main
  Overleaf also works: upload this folder; the main file is main.tex and the
  compiler is pdfLaTeX. Keep the tables/ and figures/ folders.
  Note: tables are pulled in with the primitive \@@input (macro \tabinput)
  because LaTeX's \input cannot start a tabular row.

Word copy
  python3 build_docx.py   (needs pandoc >= 3; uses ieee.csl with "et al."
  switched off). Limits: the Word file is not in the IEEE two-column template,
  table rules and column spans are simplified, equations are Word equations
  converted by pandoc, figures are PNG. If ICRCICN asks for Word, paste the
  text into the official IEEE Word conference template and re-check layout.

Figures were re-rendered for the paper with a Times-compatible font
(Liberation Serif) from the same data: PAPER_FIG_OUT=<dir> python
scripts/paper_figures.py. The data CSVs are byte-identical to
results/uq/paper/figures/.

Checks done on the compiled PDF: 5 pages; no LaTeX errors, no overfull
boxes, no undefined references or citations, no "??"; all fonts embedded;
28 references, all cited, none with "et al.". "Abstract-" and
"Index Terms-" dashes and page-range dashes in references are part of the
IEEE template, not prose.
