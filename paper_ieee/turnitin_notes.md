# Similarity and AI-use checks before submission

Targets (ICRCICN 2027, see venue_note.md): overall similarity at most 10%,
no single source above 1%. NMIMS project rule: similarity including AI
similarity under 10% (capstone guidelines, 10-mark criterion). Check the exact
NMIMS AI-score threshold and which report the library issues with your
mentor or the NMIMS library before you rely on it.

## 1. Run the check
1. Ask the NMIMS library (or your mentor) for a Turnitin submission through
   the institutional account. ICRCICN also accepts an iThenticate report.
   Do not use free online checkers: they may store and later match your text.
2. Submit `main.pdf` (or `main.docx`; Turnitin reads both). Submit the final
   text, after replacing the email placeholders.
3. Download two things: the Similarity Report (with the source list) and the
   AI writing report, if the NMIMS licence includes it.
4. Keep the PDF reports. Upload the similarity report to CMT as a
   supplementary file if the portal allows it, and keep it for the capstone
   file.

## 2. Result of the first check (v1, 11 Oct 2026) and what changed
- v1 scored 22% overall: arxiv.org 4%, regisportfolio.com 2%, export.arxiv.org
  1%, epub.fh-joanneum.at 1%, and 29 sources under 1% each. No AI-writing score
  was included.
- similarity/match_inventory.md lists every highlighted span. About 75-80% of
  the 22% (about 17 points) is the reference list. The affiliation and one
  table caption add about 0.5 point. Body text adds about 3-4 points.
- v2 (current main.pdf) rewrites all 16 highlighted body spans, the Table I
  caption, the acknowledgement and the AI disclosure, plus nearby sentences
  with the same stock wording (similarity/changes.md, main_diff.pdf). Journal
  names in the reference list now use standard IEEE/ISO 4 abbreviations. No
  other bibliographic fact changed.
- The affiliation line cannot change. It matches another NMIMS paper (<1%).

## 3. Recheck: settings to request
Ask the library to run a new submission of the v2 main.pdf with these
settings. Ask before using any filter that ICRCICN or NMIMS has not approved.
1. **Exclude bibliography.** This is a standard Turnitin option. It removes
   most of the v1 score. Without it, reference titles and author lists alone
   come to about 15-17%, so no rewrite of the body can reach 10%.
2. **Exclude quoted material.** The paper has almost no quotes, so this is a
   small effect. The one yellow "missing quotations" block in v1 was
   reference [5].
3. **Exclude small matches** under 8 to 10 words, if NMIMS allows it. This
   removes leftover hits on fixed technical strings (parameter lists,
   equation tokens).
4. **Exclude the group's own earlier submissions** (Review 2 draft, interim
   report) if they are in the repository. Do this only through the library.
5. **Request the AI-writing report as well.** The v1 report had no AI score.
   The NMIMS rubric counts AI similarity inside the 10% criterion, so the team
   needs that number. Ask which threshold NMIMS applies.
6. **1% per-source rule (ICRCICN).** With filters on, open every source of 1%
   or more. Rewrite any body sentence that still matches it. Expected after
   v2: no body source above 1%. If filters are refused, arxiv.org and
   regisportfolio.com will probably stay above 1% because of reference
   titles only. In that case write to the program chairs and explain this
   with the report.
7. Record the filters used in the CMT comments box, and keep both reports
   (unfiltered and filtered).
Never exclude a real body match just to get under the limit.

How to rewrite a flagged sentence (do it yourself, not with a paraphrasing tool):
1. Close the source. Write what the sentence must say in your own words.
2. Change the structure, not just words: split long sentences, put the
   result first, use our own symbols and numbers.
3. Keep the citation. Paraphrase still needs the reference.
4. Re-run the check after edits.

## 4. AI writing score and IEEE disclosure
- An AI assistant drafted the text of this paper, and the v2 rewrites are
  also machine edits. Expect a high AI-writing score until each author
  rewrites their sections by hand. Keep every number, symbol, citation and
  table reference unchanged. Write the way you would explain the work in a
  viva.
- Suggested split (each author rewrites their own part and then checks the
  others'):
  - **Gaurang Gupta (corresponding):** Abstract, I Introduction (including the
    contribution list), VI Conclusion, and the Acknowledgement/AI-use
    sentence. These carry the framing, so the corresponding author should
    own them.
  - **Arush Gupta:** II Related Work and III Method (A-D), equation
    lead-ins included. Related Work is the densest stock-phrase area. In v1
    most of its body matches came from there.
  - **Utkarsh Nigam:** IV Experiments (setup, baselines, Tables I-V text,
    synthetic and arbitrage subsections), V Discussion, and all figure and
    table captions.
- After the rewrite, rebuild (README.txt), run `python3 make_citation_audit.py`,
  confirm the paper is still at most 6 pages, then submit once for both the
  similarity report and the AI report.
- IEEE policy (IEEE Author Center, Conferences, "Submission Policies",
  Guidelines for AI-Generated Text): content generated by AI (text, figures,
  images, code) must be disclosed in the Acknowledgments. Name the AI system,
  identify the sections that use AI-generated content, and briefly explain
  the level of use. Use only for grammar editing does not need disclosure,
  but disclosure is recommended.
  https://conferences.ieeeauthorcenter.ieee.org/author-ethics/guidelines-and-policies/submission-policies/
- The current disclosure reads: "Use of AI: Grok Bot, an AI assistant,
  produced the experiment scripts, the figures and first drafts of every
  section. We checked, edited and verified all of it, and we are responsible
  for the content." After your own rewrite, update it so it stays true, for
  example "first drafts of every section, which the authors then rewrote".
  Do not delete it to lower a score.
- Authors stay responsible for every number, claim and reference. All numbers
  trace to CSVs under results/ (tables/SOURCES.md); all references are in
  citation_audit.md.

## 5. Before clicking Submit
- [ ] Overall similarity <= 10%, every single source <= 1% (report saved)
- [ ] NMIMS AI-similarity under its threshold (report saved)
- [ ] Filters used (if any) are allowed and written down
- [ ] AI-writing report obtained (it was missing from the v1 report)
- [ ] AI-use disclosure in the Acknowledgment matches what was done
- [ ] Rebuilt main.pdf after edits; still <= 6 pages; citation audit re-run
      (`python3 make_citation_audit.py` after the LaTeX build)
