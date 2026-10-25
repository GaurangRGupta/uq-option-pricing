"""Assemble references.bib: verified entries copied from paper_ieee/references.bib plus new_refs.bib."""
import re
from pathlib import Path

src = Path("paper_ieee/references.bib").read_text(encoding="utf-8")
entries = re.split(r"\n(?=@)", src)
want = ["black1973pricing", "cox1979option", "heston1993closed", "albrecher2007little", "longstaff2001valuing",
        "inthout2010adi", "raissi2019physics", "vovk2005algorithmic", "bostrom2020mondrian", "angelopoulos2023conformal",
        "lei2018distribution", "romano2019conformalized", "paszke2019pytorch"]
out = []
for k in want:
    e = [x for x in entries if re.match(r"@\w+\{" + k + ",", x)]
    assert e, k
    out.append(e[0].rstrip())
head = ("% Entries copied from paper_ieee/references.bib (checked there against Crossref) plus new_refs.bib.\n"
        "% The five entries from new_refs.bib are not yet verified against Crossref/arXiv: check before submission.\n\n")
Path("paper_heston/references.bib").write_text(head + "\n\n".join(out) + "\n\n" + Path("paper_heston/new_refs.bib").read_text(encoding="utf-8"), encoding="utf-8")
print(len(out) + 5, "entries")
