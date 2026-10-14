"""Best-effort Word copy: flatten main.tex (inline tables and values, PNG figures)
and convert with pandoc + IEEE CSL. Run after the LaTeX build:  python3 build_docx.py"""
import re
import subprocess
from pathlib import Path

H = Path(__file__).resolve().parent
s = (H / "main.tex").read_text()
s = re.sub(r"\\tabinput\{([^}]+)\}", lambda m: (H / m.group(1)).read_text(), s)
vals = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", (H / "tables" / "values.tex").read_text()))
for k, v in vals.items():
    s = s.replace("\\" + k + "{}", v)
s = s.replace("\\input{tables/values.tex}", "")
s = s.replace(r"\makeatletter\newcommand{\tabinput}[1]{\@@input #1 }\makeatother", "")
s = s.replace(r"\DeclareMathOperator*{\median}{median}", "").replace(r"\median", r"\operatorname{median}")
s = s.replace(r"\newcommand{\usd}{\$}", "").replace(r"\usd", r"\$")
s = re.sub(r"(figures/[^}]+)\.pdf", r"\1.png", s)
# The author block: pandoc drops IEEEtran author macros, so give it a plain author line.
s = s.replace(r"\maketitle", r"\maketitle" + "\n\n" + r"\noindent Gaurang Gupta (GAURANG.GUPTA10@nmims.in), Arush Gupta (ARUSH.GUPTA53@nmims.in), Utkarsh Nigam (UTKARSH.NIGAM54@nmims.in). MPSTME, SVKM's NMIMS, Mumbai, India." + "\n")
flat = H / "_main_flat.tex"
flat.write_text(s)
csl = (H / "ieee.csl").read_text()
csl = re.sub(r'et-al-min="\d+"', 'et-al-min="100"', csl)
csl = re.sub(r'et-al-use-first="\d+"', 'et-al-use-first="100"', csl)
(H / "_ieee_noetal.csl").write_text(csl)
subprocess.run(["pandoc", str(flat), "-s", "-o", str(H / "main.docx"), "--citeproc", "--bibliography", str(H / "references_ieeeabbrv.bib"),
                "--csl", str(H / "_ieee_noetal.csl"), "--resource-path", str(H), "-M", "reference-section-title=References"], check=True, cwd=H)
flat.unlink()
(H / "_ieee_noetal.csl").unlink()
print("wrote", H / "main.docx")
