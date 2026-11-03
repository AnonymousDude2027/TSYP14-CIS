"""``make report-pdf``: typeset the Markdown report with XeLaTeX.

The report is generated as Markdown (``report-text``) because every number in it
must come from ``results/``.  For submission it has to be a *document*, so this
module converts that Markdown to a minimal LaTeX source and runs ``xelatex``
twice (once for the page count, once for the cross-references).

Design choices, and why:

* **XeLaTeX + fontspec** rather than pdfLaTeX: the report contains λ, σ, ≥, →,
  ×, −, µ, °, – and ’ in running text, and a Unicode engine renders them without
  a per-character escape table.  DejaVu Serif/Sans/Mono are used because they are
  present on any Debian/Ubuntu image and carry those glyphs.
* **No colour anywhere.**  Rules are ``booktabs`` rules, links are hidden, and
  nothing else is styled.  A report is judged on its content; colour is a way of
  making a jury suspect that the layout is doing the work.
* **``longtable`` + ``booktabs``** for the tables, with the header repeated on
  every page, so a table that runs over a page break stays readable.  Wide
  tables drop to ``\\scriptsize`` instead of being rotated, because a rotated
  table is harder to read on screen than a small one.
* Figures arrive embedded as base64 ``<img>`` tags in the Markdown (so that
  ``results/static_report.html`` stays self-contained); they are unpacked to
  ``docs/figures/report_figN.png`` for LaTeX.
"""
from __future__ import annotations

import base64
import re
import shutil
import subprocess
import sys
from pathlib import Path

from . import config

PREAMBLE = r"""\documentclass[10pt,a4paper]{article}
\usepackage[margin=20mm,top=17mm,bottom=17mm]{geometry}
\usepackage{fontspec}
\setmainfont{texgyretermes}[Extension=.otf,UprightFont=*-regular,BoldFont=*-bold,
  ItalicFont=*-italic,BoldItalicFont=*-bolditalic]
\setsansfont{texgyreheros}[Extension=.otf,UprightFont=*-regular,BoldFont=*-bold,
  ItalicFont=*-italic,BoldItalicFont=*-bolditalic,Scale=0.94]
\setmonofont{texgyrecursor}[Extension=.otf,UprightFont=*-regular,BoldFont=*-bold,
  ItalicFont=*-italic,BoldItalicFont=*-bolditalic,Scale=0.90]
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{graphicx}
\usepackage{fancyvrb}
\usepackage{fvextra}
\usepackage{amsmath,amssymb}
\usepackage{caption}
\captionsetup{font=small,labelfont=bf,justification=justified,singlelinecheck=false}
\usepackage{titlesec}
\titleformat{\section}{\normalfont\large\bfseries}{}{0pt}{}
\titleformat{\subsection}{\normalfont\normalsize\bfseries}{}{0pt}{}
\titleformat{\subsubsection}{\normalfont\normalsize\itshape}{}{0pt}{}
\titlespacing*{\section}{0pt}{11pt plus 2pt minus 1pt}{4pt}
\titlespacing*{\subsection}{0pt}{9pt plus 2pt minus 1pt}{3pt}
\titlespacing*{\subsubsection}{0pt}{7pt plus 1pt}{2pt}
\usepackage{enumitem}
\setlist[itemize]{leftmargin=1.2em,topsep=2pt,itemsep=1pt,parsep=0pt}
\setlist[enumerate]{leftmargin=1.4em,topsep=2pt,itemsep=1pt,parsep=0pt}
\usepackage[hidelinks]{hyperref}
\usepackage{microtype}
\setlength{\parindent}{0pt}
\setlength{\parskip}{3.2pt plus 1pt}
\setlength{\emergencystretch}{2.5em}
\renewcommand{\arraystretch}{1.04}
\begin{document}
\begin{center}
{\Large\bfseries %(title)s\par}
\vspace{3pt}
%(front)s
\end{center}
\vspace{-2pt}
"""

POSTAMBLE = "\n\\end{document}\n"

# Characters the body font may not carry, mapped to math equivalents.  XeLaTeX
# does not fail on a missing glyph, it drops it silently, which is worse: the
# report once printed "10 3" where it meant "10^-3".
GLYPHS = {
    "⁻¹": r"$^{-1}$", "⁻²": r"$^{-2}$", "⁻³": r"$^{-3}$",
    "⁻": r"$^{-}$", "¹": r"$^{1}$", "²": r"$^{2}$", "³": r"$^{3}$",
    "⁰": r"$^{0}$", "⁺": r"$^{+}$", "₀": r"$_0$", "₁": r"$_1$", "₂": r"$_2$",
    "≥": r"$\geq$", "≤": r"$\leq$", "≈": r"$\approx$", "×": r"$\times$",
    "σ": r"$\sigma$", "μ": r"$\mu$", "λ": r"$\lambda$",
    "−": r"$-$",
}

SPECIALS = {
    "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
    "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def _escape(text: str) -> str:
    out = "".join(SPECIALS.get(ch, ch) for ch in text)
    # a header such as ``L0\_provenance\_per\_reading`` is one unbreakable word to
    # TeX and overflows its column; the separators are safe break points.
    for sep in ("\\_", "/", "|", "-"):
        out = out.replace(sep, sep + r"\allowbreak{}")
    return out


INLINE = re.compile(
    r"`([^`]+)`"                                   # 1: inline code
    r"|\[([^\]]+)\]\((https?://[^)]+)\)"          # 2,3: link (label, url)
    r"|(" + "|".join(re.escape(g) for g in GLYPHS) + r")"   # 4: a glyph with a math form
)


def _markup(chunk: str) -> str:
    """Escape one plain chunk and apply bold/italic."""
    out = _escape(chunk)
    out = re.sub(r"\*\*([^*]+)\*\*", r"\\textbf{\1}", out)
    out = re.sub(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])", r"\\emph{\1}", out)
    return out


def _inline(text: str) -> str:
    """Escape a line and convert inline Markdown.

    One pass over an alternation rather than successive ``str.replace`` calls:
    the earlier version substituted sentinels (``\\x00<index>\\x00``) and then
    restored them, and a literal digit sitting between two sentinels matched the
    middle one -- ``≈1σ`` came out as a null byte, so the PDF lost a character.
    """
    out: list[str] = []
    pos = 0
    for m in INLINE.finditer(text):
        out.append(_markup(text[pos:m.start()]))
        if m.group(1) is not None:                      # inline code
            out.append(r"\texttt{" + _escape(m.group(1)) + "}")
        elif m.group(2) is not None:                    # link: label only
            out.append(_markup(m.group(2)))
        else:                                           # unicode -> math
            out.append(GLYPHS[m.group(4)])
        pos = m.end()
    out.append(_markup(text[pos:]))
    return "".join(out)


# Table headers are the CSV column names, which are precise but long; in print
# they are abbreviated so that a twelve-column table fits a portrait page at a
# readable size instead of being shrunk or rotated.  The CSV files keep the full
# names, so nothing is lost for a reader who wants to match a number to a file.
HEADERS = {
    "variable": "var.", "family": "family", "magnitude_sigma": "σ",
    "magnitude_nominal": "σ nom.", "magnitude_sigma_median": "σ med.",
    "magnitude_units_median": "units med.", "detection_rate": "det. rate",
    "never_detected_rate": "miss rate", "delay_median": "delay med.",
    "delay_q1": "q1", "delay_q3": "q3", "auprc_median": "AUPRC",
    "gain_mean_abs": "gain", "replay_length": "length",
    "events_per_year": "alarms/yr", "events_in_natural": "in extreme",
    "natural_days": "extreme d", "events_calibration": "calib. rate",
    "fa_target": "target", "excess_over_budget": "excess",
    "confidence_band": "band", "mean_confidence": "conf.", "accuracy": "acc.",
    "calibration_gap": "gap", "n_readings": "readings", "ms_per_reading": "ms/read",
    "peak_python_mb": "peak MB", "rss_mb": "RSS MB", "n_detected": "hits",
    "fa_per_year": "alarms/yr", "build_seconds": "build s", "fa_median": "FA med.",
    "fa_max": "FA max.", "detection_day": "day", "detectors_fired": "detectors",
    "delay_from_midpoint": "delay", "cumulative_bias_sigma": "cum. σ",
    "mean_bias_sigma": "mean σ", "n_days_affected": "days",
    "seconds": "s", "sigma_residual": "σ res.", "sigma_z": "σz",
    "seasonal_range": "range", "sigma_over_range": "σ/range",
    "ar_order": "AR", "rank_gauss": "rank-G", "fa_test_per_year": "FA test",
    "fa_calibration_per_year": "FA calib.", "n": "n", "sites": "sites",
    "mean_increment_sigma": "incr. σ", "achieved_sigma": "achiev. σ",
    "excess_over_budget": "over", "magnitude_units_median": "units",
    "magnitude_sigma_min": "σ min", "magnitude_sigma_max": "σ max",
}

READING_SUFFIX = "_per_reading"


LAYER = {"L0_provenance": "L0", "L2_detectors": "L2",
         "L3_consistency": "L3", "SP3_whitening": "SP3"}


def _abbrev(name: str) -> str:
    name = name.replace("meta_", "")
    if name in HEADERS:
        return HEADERS[name]
    if name.endswith(READING_SUFFIX):
        stem = name[: -len(READING_SUFFIX)]
        return LAYER.get(stem, stem) + " μs"
    return name.replace("meta_", "")


def _table(rows: list[str]) -> str:
    """Convert one Markdown pipe table to a booktabs table.

    Three fallbacks, chosen from a width model rather than by hand:

    1. plain ``l``/``r`` columns at the largest size that fits the text width;
    2. wrapped ``p`` columns for the few cells that are sentences (a detector
       list, a site name) -- an ``l`` column cannot break a line, which is what
       produced the overfull boxes in the first draft;
    3. a one-page ``\\resizebox`` for the handful of appendix tables that are
       wider than the page even at 6 pt.

    A ``longtable`` is used whenever the table may run over a page, so its header
    repeats; ``\\resizebox`` needs a single box, so those tables are ``tabular``.
    Landscape was tried and abandoned: inside ``lscape`` the text block is
    rotated but its width does not grow, so it only turned the page sideways.
    """
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    header, body = cells[0], cells[1:]
    if body and set("".join(body[0])) <= set("-: "):        # the alignment row
        body = body[1:]
    ncol = len(header)
    body = [c + [""] * (ncol - len(c)) for c in body]
    header = [_abbrev(h) for h in header]

    def token_len(text: str) -> int:
        return max((len(t) for t in re.split(r"[\s_/|]+", text) if t), default=0)

    widths = [max([len(_inline(c[i])) for c in body] + [len(header[i])])
              for i in range(ncol)]
    tokens = [max([token_len(c[i]) for c in body] + [token_len(header[i])])
              for i in range(ncol)]
    long_cols = [i for i in range(ncol)
                 if (tokens[i] > 13 and widths[i] > 18) or widths[i] > 40]

    # character width as a fraction of the portrait text width, per size
    SIZES = [(r"\small", 0.0098), (r"\footnotesize", 0.0089),
             (r"\scriptsize", 0.0082), (r"\tiny", 0.0073)]
    sept = "2pt" if ncol >= 12 else ("3pt" if ncol > 7 else "4.5pt")
    gap = 2 * float(sept[:-2]) / 486.0               # 2 * tabcolsep / textwidth
    # a plain (l/r) column cannot wrap, so its width is the whole cell, not the
    # longest word in it; using the token length here is what let a table of
    # sentences be typeset as `l` columns and run 1747 pt off the page
    natural = sum(max(widths[i], len(header[i])) for i in range(ncol))

    # Columns whose content is not a short number are allowed to wrap.
    text_cols = [i for i in range(ncol)
                 if widths[i] > 10 or i in long_cols]    # data width, not header

    size, mode, shares = SIZES[-1][0], "resize", {}
    for cand, charw in SIZES:
        if charw * natural + gap * ncol <= 0.99:
            size, mode = cand, "plain"
            break
        if text_cols:
            # the numeric columns keep their natural width; the wrapped ones share
            # what is left
            fixed = sum(charw * max(widths[i], len(header[i]))
                        for i in range(ncol) if i not in text_cols)
            room = 0.99 - gap * ncol - fixed
            # shares follow the *data*, not the header: a long column name must
            # not steal half the page from a column of numbers
            total = sum(max(widths[i], 12) for i in text_cols)
            cand_shares = {i: room * max(widths[i], 12) / total for i in text_cols}
            # a wrapped column narrower than ~8 % holds two characters a line
            if min(cand_shares.values()) >= 0.08:
                size, mode, shares = cand, "wrap", cand_shares
                break

    spec: list[str] = []
    for i in range(ncol):
        if mode == "wrap" and i in shares:
            spec.append(f">{{\\raggedright\\arraybackslash}}p{{{shares[i]:.3f}\\linewidth}}")
        elif mode != "wrap" and i == 0:
            spec.append("l")
        elif mode == "wrap" and i == 0:
            spec.append("l")
        else:
            spec.append("r")
    outline = "".join(spec)

    def row(cs: list[str]) -> str:
        return " & ".join(_inline(c) for c in cs[:ncol]) + r" \\"

    head = row(header)
    if mode == "resize":
        out = ["\\begin{center}", "\\resizebox{\\textwidth}{!}{%", size,
               f"\\setlength{{\\tabcolsep}}{{{sept}}}",
               f"\\begin{{tabular}}{{{outline}}}",
               r"\toprule", head, r"\midrule"]
        out += [row(r) for r in body]
        out += [r"\bottomrule", r"\end{tabular}}", r"\end{center}", ""]
        return "\n".join(out)

    # No ``center`` environment around a longtable: ``center`` is one box, so a
    # table taller than the remaining space jumps to the next page and leaves the
    # rest blank.  Centring uses longtable's own LTleft/LTright glue, which still
    # allows a page break.
    out = ["{" + size,
           f"\\setlength{{\\tabcolsep}}{{{sept}}}",
           r"\setlength{\LTleft}{\fill}",
           r"\setlength{\LTright}{\fill}",
           f"\\begin{{longtable}}{{{outline}}}",
           r"\toprule", head, r"\midrule", r"\endfirsthead",
           r"\toprule", head, r"\midrule", r"\endhead",
           r"\bottomrule", r"\endlastfoot"]
    out += [row(r) for r in body]
    out += [r"\end{longtable}", "}", ""]
    return "\n".join(out)


def _save_images(lines: list[str], figdir: Path) -> dict[int, Path]:
    """Unpack the inline base64 figures to real files; return {line index: path}."""
    figdir.mkdir(parents=True, exist_ok=True)
    found: dict[int, Path] = {}
    n = 0
    for i, line in enumerate(lines):
        m = re.match(r'\s*<img\s+src="data:image/(\w+);base64,([^"]+)"', line)
        if not m:
            continue
        n += 1
        out = figdir / f"report_fig{n}.png"
        out.write_bytes(base64.b64decode(m.group(2)))
        found[i] = out
    return found


def md_to_tex(md: str, figdir: Path) -> tuple[str, str, str]:
    """Return (title, front matter, body) as LaTeX."""
    lines = md.split("\n")
    figs = _save_images(lines, figdir)

    title = ""
    front: list[str] = []
    body: list[str] = []
    para: list[str] = []
    list_env: str | None = None
    i = 0

    def flush_para() -> None:
        if para:
            body.append(_inline(" ".join(para)))
            body.append("")
            para.clear()

    def close_list() -> None:
        nonlocal list_env
        if list_env:
            body.append(f"\\end{{{list_env}}}")
            body.append("")
            list_env = None

    while i < len(lines):
        raw = lines[i]
        line = raw.rstrip()
        stripped = line.strip()

        if i in figs:
            # a figure is introduced by its bold caption line, which precedes the image
            caption = ""
            prepared = False      # True when the caption is already LaTeX
            # the caption line was already flushed by the blank line before the
            # <img>, so it is the last non-empty paragraph in the body
            if not para:
                for k in range(len(body) - 1, -1, -1):
                    if body[k].strip():
                        if body[k].lstrip().startswith("\\textbf{Figure"):
                            txt = body[k].lstrip()
                            txt = txt[len("\\textbf{"):]
                            txt = txt.replace("}", "", 1) if txt.endswith("}") else txt
                            caption = txt.replace("[", " ").replace("]", " ").strip()
                            prepared = True
                            body.pop(k)
                        break
            if not caption and para and para[-1].lstrip().startswith("**Figure"):
                caption = re.sub(r"^\*\*Figure\s*\d+\s*[—-]\s*", "", para.pop().strip())
                caption = caption.rstrip("*").strip()
            flush_para()
            close_list()
            # the .tex is committed, so it must not carry the absolute path of the
            # machine that generated it: figures live in docs/figures/ next to it
            rel = figs[i].relative_to(figdir.parent).as_posix()
            # not a float: the figures sit where the text puts them, so no page
            # ends half empty waiting for a float to be placed
            body.append("\\begin{center}")
            body.append(f"\\includegraphics[width=0.98\\linewidth,height=0.34\\textheight,"
                        f"keepaspectratio]{{{rel}}}")
            if caption:
                body.append("\\par\\smallskip")
                body.append("\\parbox{0.94\\linewidth}{\\small\\itshape "
                            + (caption if prepared else _inline(caption)) + "}")
            body.append("\\end{center}\n")
            i += 1
            continue

        if not stripped:
            flush_para()
            close_list()
            i += 1
            continue

        if stripped.startswith("<!--"):
            i += 1
            continue

        if stripped == "---" or stripped == "***":
            flush_para(); close_list()
            i += 1
            continue

        m = re.match(r"^(#{1,4})\s+(.*)$", stripped)
        if m:
            flush_para(); close_list()
            level, text = len(m.group(1)), m.group(2)
            if level == 1 and not title:
                title = _inline(text)
            elif level <= 2:
                body.append(f"\\section*{{{_inline(text)}}}")
            elif level == 3:
                body.append(f"\\subsection*{{{_inline(text)}}}")
            else:
                body.append(f"\\subsubsection*{{{_inline(text)}}}")
            i += 1
            continue

        if stripped.startswith("```"):
            flush_para()
            close_list()
            i += 1
            code: list[str] = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code.append(lines[i])
                i += 1
            i += 1                       # closing fence
            body.append("\\begin{Verbatim}[breaklines=true,breakanywhere=true,"
                        "fontsize=\\footnotesize,frame=none,numbers=none]\n"
                        + "\n".join(code) + "\n\\end{Verbatim}\n")
            continue

        if stripped.startswith("|"):
            flush_para(); close_list()
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(lines[i])
                i += 1
            body.append(_table(rows))
            continue

        if re.match(r"^[-*]\s+", stripped):
            flush_para()
            if list_env != "itemize":
                close_list()
                body.append("\\begin{itemize}")
                list_env = "itemize"
            body.append("\\item " + _inline(re.sub(r"^[-*]\s+", "", stripped)))
            i += 1
            continue

        if re.match(r"^\d+[.)]\s+", stripped):
            flush_para()
            if list_env != "enumerate":
                close_list()
                body.append("\\begin{enumerate}")
                list_env = "enumerate"
            body.append("\\item " + _inline(re.sub(r"^\d+[.)]\s+", "", stripped)))
            i += 1
            continue

        if stripped.startswith(">"):
            flush_para(); close_list()
            body.append("\\begin{quote}" + _inline(stripped.lstrip("> ").strip()) + "\\end{quote}")
            i += 1
            continue

        para.append(stripped)
        i += 1

    flush_para()
    close_list()

    # the italic standfirst and the bold Abstract become the title block
    head: list[str] = []
    j = 0
    while j < len(body) and not body[j].strip():
        j += 1
    if j < len(body) and body[j].lstrip().startswith("\\emph{"):
        stand = body[j].lstrip()[6:].rstrip()
        stand = stand[:-1] if stand.endswith("}") else stand
        head.append("{\\small\\itshape " + stand + "\\par}")
        head.append("\\vspace{4pt}")
        j += 1
    while j < len(body) and not body[j].strip():
        j += 1
    if j < len(body) and "\\textbf{Abstract.}" in body[j]:
        txt = body[j].replace("\\textbf{Abstract.}", "").strip()
        head.append("\\parbox{0.96\\linewidth}{\\small\\textbf{Abstract.} " + txt + "}")
        j += 1
    rest = [b for b in body[j:]]
    return title, "\n".join(head), "\n".join(rest)


def _xelatex(tex: Path) -> Path:
    exe = shutil.which("xelatex") or "/tmp/tinytex/.TinyTeX/bin/x86_64-linux/xelatex"
    if not Path(exe).exists():
        raise SystemExit(
            "report-pdf needs XeLaTeX with booktabs/longtable/fontspec.\n"
            "  minimal install:  https://yihui.org/tinytex/  (TINYTEX_DIR=/tmp/tinytex)\n"
            "  then:             tlmgr install booktabs fontspec"
        )
    for _ in range(2):                      # twice: page count + cross-references
        p = subprocess.run([exe, "-interaction=nonstopmode", "-halt-on-error",
                            tex.name], cwd=tex.parent, capture_output=True, text=True)
        if p.returncode != 0:
            log = (tex.parent / (tex.stem + ".log"))
            tail = "\n".join(log.read_text(errors="replace").splitlines()[-25:]) if log.is_file() else p.stdout[-2000:]
            print(tail, file=sys.stderr)
            raise SystemExit(f"xelatex failed on {tex.name}")
    return tex.with_suffix(".pdf")


def build(md: Path, tex: Path, *, doc_title: str = "") -> Path:
    figdir = md.parent / "figures"
    title, front, body = md_to_tex(md.read_text(encoding="utf-8"), figdir)
    src = PREAMBLE % {"title": doc_title or title, "front": front} + body + POSTAMBLE
    tex.write_text(src, encoding="utf-8")
    return _xelatex(tex)


# One PDF that a reader can follow from "what is this" to "run it yourself".
# Every file below is maintained on its own; the PDF is a bound copy, so there is
# no second source of truth to fall out of date.
DOCUMENTATION = [
    ("README.md", None, "Overview"),
    ("docs/compliance.md", None, "Compliance with the challenge brief"),
    ("docs/threat_model.md", None, "Threat model"),
    ("docs/design.md", None, "Design decisions"),
    ("docs/data_sources.md", None, "Data sources"),
    ("docs/state_of_the_art.md", None, "State of the art"),
    ("docs/reproduce.md", None, "Reproducing every number"),
]


def build_documentation(out_tex: Path, *, title: str) -> Path:
    """Bind several Markdown files into one PDF, each starting on a new page."""
    root = config.repo_root()
    figdir = out_tex.parent / "figures"
    bodies: list[str] = []
    for rel, _unused, heading in DOCUMENTATION:
        md = root / rel
        if not md.is_file():
            print(f"[report-pdf] missing {rel}, skipped")
            continue
        _t, _front, body = md_to_tex(md.read_text(encoding="utf-8"), figdir)
        bodies.append("\\clearpage\n\\section*{" + _inline(heading) + "}\n" + body)
    src = PREAMBLE % {"title": title, "front": ""} + "\n".join(bodies) + POSTAMBLE
    out_tex.write_text(src, encoding="utf-8")
    return _xelatex(out_tex)


def run() -> int:
    docs = config.repo_root() / "docs"
    jobs = [("report.md", "report", ""),
            ("report_appendix.md", "appendix", "")]
    for src, stem, label in jobs:
        md = docs / src
        if not md.is_file():
            print(f"[report-pdf] missing {md}")
            continue
        pdf = build(md, docs / f"{stem}.tex", doc_title=label)
        try:
            from pypdf import PdfReader
            n = len(PdfReader(str(pdf)).pages)
        except Exception:                   # pragma: no cover
            n = "?"
        print(f"[report-pdf] {stem}.tex -> {pdf.name}: "
              f"{pdf.stat().st_size / 1024:.0f} KB, {n} pages")

    # the documentation bundle: README + the five technical notes + the run guide
    try:
        pdf = build_documentation(docs / "documentation.tex",
                                  title="Technical documentation — Track 2")
    except SystemExit as exc:                 # a missing section must not hide the report
        print(f"[report-pdf] documentation skipped: {exc}")
        return 0
    try:
        from pypdf import PdfReader
        n = len(PdfReader(str(pdf)).pages)
    except Exception:                         # pragma: no cover
        n = "?"
    print(f"[report-pdf] documentation.tex -> {pdf.name}: "
          f"{pdf.stat().st_size / 1024:.0f} KB, {n} pages")
    return 0
