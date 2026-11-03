"""Tests for the Markdown -> LaTeX converter behind ``make report-pdf``.

The converter is the one place where a mistake is invisible: a dropped glyph or a
swallowed brace still produces a PDF, so nothing downstream complains.  These
tests cover the cases that actually broke while it was written:

* a Unicode character that the text font lacks (``≈1σ`` printed as a null byte
  because the sentinel-restore scheme matched a digit);
* a literal ``_`` or ``%`` in prose, which LaTeX would otherwise interpret;
* a table whose header is a CSV column name (``detection_rate``) and whose first
  column is text, which must become a wrapped column rather than an unbreakable
  one that runs off the page;
* a fenced code block, which must not be parsed as prose.

No LaTeX installation is needed: the tests stop at the ``.tex`` source.
"""
from __future__ import annotations

import re

import pytest

from tsfdi import report_latex as R


# --------------------------------------------------------------- inline text
def test_no_sentinel_or_null_survives():
    out = R._inline("annual mean residual moves by ≈1σ over the record")
    assert "\x00" not in out
    assert r"$\approx$1$\sigma$" in out


def test_latex_specials_are_escaped():
    out = R._inline("50 % of a_b & c#1 costs $5")
    for raw in ("%", "&", "#", "$"):
        assert f"\\{raw}" in out
    assert "a\\_\\allowbreak{}b" in out


def test_bold_italic_and_code():
    out = R._inline("**bold** and *ital* and `code_here`")
    assert r"\textbf{bold}" in out
    assert r"\emph{ital}" in out
    assert r"\texttt{code\_\allowbreak{}here}" in out


def test_links_keep_the_label_only():
    out = R._inline("see [the guide](https://example.org/a_b) for more")
    assert "the guide" in out
    assert "http" not in out


# -------------------------------------------------------------------- tables
def _table(md: str) -> str:
    rows = [line for line in md.splitlines() if line.strip().startswith("|")]
    return R._table(rows)


def test_narrow_numeric_table_is_a_longtable_with_a_repeating_header():
    out = _table("| site | n |\n|:--|--:|\n| CARTHAGE | 12 |\n| GAFSA | 7 |")
    assert r"\begin{longtable}" in out
    assert out.count(r"\endfirsthead") == 1 and out.count(r"\endhead") == 1
    assert "CARTHAGE" in out


def test_columns_are_abbreviated_in_the_header_not_in_the_data():
    out = _table("| variable | detection_rate |\n|:--|--:|\n| TMAX | 0.883 |")
    assert "det. rate" in out
    assert "0.883" in out


def test_a_table_of_sentences_never_overflows():
    """Wide text columns must wrap: a plain ``l`` column cannot break a line."""
    rows = "| level | capability | honest statement |\n|:--|:--|:--|\n"
    rows += "".join(
        f"| A{i} | a capability that is described in a full sentence | "
        f"and an honest statement that is also a full sentence, twice as long |\n"
        for i in range(4))
    out = _table(rows)
    assert r"\resizebox" in out or ">{\\raggedright\\arraybackslash}p{" in out


def test_a_very_wide_table_falls_back_to_a_single_page_tabular():
    # sixteen columns each holding a long identifier: no size makes it fit a page
    # without wrapping every column to two characters a line
    ncol = 16
    head = "| " + " | ".join(f"identifier_number_{i:02d}_long" for i in range(ncol)) + " |"
    sep = "|" + "|".join([":--"] * ncol) + "|"
    body = "| " + " | ".join("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123" for _ in range(ncol)) + " |"
    out = _table("\n".join([head, sep, body]))
    assert r"\resizebox{\textwidth}{!}{" in out
    assert r"\begin{tabular}" in out


# ---------------------------------------------------------------- document
def test_front_matter_title_and_abstract(tmp_path):
    md = ("# A title with 50 % and a_b\n\n"
          "*A standfirst.*\n\n"
          "**Abstract.** The abstract text.\n\n"
          "## 1. Section\n\nBody paragraph.\n")
    title, front, body = R.md_to_tex(md, tmp_path)
    assert title == "A title with 50 \\% and a\\_\\allowbreak{}b"
    assert "standfirst" in front and "Abstract." in front
    assert r"\section*{1. Section}" in body
    assert "Abstract." not in body


def test_fenced_code_becomes_verbatim(tmp_path):
    md = "# T\n\n```\nmake all   # comment\nx = 1_000\n```\n"
    _t, _f, body = R.md_to_tex(md, tmp_path)
    assert r"\begin{Verbatim}" in body
    assert "x = 1_000" in body


def test_figures_are_written_next_to_the_markdown_and_referenced_relatively(tmp_path):
    import base64
    png = base64.b64encode(
        bytes.fromhex("89504e470d0a1a0a0000000d4948445200000001000000010806000"
                      "0001f15c4890000000a49444154789c6300010000050001") ).decode()
    md = (f"# T\n\n**Figure 1 — a caption.**\n\n"
          f'<img src="data:image/png;base64,{png}" style="width:100%">\n')
    _t, _f, body = R.md_to_tex(md, tmp_path / "figures")
    assert (tmp_path / "figures" / "report_fig1.png").is_file()
    assert "figures/report_fig1.png" in body
    assert "/home/" not in body          # no absolute path may leak into the .tex
    assert "a caption" in body
