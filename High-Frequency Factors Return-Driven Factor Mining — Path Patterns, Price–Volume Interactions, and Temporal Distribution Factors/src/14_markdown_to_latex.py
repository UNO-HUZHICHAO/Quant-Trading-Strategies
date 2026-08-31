# -*- coding: utf-8 -*-
"""将本项目预处理后的 Markdown 转成独立 LaTeX。

只实现研报实际使用的标题、段落、强调、行内代码、列表、表格和原生
LaTeX 块，避免依赖系统级 Pandoc。路径均相对项目根目录解析。
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = PROJECT_ROOT / "src" / "report_template_revised.tex"


def esc_plain(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "≤": r"\ensuremath{\leq}",
        "≥": r"\ensuremath{\geq}",
        "≈": r"\ensuremath{\approx}",
        "β": r"\ensuremath{\beta}",
        "①": "1.",
        "②": "2.",
        "③": "3.",
        "④": "4.",
        "⑤": "5.",
        "⑥": "6.",
        "⑦": "7.",
    }
    return "".join(replacements.get(char, char) for char in text)


TOKEN_RE = re.compile(r"(\$[^$]+\$|\*\*.+?\*\*|`[^`]+`)")


def inline(text: str) -> str:
    parts = TOKEN_RE.split(text)
    out: list[str] = []
    for part in parts:
        if not part:
            continue
        if part.startswith("$") and part.endswith("$"):
            out.append(part)
        elif part.startswith("**") and part.endswith("**"):
            out.append(r"\textbf{" + inline(part[2:-2]) + "}")
        elif part.startswith("`") and part.endswith("`"):
            out.append(r"\texttt{" + esc_plain(part[1:-1]) + "}")
        else:
            out.append(esc_plain(part))
    return "".join(out)


def is_table_sep(line: str) -> bool:
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)


def table_spec(ncols: int, rows: list[list[str]]) -> str:
    numeric = []
    for col in range(ncols):
        vals = [row[col].strip() for row in rows[1:] if col < len(row)]
        numeric.append(bool(vals) and all(re.fullmatch(r"[-+]?\d+(?:\.\d+)?%?", v) for v in vals))
    if ncols == 2:
        return r"@{}p{3.7cm}p{11.4cm}@{}"
    if ncols == 3:
        return r"@{}p{2.4cm}p{8.5cm}p{4.2cm}@{}"
    if ncols == 4:
        return r"@{}p{1.5cm}p{7.1cm}p{2.5cm}p{4.2cm}@{}"
    if ncols >= 5 and not numeric[0] and not numeric[-1]:
        middle = "".join("r" if flag else "l" for flag in numeric[1:-1])
        first_width = "2.2cm" if ncols >= 8 else "2.8cm"
        last_width = "3.2cm" if ncols >= 8 else "5.0cm"
        return rf"@{{}}p{{{first_width}}}{middle}p{{{last_width}}}@{{}}"
    return "@{}" + "".join("r" if flag else "l" for flag in numeric) + "@{}"


def render_table(rows: list[list[str]], caption: str | None) -> str:
    ncols = max(len(row) for row in rows)
    rows = [row + [""] * (ncols - len(row)) for row in rows]
    spec = table_spec(ncols, rows)
    out = [rf"\begin{{longtable}}{{{spec}}}"]
    if caption:
        out.append(rf"\caption{{{inline(caption)}}}\\")
    header = " & ".join(r"\textbf{" + inline(cell) + "}" for cell in rows[0]) + r" \\"
    out.extend([r"\toprule", header, r"\midrule", r"\endfirsthead", r"\toprule", header,
                r"\midrule", r"\endhead", r"\bottomrule", r"\endfoot"])
    for row in rows[1:]:
        out.append(" & ".join(inline(cell) for cell in row) + r" \\")
    out.append(r"\end{longtable}")
    return "\n".join(out)


def convert(text: str) -> str:
    lines = text.splitlines()
    out: list[str] = []
    paragraph: list[str] = []
    pending_caption: str | None = None

    def flush_paragraph() -> None:
        if paragraph:
            out.append(inline(" ".join(item.strip() for item in paragraph)))
            out.append("")
            paragraph.clear()

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            flush_paragraph()
            i += 1
            continue
        if stripped.startswith("Table: "):
            flush_paragraph()
            pending_caption = stripped[7:].strip()
            i += 1
            continue
        raw_begin = re.match(r"^\\begin\{([^}]+)\}", stripped)
        if raw_begin:
            flush_paragraph()
            env = raw_begin.group(1)
            depth = 0
            j = i
            while j < len(lines):
                current = lines[j]
                depth += len(re.findall(rf"\\begin\{{{re.escape(env)}\}}", current))
                depth -= len(re.findall(rf"\\end\{{{re.escape(env)}\}}", current))
                out.append(current)
                j += 1
                if depth == 0:
                    break
            i = j
            continue
        if stripped.startswith("|") and i + 1 < len(lines) and is_table_sep(lines[i + 1]):
            flush_paragraph()
            table_rows: list[list[str]] = []
            j = i
            while j < len(lines) and lines[j].strip().startswith("|"):
                if not is_table_sep(lines[j]):
                    table_rows.append([cell.strip() for cell in lines[j].strip().strip("|").split("|")])
                j += 1
            out.append(render_table(table_rows, pending_caption))
            out.append("")
            pending_caption = None
            i = j
            continue
        if stripped.startswith("## "):
            flush_paragraph()
            out.append(r"\section{" + inline(stripped[3:].strip()) + "}")
            out.append("")
            i += 1
            continue
        if stripped.startswith("### "):
            flush_paragraph()
            out.append(r"\subsection{" + inline(stripped[4:].strip()) + "}")
            out.append("")
            i += 1
            continue
        if stripped.startswith("#### "):
            flush_paragraph()
            out.append(r"\subsubsection{" + inline(stripped[5:].strip()) + "}")
            out.append("")
            i += 1
            continue
        if stripped.startswith("- "):
            flush_paragraph()
            items: list[str] = []
            j = i
            while j < len(lines) and lines[j].strip().startswith("- "):
                items.append(lines[j].strip()[2:].strip())
                j += 1
            out.append(r"\begin{itemize}")
            out.extend(r"\item " + inline(item) for item in items)
            out.append(r"\end{itemize}")
            out.append("")
            i = j
            continue
        if re.match(r"^\d+\.\s+", stripped):
            flush_paragraph()
            items: list[str] = []
            j = i
            while j < len(lines):
                match = re.match(r"^\d+\.\s+(.+)$", lines[j].strip())
                if not match:
                    break
                items.append(match.group(1))
                j += 1
            out.append(r"\begin{enumerate}")
            out.extend(r"\item " + inline(item) for item in items)
            out.append(r"\end{enumerate}")
            out.append("")
            i = j
            continue
        if stripped.startswith(">"):
            flush_paragraph()
            quote_lines: list[str] = []
            j = i
            while j < len(lines) and lines[j].strip().startswith(">"):
                quote_lines.append(lines[j].strip()[1:].strip())
                j += 1
            out.append(r"\begin{tcolorbox}[colback=reportgray,colframe=reportred!35,boxrule=.4pt]")
            out.append(inline(" ".join(quote_lines)))
            out.append(r"\end{tcolorbox}")
            out.append("")
            i = j
            continue
        if stripped == "---":
            flush_paragraph()
            i += 1
            continue
        if stripped.startswith("\\") or stripped.startswith("{"):
            flush_paragraph()
            out.append(line)
            i += 1
            continue
        paragraph.append(line)
        i += 1
    flush_paragraph()
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--template", type=Path, default=TEMPLATE)
    args = parser.parse_args()
    input_path = args.input if args.input.is_absolute() else PROJECT_ROOT / args.input
    output_path = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    template_path = args.template if args.template.is_absolute() else PROJECT_ROOT / args.template
    body = convert(input_path.read_text(encoding="utf-8"))
    template = template_path.read_text(encoding="utf-8")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(template.replace("$body$", body), encoding="utf-8")
    print(f"写出 {output_path}")


if __name__ == "__main__":
    main()
