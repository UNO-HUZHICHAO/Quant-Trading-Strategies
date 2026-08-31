# -*- coding: utf-8 -*-
r"""研报 Markdown → LaTeX 预处理（红色主题 + 图注修复 + 表注居中）。
确定性改写五处，其余原样交给 pandoc（保证无遗漏）：
  1) 标题/副标题/元信息块 → LaTeX titlepage + tableofcontents；
  2) 「**图 x-y　caption**」+ 图片行 → raw LaTeX figure（单图 / 并排双图 minipage [b] 对齐），
     图号红色加粗（修复"图"字丢失：捕获组含"图 "）；
  3) 每张表格前插入 `Table: 表 N　说明`（pandoc 转为居中 \caption，配合模板 labelformat=empty 防重号）；
  4) "## 报告要点" 标题 → 红色 tcolorbox 色带；
  5) 「因子 | 中性化 | …」因子总览表（每因子 S1/S2 两行）→ raw LaTeX longtable，
     因子名与判读用 \multirow 跨两行垂直居中，因子组间 \cmidrule 分隔。
默认输出：docs/研报V2_texready.md
新版用法：python 10_md_to_tex.py --input docs/新版.md --output docs/新版_texready.md
"""
import argparse
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS = PROJECT_ROOT / 'src' / 'report'
DEFAULT_MD = DOCS / 'report.md'
DEFAULT_OUT = DOCS / '研报V2_texready.md'

# 图注：捕获组必须含"图 "，否则"图"字丢失。
CAP_RE = re.compile(r'^\*\*(图 .+?)\*\*\s*$')
IMG_SPLIT = re.compile(r'!\[[^\]]*\]\((figs(?:_revised|_v3|_v21)?/[^)]+)\)')
TABLE_CAP_RE = re.compile(r'^\|(.+)\|\s*$')
TABLE_SEP_RE = re.compile(r'^\|[\s\-:|]+\|\s*$')
# 因子总览表（每因子 S1/S2 两行）：改由 multirow LaTeX 渲染。
OV_HEADER_RE = re.compile(r'^\|\s*因子\s*\|\s*中性化\s*\|')
OV_HEAD = '因子 & 中性化 & 沪深300 & 中证500 & 中证1000 & 中证全指 & 判读 \\\\'

# 表注（pandoc `Table:` caption）。键为表头首行片段（宽松匹配，不要求尾随管道符）；
# 值为说明文字或 None（None 用当前小节判定模块 A/B/C）。
TABLE_CAPS = [
    (r'^\| 模块 \| 研究问题 \|', '三模块设计总览'),
    (r'^\| 数据 \| 内容 \| 覆盖 \| 用途 \|', '数据说明'),
    (r'^\| 符号 \| 含义 \|', '符号约定（全文统一）'),
    (r'^\| 因子 \| 公式（日度原子量）', None),        # 模块 A/B/C 因子定义
    (r'^\| 因子 \| 中性化 \| 沪深300 \|', None),      # 模块 A/B/C 因子总览
    (r'^\| 年 \| 2016H2 \|', 'A3 分年 IC（中证1000，S1）'),
    (r'^\| 因子 \| 波动率 \| 换手率 \| 动量 \|', '18 因子 × 9 风格相关系数（S1 残差，中证全指）'),
    (r'^\| 组合（RankICIR）', 'S1 / S2 / S3 中性化对照（RankICIR）'),
    (r'^\| 股票池 \| (?:平均)?单组月换手', '单组月换手与多空年化成本拖累'),
    (r'^\| 组合 \| (?:毛多空年化|RankICIR)', '扣成本后净多空年化 > 5% 的组合'),
    (r'^\| 维度 \| 本研究的处理', '方法论定位与本研究贡献'),
    (r'^\| 因子（中文名）', '18 因子算子链速览'),
]
# 模块内"公式/总览"两类表：按当前小节号判定。
SUBSEC_TABLE_CAPS = {
    ('2.2', 'def'): '模块 A 因子定义与计算方法',
    ('3.2', 'def'): '模块 B 因子定义与计算方法',
    ('4.2', 'def'): '模块 C 因子定义与计算方法',
    ('2.4', 'ov'): '模块 A 因子总览（RankICIR，S1/S2 四池）',
    ('3.4', 'ov'): '模块 B 因子总览（RankICIR，S1/S2 四池）',
    ('4.4', 'ov'): '模块 C 因子总览（RankICIR，S1/S2 四池）',
}


def esc(s: str) -> str:
    s = (s.replace('\\', r'\textbackslash{}')
          .replace('&', r'\&').replace('%', r'\%').replace('#', r'\#')
          .replace('_', r'\_').replace('{', r'\{').replace('}', r'\}')
          .replace('~', r'\textasciitilde{}').replace('^', r'\^{}'))
    return s


def split_cap(cap: str):
    m = re.match(r'^(图 [\d-]+)　?(.*)$', cap)
    return (m.group(1), m.group(2)) if m else ('', cap)


def cap_latex(cap: str) -> str:
    num, rest = split_cap(cap)
    if num:
        return rf'\caption*{{\textbf{{\textcolor{{reportred}}{{{num}}}}}　{esc(rest)}}}'
    return rf'\caption*{{{esc(cap)}}}'


def fig_block(imgs: list[str], caps: list[str]) -> str:
    if len(imgs) == 1:
        return ('\\begin{figure}[htbp]\n\\centering\n'
                f'\\includegraphics[width=.92\\textwidth,height=.72\\textheight,keepaspectratio]{{{imgs[0]}}}\n'
                f'{cap_latex(caps[0])}\n'
                '\\end{figure}')
    parts = []
    for img, cap in zip(imgs, caps):
        parts.append('\\begin{minipage}[t]{0.49\\textwidth}\n\\centering\n'
                     f'\\includegraphics[width=\\linewidth]{{{img}}}\n'
                     f'\\vspace{{2pt}}{cap_latex(cap)}\n'
                     '\\end{minipage}')
    return ('\\begin{figure}[htbp]\n\\centering\n'
            + '\\hfill\n'.join(parts)
            + '\n\\end{figure}')


def titlepage(title: str, subtitle: str, meta: list[str], include_author: bool = True) -> str:
    t = ['\\begin{titlepage}', '\\centering', '\\vspace*{1.3cm}',
         f'{{\\Huge\\bfseries\\heiti\\textcolor{{reportred}}{{{esc(title)}}}}}\\\\[8pt]',
         f'{{\\Large {esc(subtitle)}}}\\\\[16pt]',
         '\\textcolor{reportred}{\\rule{0.6\\textwidth}{1pt}}\\\\[14pt]']
    meta_text = '\\\\[3pt]'.join(esc(m) for m in meta if m)
    if meta_text:
        t.append(f'{{\\small\\color{{black!60}} {meta_text}}}\\\\[22pt]')
    t += ['\\vfill']
    if include_author:
        t += ['{\\small\\color{black!50} 作者：}\\\\[2pt]',
              '{\\LARGE\\heiti\\textcolor{reportred}{胡智超}}\\\\[18pt]']
    t += ['{\\large\\heiti\\textcolor{reportred}{金融工程深度研究报告}}\\\\[6pt]',
          '{\\normalsize 2026 年 8 月}\\\\[1.5cm]',
          '\\end{titlepage}',
          '\\tableofcontents', '\\newpage']
    return '\n'.join(t)


def table_caption_for(first_header: str, subsection: str, table_no: int) -> str:
    for pat, cap in TABLE_CAPS:
        if re.search(pat, first_header):
            if cap:
                return f'表 {table_no}　{cap}'
            kind = 'def' if '公式' in first_header else 'ov'
            return f'表 {table_no}　{SUBSEC_TABLE_CAPS.get((subsection[:3], kind), "") or "因子表"}'
    return f'表 {table_no}'


def ov_cell(s: str) -> str:
    """表格单元格：LaTeX 转义 + md 粗体还原为 \\textbf{}。"""
    return re.sub(r'\*\*(.+?)\*\*', r'\\textbf{\1}', esc(s.strip()))


def overview_table(rows: list, caption: str) -> str:
    """因子总览表 → longtable + multirow：因子名与判读跨 S1/S2 两行垂直居中。

    rows 为数据行（已去表头与分隔行），每行 7 个单元格，S1/S2 成对。
    """
    out = ['\\begin{longtable}{@{}p{3.0cm} c r r r r p{5.2cm}@{}}',
           f'\\caption{{{caption}}}\\\\',
           '\\toprule', OV_HEAD, '\\midrule', '\\endfirsthead',
           '\\toprule', OV_HEAD, '\\midrule', '\\endhead',
           '\\bottomrule', '\\endfoot']
    for idx in range(0, len(rows), 2):
        r1, r2 = rows[idx], rows[idx + 1]
        out.append('\\multirow{2}{\\hsize}{%s} & %s & %s & %s & %s & %s & '
                   '\\multirow{2}{\\hsize}{%s} \\\\' % (
                       ov_cell(r1[0]), ov_cell(r1[1]), ov_cell(r1[2]), ov_cell(r1[3]),
                       ov_cell(r1[4]), ov_cell(r1[5]), ov_cell(r1[6])))
        out.append(' & %s & %s & %s & %s & %s & \\\\' % (
            ov_cell(r2[1]), ov_cell(r2[2]), ov_cell(r2[3]), ov_cell(r2[4]), ov_cell(r2[5])))
        if idx + 2 < len(rows):
            out.append('\\cmidrule(lr){2-6}')
    out.append('\\end{longtable}')
    return '\n'.join(out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='将研报 Markdown 预处理为 Pandoc 可用版本')
    parser.add_argument('--input', type=Path, default=DEFAULT_MD)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--no-author', action='store_true', help='标题页不显示作者姓名')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    md_path = args.input if args.input.is_absolute() else PROJECT_ROOT / args.input
    out_path = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    text = md_path.read_text(encoding='utf-8')
    lines = text.split('\n')

    k = 0
    while k < len(lines) and not lines[k].strip():
        k += 1
    title = lines[k].lstrip('#').strip()
    k += 1
    while k < len(lines) and not lines[k].strip():
        k += 1
    subtitle = lines[k].lstrip('#').strip()
    k += 1
    while k < len(lines) and not lines[k].strip():
        k += 1
    meta = []
    while k < len(lines) and lines[k].strip().startswith('>'):
        t = lines[k].strip()
        meta.append(t[1:].strip() if len(t) > 1 else '')
        k += 1
    while k < len(lines) and not lines[k].strip():
        k += 1
    if lines[k].strip() == '---':
        k += 1
    while k < len(lines) and not lines[k].strip():
        k += 1
    body = lines[k:]

    out = [titlepage(title, subtitle, meta, include_author=not args.no_author), '']
    i, n = 0, len(body)
    subsection = ''     # 当前 ### 小节号（如 "2.2"）
    table_no = 0
    while i < n:
        line = body[i]
        # 跟踪小节号
        sm = re.match(r'^### (\d\.\d)', line.strip())
        if sm:
            subsection = sm.group(1)
        # 图块
        cap_m = CAP_RE.match(line.strip())
        if cap_m:
            j = i + 1
            while j < n and not body[j].strip():
                j += 1
            img_line = body[j].strip() if j < n else ''
            imgs = IMG_SPLIT.findall(img_line)
            if imgs:
                caps = cap_m.group(1)
                cap_list = caps.split('　|　') if '　|　' in caps else [caps]
                out.append(fig_block(imgs, cap_list))
                out.append('')
                i = j + 1
                continue
        # 表格：表头行 + 下一行是分隔行
        if TABLE_CAP_RE.match(line) and i + 1 < n and TABLE_SEP_RE.match(body[i + 1]):
            table_no += 1
            cap = table_caption_for(line, subsection, table_no)
            if OV_HEADER_RE.match(line):
                # 因子总览表：收集 S1/S2 成对的数据行，转 multirow LaTeX
                j = i + 2
                rows = []
                while j < n and TABLE_CAP_RE.match(body[j]):
                    rows.append([c.strip() for c in body[j].strip().strip('|').split('|')])
                    j += 1
                assert len(rows) % 2 == 0, f'因子总览表行数应为偶数（S1/S2 成对），表注：{cap}'
                assert all(len(r) == 7 for r in rows), f'因子总览表应为 7 列，表注：{cap}'
                out.append(overview_table(rows, cap))
                out.append('')
                i = j
                continue
            out.append(f'Table: {cap}')
            out.append('')
        # "报告要点"色带
        if line.strip() == '## 报告要点':
            out.append('\\begin{tcolorbox}[colback=reportred,coltext=white,'
                       'boxrule=0pt,left=8pt,right=8pt,top=5pt,bottom=5pt]')
            out.append('{\\Large\\bfseries\\heiti 报告要点}')
            out.append('\\end{tcolorbox}')
            out.append('')
            i += 1
            continue
        out.append(line)
        i += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text('\n'.join(out), encoding='utf-8')
    print(f'写出 {out_path}，共 {len(out)} 行，表格 {table_no} 张')


if __name__ == '__main__':
    main()
