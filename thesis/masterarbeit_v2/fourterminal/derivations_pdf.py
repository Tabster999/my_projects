"""Build DERIVATIONS.pdf from DERIVATIONS.md (the Markdown is the source; rebuild after every edit).

    py derivations_pdf.py

Needs pandoc (on PATH, in the PANDOC environment variable, or via `pip install pypandoc_binary`) and lualatex.
What the PDF adds over the Markdown:
- code links become vscode://file/...:LINE links, so a click opens the file at that line in VS Code (on this
  machine; a plain relative link could make the PDF viewer run the .py file);
- every "§x.y" becomes an internal link to that section;
- a table of contents and PDF bookmarks.
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path
from urllib.parse import quote

HERE = Path(__file__).resolve().parent
SRC, OUT = HERE / 'DERIVATIONS.md', HERE / 'DERIVATIONS.pdf'

HEADER = r"""
\usepackage{newunicodechar}
\newunicodechar{→}{\ensuremath{\rightarrow}}
\usepackage{etoolbox}
\AtBeginEnvironment{longtable}{\small}
\makeatletter\def\verbatim@font{\normalfont\ttfamily\footnotesize}\makeatother
\setlength{\parskip}{4pt plus 1pt}
\setlength{\emergencystretch}{2em}
\let\origtableofcontents\tableofcontents
\renewcommand{\tableofcontents}{{\small\origtableofcontents}\clearpage}
"""

LUA_FILTER = r"""
-- "§x.y" -> internal link to the section whose heading starts with x.y; "arXiv:ID" -> link to the abstract page
function Pandoc(doc)
  local ids = {}
  doc:walk{ Header = function(h)
    local n = pandoc.utils.stringify(h):match('^(%d[%d%.]*)')
    if n then ids[(n:gsub('%.$', ''))] = h.identifier end
  end }
  local function around(pre, link, post)
    local out = {}
    if pre ~= '' then table.insert(out, pandoc.Str(pre)) end
    table.insert(out, link)
    if post ~= '' then table.insert(out, pandoc.Str(post)) end
    return out
  end
  return doc:walk{ Str = function(s)
    local pre, num, post = s.text:match('^(.-)§(%d[%d%.]*)(.*)$')
    if num then
      if num:sub(-1) == '.' then num = num:sub(1, -2); post = '.' .. post end
      if ids[num] then return around(pre, pandoc.Link(pandoc.Str('§' .. num), '#' .. ids[num]), post) end
      return nil
    end
    local pre2, id, post2 = s.text:match('^(.-)arXiv:([%w%./%-]-)([%.,;:%)]*)$')
    if id and id ~= '' then
      return around(pre2, pandoc.Link(pandoc.Str('arXiv:' .. id), 'https://arxiv.org/abs/' .. id), post2)
    end
  end }
end

-- code links never break across lines (no "study_oscilla-tion.py")
function Link(l)
  if l.target:match('^vscode://') then
    return {pandoc.RawInline('latex', '\\mbox{'), l, pandoc.RawInline('latex', '}')}
  end
end

-- gfm pipe tables carry no widths, so LaTeX would not wrap the cells.  Each column gets a share of the text width
-- proportional to sqrt(longest cell), but never less than its longest word (TeX source of math counts half).
function Table(tbl)
  local n = #tbl.colspecs
  local len, tok = {}, {}
  for i = 1, n do len[i], tok[i] = 1, 1 end
  local function text(blocks)
    return pandoc.utils.stringify(blocks:walk{ Math = function(m)
      return pandoc.Str(string.rep('x', math.ceil(#m.text / 2))) end })
  end
  local function scan(rows)
    for _, row in ipairs(rows) do
      for i, cell in ipairs(row.cells) do
        local t = text(cell.contents)
        len[i] = math.max(len[i], utf8.len(t) or #t)
        for word in t:gmatch('%S+') do tok[i] = math.max(tok[i], utf8.len(word) or #word) end
      end
    end
  end
  scan(tbl.head.rows)
  for _, body in ipairs(tbl.bodies) do scan(body.body) end
  local s, tot = {}, 0
  for i = 1, n do s[i] = math.sqrt(len[i]); tot = tot + s[i] end
  local w, fixed, free = {}, 0, 0
  for i = 1, n do
    local floor = (tok[i] + 2) * 0.0105             -- one \small character ~ 1/95 of the text width
    if 0.99 * s[i] / tot < floor then w[i] = floor; fixed = fixed + floor else free = free + s[i] end
  end
  for i = 1, n do
    w[i] = w[i] or (0.99 - fixed) * s[i] / free
    tbl.colspecs[i] = {tbl.colspecs[i][1], w[i]}
  end
  return tbl
end
"""


def find_pandoc():
    exe = os.environ.get('PANDOC') or shutil.which('pandoc')
    if exe:
        return exe
    try:
        import pypandoc
        return pypandoc.get_pandoc_path()
    except (ImportError, OSError):
        sys.exit('pandoc not found: put it on PATH, set PANDOC=..., or `pip install pypandoc_binary`')


def preprocess(md):
    intro = re.compile(r'Links point to the exact lines; in VS Code they open the file at that line, and the preview '
                       r'renders the\s+equations\.')
    md, n = intro.subn('In this PDF a code link opens the file at that line in VS Code. A section sign (§) links to '
                       'that section. The PDF is built from DERIVATIONS.md by derivations_pdf.py; the Markdown is the '
                       'source.', md)
    if not n:
        print('note: intro sentence about links not found, left as is')
    md = re.sub(r'^---\s*$', '', md, flags=re.M)                          # section rules: headings suffice

    def to_vscode(m):
        line = f':{m.group(2)}' if m.group(2) else ''
        return '](vscode://file/' + quote((HERE / m.group(1)).as_posix(), safe='/:') + line + ')'
    md, n = re.subn(r'\]\(((?:computations|my_functions)/[^)#\s]+)(?:#L(\d+)(?:-L\d+)?)?\)', to_vscode, md)
    print(f'{n} code links -> vscode://')
    return md


def main():
    md = preprocess(SRC.read_text(encoding='utf-8'))
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / 'in.md').write_text(md, encoding='utf-8')
        (tmp / 'header.tex').write_text(HEADER, encoding='utf-8')
        (tmp / 'sections.lua').write_text(LUA_FILTER, encoding='utf-8')
        cmd = [find_pandoc(), str(tmp / 'in.md'), '-o', str(OUT), '--from=gfm+smart',   # GitHub/VS Code dialect
               '--standalone', '--pdf-engine=lualatex', '--shift-heading-level-by=-1', '--toc', '--toc-depth=2',
               '--lua-filter', str(tmp / 'sections.lua'), '-H', str(tmp / 'header.tex'),
               '-M', 'subtitle=fourterminal: derivations, sources and code locations',
               '-M', f'date=Built from DERIVATIONS.md on {date.today():%d %B %Y}',
               '-V', 'documentclass=article', '-V', 'fontsize=11pt', '-V', 'lang=en-GB',
               '-V', 'geometry:a4paper,margin=2.2cm', '-V', 'colorlinks=true',
               '-V', 'linkcolor=blue!50!black', '-V', 'urlcolor=blue!50!black', '-V', 'toccolor=black']
        if len(sys.argv) > 1:                          # debugging: `py derivations_pdf.py out.tex` writes the LaTeX
            cmd[3] = sys.argv[1]
        subprocess.run(cmd, check=True)
    print(f'wrote {cmd[3]}')


if __name__ == '__main__':
    main()
