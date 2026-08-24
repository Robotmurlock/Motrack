"""
Renders a Markdown report to PDF.

The report is written for reading on GitHub, so it uses Markdown tables, relative image paths,
inline HTML anchors for the reference list, and internal links from each citation to its entry.
All four have to survive the conversion, which rules out a plain text-to-PDF path. Markdown is
converted to HTML, styled for print, and given to a print engine, which keeps the internal links
live as PDF annotations.

Two engines are supported. WeasyPrint is preferred because it implements the `@page` margin
boxes, so the output gets page numbers and a running section header; it is used through `uvx` if
it is not installed. Headless Chrome is the fallback and produces the same layout without those
two, since it ignores margin boxes.

Usage:
    python -m tools.render_report docs/cmc/report.md
    python -m tools.render_report docs/cmc/report.md --engine chrome --keep-html
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from typing import List, Optional

import markdown

EXTENSIONS = ['tables', 'fenced_code', 'attr_list', 'md_in_html', 'sane_lists']

# `--headless=new` is the supported mode from Chrome 112 onwards; the older `--headless` prints a
# deprecation notice and resolves fonts differently.
CHROME_CANDIDATES = ['google-chrome', 'chromium', 'chromium-browser', 'google-chrome-stable']

STYLE = """
@page {
    size: A4;
    margin: 19mm 17mm 18mm 17mm;

    /* Margin boxes are a WeasyPrint feature; Chrome ignores them and prints without numbers. */
    @bottom-center {
        content: counter(page) " / " counter(pages);
        font-family: "Inter", "DejaVu Sans", sans-serif;
        font-size: 8pt;
        color: #888;
    }
    @top-right {
        content: string(section);
        font-family: "Inter", "DejaVu Sans", sans-serif;
        font-size: 8pt;
        color: #aaa;
    }
}

/* The title page carries neither, since it names the document itself. */
@page :first {
    @bottom-center { content: ""; }
    @top-right { content: ""; }
}

:root {
    --ink: #1a1a1a;
    --muted: #555;
    --rule: #d0d0d0;
    --accent: #1451a4;
    --code-bg: #f4f4f6;
}

body {
    font-family: "Charter", "Georgia", "DejaVu Serif", "Liberation Serif", serif;
    font-size: 10pt;
    line-height: 1.5;
    color: var(--ink);
    margin: 0;
}

h1, h2, h3, h4, h5 {
    font-family: "Inter", "Helvetica Neue", "DejaVu Sans", "Liberation Sans", sans-serif;
    line-height: 1.25;
    break-after: avoid;
    page-break-after: avoid;
    margin: 1.4em 0 0.5em;
}

/* The document title carries the first page on its own terms. */
h1 {
    font-size: 20pt;
    margin: 0 0 0.2em;
    letter-spacing: -0.01em;
}

h2 {
    font-size: 14pt;
    border-bottom: 1px solid var(--rule);
    padding-bottom: 0.25em;
    break-before: page;
    page-break-before: always;
    /* Feeds the running header, so each page names the section it belongs to. */
    string-set: section content();
}

/* Section 1 follows the title block, so it must not push a blank first page. */
h2.first-section {
    break-before: auto;
    page-break-before: auto;
}

h3 { font-size: 11.5pt; }
h4 { font-size: 10.5pt; color: var(--muted); }

p { margin: 0 0 0.7em; orphans: 3; widows: 3; }

a { color: var(--accent); text-decoration: none; }

/* Citations read as numbers in running text, so they get no link colour. */
a[href^="#ref-"] { color: var(--ink); }

code, kbd {
    font-family: "JetBrains Mono", "DejaVu Sans Mono", "Liberation Mono", monospace;
    font-size: 0.86em;
    background: var(--code-bg);
    padding: 0.05em 0.28em;
    border-radius: 2px;
}

pre {
    background: var(--code-bg);
    border: 1px solid var(--rule);
    border-radius: 3px;
    padding: 0.6em 0.8em;
    font-size: 8pt;
    line-height: 1.4;
    overflow-x: auto;
    break-inside: avoid;
    page-break-inside: avoid;
}

pre code { background: none; padding: 0; font-size: 1em; }

table {
    border-collapse: collapse;
    width: 100%;
    font-size: 8.5pt;
    margin: 0.6em 0 1em;
    font-variant-numeric: tabular-nums;
}

/* Long tables may break across pages, but never inside a row, and the header repeats. */
thead { display: table-header-group; }
tr { break-inside: avoid; page-break-inside: avoid; }

th, td {
    border-bottom: 1px solid var(--rule);
    padding: 0.32em 0.5em;
    text-align: left;
}

thead th {
    border-bottom: 1.2px solid #888;
    font-family: "Inter", "DejaVu Sans", sans-serif;
    font-size: 8pt;
    font-weight: 600;
}

table code { background: none; padding: 0; }

blockquote {
    margin: 0.8em 0;
    padding: 0.5em 0.9em;
    border-left: 3px solid #b8b8b8;
    background: #fafafa;
    color: #333;
    break-inside: avoid;
    page-break-inside: avoid;
}

blockquote p:last-child { margin-bottom: 0; }

figure {
    margin: 1em 0;
    text-align: center;
    break-inside: avoid;
    page-break-inside: avoid;
}

figure img { max-width: 100%; height: auto; }

figcaption {
    font-size: 8.5pt;
    color: var(--muted);
    margin-top: 0.4em;
    text-align: center;
}

ul, ol { margin: 0 0 0.8em; padding-left: 1.4em; }
li { margin-bottom: 0.25em; }

hr { border: none; border-top: 1px solid var(--rule); margin: 1.6em 0; }

/* Reference list: numbers stay tight against the entries and entries never split. */
.references ol { padding-left: 1.9em; }
.references li { margin-bottom: 0.4em; font-size: 9pt; break-inside: avoid; }

.title-block {
    border-bottom: 2px solid var(--ink);
    padding-bottom: 0.8em;
    margin-bottom: 1.6em;
}

.title-block .subtitle {
    font-family: "Inter", "DejaVu Sans", sans-serif;
    font-size: 10pt;
    color: var(--muted);
    font-style: normal;
}
"""


def weasyprint_command() -> Optional[List[str]]:
    """
    How to invoke WeasyPrint, installed or through `uvx`, or None if neither works.
    """
    if shutil.which('weasyprint'):
        return ['weasyprint']
    if shutil.which('uvx'):
        probe = subprocess.run(['uvx', '--from', 'weasyprint', 'weasyprint', '--version'],
                               capture_output=True, text=True, check=False)
        if probe.returncode == 0:
            return ['uvx', '--from', 'weasyprint', 'weasyprint']
    return None


def chrome_command() -> Optional[List[str]]:
    """
    First Chrome-family binary on PATH, or None.
    """
    for name in CHROME_CANDIDATES:
        path = shutil.which(name)
        if path:
            return [path]
    return None


def promote_figures(html: str) -> str:
    """
    Turns a standalone image into a figure with its alt text as the caption.

    Markdown has no caption syntax, so the report writes the caption as alt text. In HTML that
    text is invisible, and in a printed report the figures need captions, so the paragraphs that
    hold nothing but an image are rewritten.
    """
    pattern = re.compile(r'<p>(<img [^>]*?alt="([^"]*)"[^>]*?/?>)</p>')
    return pattern.sub(lambda m: f'<figure>{m.group(1)}<figcaption>{m.group(2)}</figcaption></figure>',
                       html)


def build_html(markdown_path: str) -> str:
    """
    The report as a self-contained print stylesheet plus its converted body.
    """
    source = open(markdown_path, encoding='utf-8').read()

    # The first heading becomes the title block, and the italic line under it the subtitle, so
    # neither is repeated in the body.
    lines = source.split('\n')
    title = lines[0].lstrip('# ').strip()
    subtitle = ''
    body_start = 1
    for index in range(1, min(6, len(lines))):
        stripped = lines[index].strip()
        if stripped.startswith('*') and stripped.endswith('*') and len(stripped) > 2:
            subtitle = stripped.strip('*')
            body_start = index + 1
            break
    source = '\n'.join(lines[body_start:])

    html = markdown.Markdown(extensions=EXTENSIONS).convert(source)
    html = promote_figures(html)

    # The reference list is the only place where the numbering has to line up with the anchors,
    # so it is marked up for its own spacing rules.
    html = html.replace('<h2>References</h2>', '<h2>References</h2><div class="references">')
    if '<div class="references">' in html:
        html += '</div>'

    # Every other section starts on a fresh page; the first would otherwise leave page one empty.
    # Matched by position rather than by text, since the headings carry cross-reference anchors.
    html = re.sub(r'<h2>', '<h2 class="first-section">', html, count=1)

    base = 'file://' + os.path.dirname(os.path.abspath(markdown_path)) + '/'
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<base href="{base}">
<title>{title}</title>
<style>{STYLE}</style>
</head>
<body>
<div class="title-block">
<h1>{title}</h1>
<div class="subtitle">{subtitle}</div>
</div>
{html}
</body>
</html>
"""


def print_with_weasyprint(command: List[str], html_path: str, pdf_path: str, base: str) -> None:
    """
    Prints with WeasyPrint. The base URL is what resolves the figures' relative paths.
    """
    result = subprocess.run(command + [html_path, pdf_path, '--base-url', base],
                            capture_output=True, text=True, check=False)
    if not os.path.exists(pdf_path):
        sys.stderr.write(result.stderr)
        raise SystemExit('WeasyPrint did not produce a PDF')


def print_with_chrome(command: List[str], html_path: str, pdf_path: str) -> None:
    """
    Prints with headless Chrome.

    The virtual time budget gives layout time to settle before the snapshot; without it Chrome
    can print before the images have been decoded. The header and footer are suppressed because
    Chrome fills them with the source file's URL rather than anything the report should carry.
    """
    command = command + [
        '--headless=new', '--disable-gpu', '--no-sandbox',
        '--no-pdf-header-footer', '--virtual-time-budget=20000',
        f'--print-to-pdf={pdf_path}', 'file://' + html_path,
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if not os.path.exists(pdf_path):
        sys.stderr.write(result.stderr)
        raise SystemExit('Chrome did not produce a PDF')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', nargs='?', default='docs/cmc/report.md')
    parser.add_argument('--output', default=None, help='Defaults to the source with a .pdf suffix')
    parser.add_argument('--keep-html', action='store_true', help='Leave the intermediate HTML in place')
    parser.add_argument('--engine', choices=['auto', 'weasyprint', 'chrome'], default='auto')
    args = parser.parse_args()

    source = os.path.abspath(args.source)
    output = os.path.abspath(args.output or os.path.splitext(source)[0] + '.pdf')
    base = 'file://' + os.path.dirname(source) + '/'

    html = build_html(source)
    handle, html_path = tempfile.mkstemp(suffix='.html', prefix='report-')
    with os.fdopen(handle, 'w', encoding='utf-8') as file:
        file.write(html)

    weasyprint = weasyprint_command() if args.engine in ('auto', 'weasyprint') else None
    chrome = chrome_command() if args.engine in ('auto', 'chrome') else None

    try:
        if weasyprint:
            print_with_weasyprint(weasyprint, html_path, output, base)
            engine = 'weasyprint'
        elif chrome:
            print_with_chrome(chrome, html_path, output)
            engine = 'chrome (no page numbers)'
        else:
            raise SystemExit('no print engine available: install weasyprint, or a Chrome binary')
    finally:
        if args.keep_html:
            print(f'html: {html_path}')
        else:
            os.unlink(html_path)

    print(f'wrote {output}  ({os.path.getsize(output) / 1e6:.1f} MB, {engine})')


if __name__ == '__main__':
    main()
