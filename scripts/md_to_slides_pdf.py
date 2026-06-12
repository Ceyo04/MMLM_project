"""
md_to_slides_pdf.py - 将 Marp 格式 Markdown 幻灯片转换为 PDF
"""
import re
import sys
from pathlib import Path
from weasyprint import HTML


def md_to_html(text: str) -> str:
    """简易 Markdown → HTML 转换。"""
    # 图片已在 convert() 中预处理为绝对路径，此处跳过
    # 粗体
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    # 行内代码
    text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
    # HTML div
    text = re.sub(
        r'<div style="([^"]+)">',
        r'<div style="\1">',
        text,
    )
    text = text.replace('</div>', '</div>')
    # 标题
    text = re.sub(r'^#### (.+)$', r'<h4>\1</h4>', text, flags=re.MULTILINE)
    text = re.sub(r'^### (.+)$', r'<h3>\1</h3>', text, flags=re.MULTILINE)
    text = re.sub(r'^## (.+)$', r'<h2>\1</h2>', text, flags=re.MULTILINE)
    text = re.sub(r'^# (.+)$', r'<h1>\1</h1>', text, flags=re.MULTILINE)
    # 无序列表项
    text = re.sub(r'^- (.+)$', r'<li>\1</li>', text, flags=re.MULTILINE)
    # 引用
    text = re.sub(r'^> (.+)$', r'<blockquote>\1</blockquote>', text, flags=re.MULTILINE)
    # 换行
    text = text.replace('\n\n', '<br><br>')
    return text


def convert(input_path: str, output_path: str) -> None:
    """将 Marp 格式 MD 转换为 PDF。"""
    input_file = Path(input_path).resolve()
    base_dir = input_file.parent

    with open(input_file, "r", encoding="utf-8") as f:
        content = f.read()

    # 将相对图片路径转为绝对 file:/// URL
    def resolve_img_path(match):
        alt = match.group(1)
        src = match.group(2)
        if not src.startswith(("http://", "https://", "file://", "data:")):
            abs_path = (base_dir / src).resolve()
            src = abs_path.as_uri()
        return f'<img src="{src}" alt="{alt}" style="max-width:90%;max-height:400px;display:block;margin:0 auto;">'

    content = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', resolve_img_path, content)

    # 分离 YAML frontmatter
    parts = content.split("---", 2)
    body = parts[2] if len(parts) > 2 else content

    # 将 --- 分隔的幻灯片切割
    raw_slides = body.split("\n---\n")

    # 转换每张幻灯片
    slide_htmls = []
    for slide in raw_slides:
        slide = slide.strip()
        if not slide:
            continue
        html = md_to_html(slide)
        slide_htmls.append(f'<section class="slide">{html}</section>')

    css = """
    @page { size: 16in 9in; margin: 0; }
    body { margin: 0; font-family: 'Segoe UI', 'Noto Sans CJK SC', 'WenQuanYi Zen Hei', sans-serif; }
    .slide {
        width: 16in; height: 9in; padding: 0.8in 1in; box-sizing: border-box;
        page-break-after: always; display: flex; flex-direction: column; justify-content: center;
    }
    .slide:last-child { page-break-after: avoid; }
    h1 { font-size: 42pt; color: #1a1a1a; margin: 0 0 20pt 0; }
    h2 { font-size: 32pt; color: #2c3e50; margin: 0 0 16pt 0;
         border-bottom: 3px solid #3498db; padding-bottom: 8pt; }
    h3 { font-size: 24pt; color: #333; margin: 0 0 12pt 0; }
    h4 { font-size: 20pt; color: #555; margin: 0 0 8pt 0; }
    p, li { font-size: 18pt; line-height: 1.6; color: #333; }
    strong { color: #2c3e50; }
    code { background: #f0f0f0; padding: 2pt 6pt; border-radius: 3pt; font-size: 16pt; }
    table { border-collapse: collapse; width: 100%; margin: 16pt 0; font-size: 16pt; }
    td, th { border: 1px solid #ddd; padding: 8pt 12pt; text-align: left; }
    th { background: #3498db; color: white; }
    blockquote {
        border-left: 4pt solid #3498db; padding: 8pt 16pt;
        margin: 12pt 0; background: #f8f9fa;
    }
    img { max-width: 90%; max-height: 400pt; display: block; margin: 10pt auto; }
    """

    full_html = (
        f"<html><head><meta charset=\"utf-8\"><style>{css}</style></head>"
        f"<body>{''.join(slide_htmls)}</body></html>"
    )

    HTML(string=full_html).write_pdf(output_path)
    size_kb = Path(output_path).stat().st_size / 1024
    print(f"PDF: {output_path} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python md_to_slides_pdf.py <input.md> <output.pdf>")
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2])
