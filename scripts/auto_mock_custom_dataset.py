"""
auto_mock_custom_dataset.py - 自动生成自建中文图文问答数据集

使用 PIL 生成带有文字和形状的测试图片，并生成对应的问答对。
覆盖 document（文档/幻灯片）和 natural（自然场景）两类。
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from PIL import Image, ImageDraw, ImageFont
import numpy as np

# 输出路径
OUTPUT_DIR = _project_root / "data" / "custom"
IMAGE_DIR = OUTPUT_DIR / "custom_images"
DATASET_FILE = OUTPUT_DIR / "custom_dataset.json"

# 颜色常量
COLORS = {
    "red": (255, 0, 0),
    "blue": (0, 0, 255),
    "green": (0, 255, 0),
    "yellow": (255, 255, 0),
    "orange": (255, 165, 0),
    "purple": (128, 0, 128),
    "pink": (255, 192, 203),
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "gray": (128, 128, 128),
    "brown": (139, 69, 19),
    "cyan": (0, 255, 255),
}

COLOR_NAMES_CN = {
    "red": "红色", "blue": "蓝色", "green": "绿色", "yellow": "黄色",
    "orange": "橙色", "purple": "紫色", "pink": "粉色", "white": "白色",
    "black": "黑色", "gray": "灰色", "brown": "棕色", "cyan": "青色",
}

SHAPES = ["圆形", "正方形", "三角形", "长方形", "椭圆"]


def _get_font(size: int = 24) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """尝试获取中文字体，失败则用默认字体。"""
    font_paths = [
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        "/usr/share/fonts/truetype/arphic/uming.ttc",
        "/usr/share/fonts/truetype/arphic/ukai.ttc",
        "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    ]
    for fp in font_paths:
        if Path(fp).exists():
            return ImageFont.truetype(fp, size)
    # Fallback: default font (won't show CJK but won't crash)
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _draw_text_centered(draw: ImageDraw.Draw, text: str, y: int, img_width: int,
                        fill: tuple = (0, 0, 0), font: ImageFont.FreeTypeFont | None = None) -> None:
    """在指定 y 坐标居中绘制文字。"""
    if font is None:
        font = _get_font(28)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    x = max(0, (img_width - tw) // 2)
    draw.text((x, y), text, fill=fill, font=font)


def generate_document_images() -> list[dict]:
    """生成文档/幻灯片类图片和问答对（30 条）。"""
    entries = []
    font = _get_font(28)
    font_small = _get_font(20)
    font_title = _get_font(36)

    templates = [
        {
            "title": "2024年度工作总结",
            "points": ["全年营收增长15%", "新增客户320家", "研发投入占比8.5%", "员工满意度92分"],
            "questions": [
                ("这份文档的标题是什么？", "2024年度工作总结"),
                ("全年营收增长了多少？", "15%"),
                ("新增了多少家客户？", "320家"),
                ("研发投入占比是多少？", "8.5%"),
            ],
        },
        {
            "title": "项目进度报告 — Q3",
            "points": ["前端开发完成90%", "后端API完成75%", "测试用例覆盖60%", "预计11月上线"],
            "questions": [
                ("这份报告的标题是什么？", "项目进度报告 — Q3"),
                ("前端开发完成了多少？", "90%"),
                ("后端API完成了多少？", "75%"),
                ("预计什么时候上线？", "11月"),
            ],
        },
        {
            "title": "市场分析报告",
            "points": ["市场规模：500亿元", "年增长率：12.3%", "主要竞争者：3家头部企业", "目标份额：8%"],
            "questions": [
                ("市场规模是多少？", "500亿元"),
                ("年增长率是多少？", "12.3%"),
                ("有几家主要竞争者？", "3家"),
                ("目标市场份额是多少？", "8%"),
            ],
        },
        {
            "title": "产品规格说明",
            "points": ["型号：X500 Pro", "处理器：八核2.8GHz", "内存：16GB DDR5", "存储：512GB SSD", "价格：¥6,999"],
            "questions": [
                ("产品型号是什么？", "X500 Pro"),
                ("处理器规格是什么？", "八核2.8GHz"),
                ("内存容量是多少？", "16GB"),
                ("产品价格是多少？", "¥6,999"),
            ],
        },
        {
            "title": "课程：多模态大模型原理与应用",
            "points": ["第12讲：视觉语言模型", "主讲教师：张教授", "考核方式：期末项目+报告", "学分：3学分"],
            "questions": [
                ("课程名称是什么？", "多模态大模型原理与应用"),
                ("这一讲的标题是什么？", "视觉语言模型"),
                ("主讲教师是谁？", "张教授"),
                ("考核方式是什么？", "期末项目+报告"),
            ],
        },
    ]

    for idx, tmpl in enumerate(templates):
        img = Image.new("RGB", (800, 500), color=(245, 245, 250))
        draw = ImageDraw.Draw(img)

        # 标题
        y = 30
        _draw_text_centered(draw, tmpl["title"], y, 800, fill=(0, 51, 102), font=font_title)
        y += 60

        # 分隔线
        draw.line([(50, y), (750, y)], fill=(0, 51, 102), width=2)
        y += 30

        # 要点列表
        for i, point in enumerate(tmpl["points"]):
            text = f"  {i+1}. {point}"
            _draw_text_centered(draw, text, y, 800, fill=(30, 30, 30), font=font)
            y += 45

        # 底部信息
        y = 450
        _draw_text_centered(draw, f"文档编号：DOC-{idx+1:03d}", y, 800, fill=(150, 150, 150), font=font_small)

        filename = f"doc_slide_{idx+1:03d}.jpg"
        img.save(str(IMAGE_DIR / filename), quality=90)

        entries.append({
            "image": f"custom_images/{filename}",
            "category": "document",
            "qa_pairs": [{"question": q, "answer": a} for q, a in tmpl["questions"]],
        })

    return entries


def generate_natural_images() -> list[dict]:
    """生成自然场景类图片和问答对（20 条）。"""
    entries = []
    font = _get_font(26)
    font_small = _get_font(18)

    # 颜色块测试
    for color_name, rgb in list(COLORS.items())[:8]:
        img = Image.new("RGB", (400, 300), color=rgb)
        draw = ImageDraw.Draw(img)
        cn_name = COLOR_NAMES_CN.get(color_name, color_name)
        _draw_text_centered(draw, f"这是一个{cn_name}的物体", 130, 400, fill=(255, 255, 255) if sum(rgb) < 400 else (0, 0, 0), font=font)

        filename = f"natural_color_{color_name}.jpg"
        img.save(str(IMAGE_DIR / filename), quality=90)

        entries.append({
            "image": f"custom_images/{filename}",
            "category": "natural_scene",
            "qa_pairs": [
                {"question": f"这个物体是什么颜色？", "answer": cn_name},
                {"question": f"What color is this object?", "answer": color_name},
            ],
        })

    # 形状数量测试
    shape_configs = [
        ("圆形", "circle", "red", 3),
        ("三角形", "triangle", "blue", 5),
        ("正方形", "square", "green", 2),
        ("长方形", "rectangle", "orange", 4),
    ]

    for idx, (shape_cn, shape_en, color, count) in enumerate(shape_configs):
        img = Image.new("RGB", (500, 400), color=(240, 240, 240))
        draw = ImageDraw.Draw(img)

        rng = random.Random(idx + 42)
        color_rgb = COLORS[color]
        positions = []
        for _ in range(count):
            x = rng.randint(50, 400)
            y = rng.randint(50, 300)
            positions.append((x, y))
            if shape_en == "circle":
                draw.ellipse([x, y, x + 60, y + 60], fill=color_rgb, outline=(0, 0, 0))
            elif shape_en == "triangle":
                draw.polygon([(x + 30, y), (x, y + 60), (x + 60, y + 60)], fill=color_rgb, outline=(0, 0, 0))
            elif shape_en == "square":
                draw.rectangle([x, y, x + 55, y + 55], fill=color_rgb, outline=(0, 0, 0))
            elif shape_en == "rectangle":
                draw.rectangle([x, y, x + 80, y + 45], fill=color_rgb, outline=(0, 0, 0))

        _draw_text_centered(draw, f"图中有 {count} 个{COLOR_NAMES_CN[color]}{shape_cn}", 350, 500, fill=(0, 0, 0), font=font_small)

        filename = f"natural_shape_{shape_en}_{idx+1}.jpg"
        img.save(str(IMAGE_DIR / filename), quality=90)

        entries.append({
            "image": f"custom_images/{filename}",
            "category": "natural_scene",
            "qa_pairs": [
                {"question": f"图中有几个{shape_cn}？", "answer": str(count)},
                {"question": f"{shape_cn}是什么颜色的？", "answer": COLOR_NAMES_CN[color]},
                {"question": f"How many {shape_en}s are there?", "answer": str(count)},
            ],
        })

    # 混合场景测试
    for i in range(7):
        img = Image.new("RGB", (600, 350), color=(220, 240, 255))
        draw = ImageDraw.Draw(img)

        # 随机放几个不同颜色的圆
        rng = random.Random(i + 100)
        circle_count = rng.randint(1, 6)
        colors_used = rng.sample(list(COLORS.keys()), min(circle_count, len(COLORS)))
        for c_idx, c_name in enumerate(colors_used):
            x = 30 + c_idx * 90
            y = 100 + rng.randint(-30, 50)
            draw.ellipse([x, y, x + 70, y + 70], fill=COLORS[c_name], outline=(0, 0, 0))

        # 放一个矩形
        rect_color = rng.choice(list(COLORS.keys()))
        while rect_color in colors_used:
            rect_color = rng.choice(list(COLORS.keys()))
        draw.rectangle([400, 180, 520, 240], fill=COLORS[rect_color], outline=(0, 0, 0))

        dominant_color = rng.choice(colors_used)
        dominant_cn = COLOR_NAMES_CN.get(dominant_color, dominant_color)

        _draw_text_centered(draw, f"场景 {i+1}：多物体组合", 300, 600, fill=(0, 0, 0), font=font_small)

        filename = f"natural_mixed_{i+1:03d}.jpg"
        img.save(str(IMAGE_DIR / filename), quality=90)

        entries.append({
            "image": f"custom_images/{filename}",
            "category": "natural_scene",
            "qa_pairs": [
                {"question": f"图中共有几个圆形物体？", "answer": str(circle_count)},
                {"question": f"出现最多的颜色是什么？", "answer": dominant_cn},
                {"question": f"图中是否有{COLOR_NAMES_CN.get(rect_color)}的物体？", "answer": "是"},
            ],
        })

    return entries


def main() -> None:
    """生成完整数据集。"""
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    print("生成文档/幻灯片类图片和问答对...")
    doc_entries = generate_document_images()
    print(f"  文档类: {len(doc_entries)} 张图片, {sum(len(e['qa_pairs']) for e in doc_entries)} 条问答对")

    print("生成自然场景类图片和问答对...")
    natural_entries = generate_natural_images()
    print(f"  自然类: {len(natural_entries)} 张图片, {sum(len(e['qa_pairs']) for e in natural_entries)} 条问答对")

    all_entries = doc_entries + natural_entries
    total_qa = sum(len(e["qa_pairs"]) for e in all_entries)

    with open(DATASET_FILE, "w", encoding="utf-8") as f:
        json.dump(all_entries, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 数据集生成完成：")
    print(f"   图片：{len(all_entries)} 张 -> {IMAGE_DIR}")
    print(f"   问答对：{total_qa} 条 -> {DATASET_FILE}")

    # 验证
    with open(DATASET_FILE, "r") as f:
        loaded = json.load(f)
    assert len(loaded) == len(all_entries)
    total = sum(len(e["qa_pairs"]) for e in loaded)
    assert total == total_qa
    print(f"   验证通过：{len(loaded)} 张图片, {total} 条问答对")


if __name__ == "__main__":
    main()
