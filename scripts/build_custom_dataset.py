"""
build_custom_dataset.py - 自建中文图文问答数据集

生成数据集模板 JSON 文件，用户手动填充图片和问答对。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))


def create_template(output_path: str) -> None:
    """创建数据集模板 JSON 文件。"""
    template = [
        {
            "image": "custom_images/doc_001.png",
            "category": "document",
            "qa_pairs": [
                {"question": "这份文档的标题是什么？", "answer": "请填入正确答案"},
                {"question": "文档中提到了哪几个要点？", "answer": "请填入正确答案"},
            ],
        },
        {
            "image": "custom_images/doc_002.png",
            "category": "document",
            "qa_pairs": [
                {"question": "图中表格第二列的最大值是多少？", "answer": "请填入正确答案"},
            ],
        },
        {
            "image": "custom_images/slide_001.png",
            "category": "document",
            "qa_pairs": [
                {"question": "这张幻灯片的主题是什么？", "answer": "请填入正确答案"},
                {"question": "第3点的主要内容是什么？", "answer": "请填入正确答案"},
            ],
        },
        {
            "image": "custom_images/natural_001.jpg",
            "category": "natural_scene",
            "qa_pairs": [
                {"question": "图中有几个人？", "answer": "请填入正确答案"},
                {"question": "这是什么场景？", "answer": "请填入正确答案"},
            ],
        },
        {
            "image": "custom_images/natural_002.jpg",
            "category": "natural_scene",
            "qa_pairs": [
                {"question": "图中的商品是什么品牌？", "answer": "请填入正确答案"},
            ],
        },
    ]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(template, f, ensure_ascii=False, indent=2)

    print(f"模板已生成: {output_path}")
    print("使用说明:")
    print("1. 创建 custom_images/ 目录，放入 30-50 张图片（文档截图、幻灯片、自然场景照片等）")
    print("2. 编辑 JSON 文件，为每张图片填写实际路径和 1-3 个中文问答对")
    print("3. 将 answer 字段替换为标准答案")
    print("4. 确保至少 50 条 QA 对（验收标准）")
    print()
    print("图片来源建议:")
    print("  - 课程 PPT 截图（Ctrl+PrintScreen）")
    print("  - 教材/论文页面截图")
    print("  - 日常拍摄的商店/街景照片")
    print("  - 网购商品截图")
    print("  - 维基百科/百度百科词条截图")


def main() -> None:
    """命令行入口。"""
    import argparse
    parser = argparse.ArgumentParser(description="自建中文图文问答数据集模板生成")
    parser.add_argument("--output", type=str, default="data/custom/custom_dataset.json",
                        help="输出路径")
    args = parser.parse_args()
    create_template(args.output)


if __name__ == "__main__":
    main()
