"""
error_analysis.py - 错误分析与案例研究

读取评测结果中的预测和错误数据，进行错误分类、可视化和案例提取。
"""

from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# 中文字体设置
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "SimHei", "Arial Unicode MS", "Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False


# --- 错误分类 ---

def classify_error(question: str, predicted: str, ground_truths: list[str]) -> str:
    """
    根据问题和回答对错误进行分类。

    返回错误类型标签：
    - visual: 视觉理解错误（物体识别、颜色、形状等）
    - ocr: OCR/文字识别错误
    - reasoning: 推理/计数错误
    - knowledge: 外部知识缺失
    - expression: 语言表达问题
    """
    q_lower = question.lower()

    # 检查是否涉及文字/OCR
    ocr_keywords = ["text", "written", "read", "say", "letter", "word", "sign", "logo", "brand"]
    if any(w in q_lower for w in ocr_keywords):
        # 检测预测结果是否正确识别了文字
        for gt in ground_truths:
            if gt.lower() in predicted.lower() or predicted.lower() in gt.lower():
                return "ocr_match_ok"
        return "ocr"

    # 检查是否涉及计数
    count_keywords = ["how many", "count", "number of"]
    if any(w in q_lower for w in count_keywords):
        return "reasoning"

    # 检查是否涉及颜色
    color_keywords = ["what color", "color of", "what colour"]
    if any(w in q_lower for w in color_keywords):
        return "visual"

    # 检查是否需要外部知识
    knowledge_keywords = ["who", "what brand", "what kind", "what type", "what animal", "what bird",
                          "name of", "species", "breed"]
    if any(w in q_lower for w in knowledge_keywords):
        return "knowledge"

    # 默认分类
    if predicted.strip() == "" or "sorry" in predicted.lower() or "cannot" in predicted.lower():
        return "expression"

    # 短答案 vs 长答案
    if len(predicted.strip()) < 5:
        return "visual"  # 短答案更可能是视觉理解问题
    else:
        return "reasoning"


def analyze_errors(predictions_file: str, output_dir: str) -> dict[str, Any]:
    """
    分析预测文件中的错误。

    Args:
        predictions_file: 评测生成的 predictions.json 路径
        output_dir: 输出目录

    Returns:
        分析结果字典
    """
    with open(predictions_file, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    errors = [p for p in predictions if not p.get("correct", False)]
    successes = [p for p in predictions if p.get("correct", False)]

    logger.info(f"Total predictions: {len(predictions)}")
    logger.info(f"Correct: {len(successes)}, Errors: {len(errors)}")

    # 错误分类
    error_by_type: dict[str, list[dict]] = defaultdict(list)
    for err in errors:
        error_type = classify_error(
            err.get("question", ""),
            err.get("predicted", ""),
            err.get("ground_truths", []),
        )
        err["error_type"] = error_type
        error_by_type[error_type].append(err)

    # 统计
    error_stats = {
        "total_errors": len(errors),
        "total_predictions": len(predictions),
        "error_rate": round(len(errors) / len(predictions), 4) if predictions else 0,
        "by_type": {
            etype: {
                "count": len(elist),
                "percentage": round(len(elist) / len(errors) * 100, 1) if errors else 0,
            }
            for etype, elist in error_by_type.items()
        },
    }

    # 生成错误分布图
    _plot_error_distribution(error_by_type, output_dir)

    # 提取典型成功/失败案例
    case_studies = _extract_case_studies(successes, errors, n=7)

    # 保存结果
    analysis_result = {
        "error_stats": error_stats,
        "case_studies": case_studies,
    }

    analysis_path = Path(output_dir) / "error_analysis.json"
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(analysis_result, f, ensure_ascii=False, indent=2)

    # 错误详情（带分类）
    errors_with_types_path = Path(output_dir) / "errors_classified.json"
    # 展平 error_by_type 为列表
    all_errors = []
    for etype, elist in error_by_type.items():
        all_errors.extend(elist)
    with open(errors_with_types_path, "w", encoding="utf-8") as f:
        json.dump(all_errors, f, ensure_ascii=False, indent=2)

    logger.info(f"Error analysis saved to {analysis_path}")
    return analysis_result


def _plot_error_distribution(error_by_type: dict[str, list[dict]], output_dir: str) -> None:
    """生成错误类型分布图。"""
    # 重命名标签
    label_map = {
        "visual": "视觉理解",
        "ocr": "OCR/文字",
        "ocr_match_ok": "OCR(正确)",
        "reasoning": "推理/计数",
        "knowledge": "知识缺失",
        "expression": "表达问题",
    }

    types = list(error_by_type.keys())
    counts = [len(error_by_type[t]) for t in types]
    labels = [label_map.get(t, t) for t in types]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 饼图
    colors = plt.cm.Set2(np.linspace(0, 1, len(types)))
    wedges, texts, autotexts = ax1.pie(
        counts, labels=labels, autopct="%1.1f%%",
        colors=colors, startangle=90,
    )
    ax1.set_title("错误类型分布", fontsize=14)

    # 柱状图
    bars = ax2.bar(range(len(types)), counts, color=colors, edgecolor="white")
    ax2.set_xticks(range(len(types)))
    ax2.set_xticklabels(labels, rotation=45, ha="right")
    ax2.set_ylabel("错误数量")
    ax2.set_title("错误类型计数", fontsize=14)
    for bar, count in zip(bars, counts):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 str(count), ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    fig_path = Path(output_dir) / "error_distribution.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Error distribution plot saved to {fig_path}")


def _extract_case_studies(
    successes: list[dict],
    errors: list[dict],
    n: int = 7,
) -> dict[str, list[dict]]:
    """提取典型成功和失败案例。"""
    # 成功案例：按推理时间排序，取最快的前n个（说明模型很自信）
    success_cases = sorted(successes, key=lambda x: x.get("infer_time", 999))[:n]

    # 失败案例：按错误类型各取一些
    error_by_type: dict[str, list[dict]] = defaultdict(list)
    for err in errors:
        error_by_type[err.get("error_type", "unknown")].append(err)

    failure_cases = []
    for etype, elist in error_by_type.items():
        failure_cases.extend(elist[:max(1, n // len(error_by_type))])
    failure_cases = failure_cases[:n]

    # 精简字段
    def simplify(item: dict) -> dict:
        return {
            "question": item.get("question", ""),
            "predicted": item.get("predicted", ""),
            "ground_truths": item.get("ground_truths", [])[:5],
            "correct": item.get("correct", False),
            "error_type": item.get("error_type", ""),
        }

    return {
        "successes": [simplify(s) for s in success_cases],
        "failures": [simplify(f) for f in failure_cases],
    }


def main() -> None:
    """命令行入口。"""
    import argparse
    parser = argparse.ArgumentParser(description="错误分析与案例研究")
    parser.add_argument("--predictions", type=str, required=True, help="predictions.json 路径")
    parser.add_argument("--output", type=str, required=True, help="输出目录")
    args = parser.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)
    analyze_errors(args.predictions, args.output)


if __name__ == "__main__":
    main()
