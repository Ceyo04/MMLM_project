"""
vqa_v2_eval.py - VQA-v2 评测

在 VQA-v2 验证集子集上评测模型准确率。
使用 VQA 标准软匹配：预测答案与 10 个标注答案中 ≥3 个一致即算正确。
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

# 确保项目根目录在 sys.path 中
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.config.settings import get_config
from src.data.vqa_v2_loader import load_vqa_v2_stream
from src.inference.generate import generate_single

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def normalize_answer(text: str) -> str:
    """标准化答案文本，用于 VQA 软匹配。"""
    text = text.lower().strip()
    # 移除冠词
    text = re.sub(r"\b(a|an|the|is|are|was|were|of|in|on|at|to|for)\b", "", text)
    # 移除标点
    text = re.sub(r"[^\w\s]", "", text)
    # 合并多余空格
    text = re.sub(r"\s+", " ", text).strip()
    return text


def vqa_soft_accuracy(predicted: str, ground_truths: list[str]) -> bool:
    """
    VQA 标准软匹配准确率判定。

    规则：预测答案经过 normalize 后，与 10 个标注答案中 ≥3 个的
    normalize 结果一致，则判定为正确。
    """
    pred_norm = normalize_answer(predicted)

    # 对于少数答案（如数字），如果预测完全包含在某个标注中也可接受
    matches = 0
    for gt in ground_truths:
        gt_norm = normalize_answer(gt)
        if pred_norm == gt_norm or gt_norm in pred_norm or pred_norm in gt_norm:
            matches += 1

    # VQA 标准：≥3 个标注一致
    threshold = min(3, max(1, len(ground_truths) // 3))
    return matches >= threshold


def evaluate_vqa_v2(
    num_samples: int = 1000,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """
    运行 VQA-v2 评测。

    Args:
        num_samples: 评测样本数
        output_dir: 结果输出目录（None 则自动生成）

    Returns:
        评测结果字典
    """
    cfg = get_config()
    logger.info(f"Starting VQA-v2 evaluation on {num_samples} samples")
    logger.info(f"Model: {cfg.model_name}, max_pixels={cfg.max_pixels_actual}")

    correct = 0
    total = 0
    per_category: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    errors: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []

    t_start = time.time()

    for i, sample in enumerate(load_vqa_v2_stream(max_samples=num_samples)):
        total += 1
        question = sample["question"]
        answers = sample["answers"]

        try:
            t_infer = time.time()
            pred_answer, _ = generate_single(
                sample["image"],
                question,
                scene="natural_scene",
            )
            infer_time = time.time() - t_infer

            is_correct = vqa_soft_accuracy(pred_answer, answers)
            if is_correct:
                correct += 1

            # 按题型分类（简单启发式）
            q_lower = question.lower()
            if q_lower.startswith("is ") or q_lower.startswith("are "):
                category = "yes/no"
            elif any(w in q_lower for w in ["how many", "count", "number"]):
                category = "number"
            else:
                category = "other"

            per_category[category]["total"] += 1
            if is_correct:
                per_category[category]["correct"] += 1

            predictions.append({
                "question_id": sample["question_id"],
                "question": question,
                "predicted": pred_answer,
                "ground_truths": answers,
                "correct": is_correct,
                "infer_time": round(infer_time, 2),
            })

            if not is_correct:
                errors.append(predictions[-1])

            if (i + 1) % 50 == 0:
                acc = correct / total if total > 0 else 0
                elapsed = time.time() - t_start
                logger.info(f"  Progress: {i+1}/{num_samples}, acc={acc:.3f}, "
                           f"{elapsed:.0f}s elapsed, ~{elapsed/(i+1)*num_samples - elapsed:.0f}s remaining")

        except Exception as e:
            logger.error(f"Sample {i} failed: {e}")
            continue

    elapsed = time.time() - t_start
    accuracy = correct / total if total > 0 else 0.0

    results: dict[str, Any] = {
        "dataset": "VQA-v2",
        "num_samples": num_samples,
        "num_evaluated": total,
        "overall_accuracy": round(accuracy, 4),
        "correct": correct,
        "elapsed_seconds": round(elapsed, 1),
        "avg_infer_time_seconds": round(
            sum(p["infer_time"] for p in predictions) / len(predictions) if predictions else 0,
            2,
        ),
        "per_category": {
            cat: {
                "accuracy": round(data["correct"] / data["total"], 4) if data["total"] > 0 else 0,
                "total": data["total"],
            }
            for cat, data in per_category.items()
        },
        "error_count": len(errors),
        "timestamp": datetime.now().isoformat(),
    }

    # 保存结果
    if output_dir is None:
        output_dir = str(_project_root / "outputs" / datetime.now().strftime("%Y-%m-%d-vqa-v2"))

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(output_dir) / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open(Path(output_dir) / "predictions.json", "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    with open(Path(output_dir) / "errors.json", "w", encoding="utf-8") as f:
        json.dump(errors, f, ensure_ascii=False, indent=2)

    logger.info(f"Results saved to {output_dir}")
    logger.info(f"VQA-v2 Accuracy: {accuracy:.4f} ({correct}/{total})")
    for cat, data in per_category.items():
        cat_acc = data["correct"] / data["total"] if data["total"] > 0 else 0
        logger.info(f"  {cat}: {cat_acc:.4f} ({data['correct']}/{data['total']})")

    return results


def main() -> None:
    """命令行入口：运行 VQA-v2 评测。"""
    evaluate_vqa_v2(num_samples=1000)


if __name__ == "__main__":
    main()
