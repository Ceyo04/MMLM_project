"""
textvqa_eval.py - TextVQA 评测

在 TextVQA 验证集子集上评测模型对图像中文字的阅读理解能力。
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.config.settings import get_config
from src.data.textvqa_loader import load_textvqa_stream
from src.eval.vqa_v2_eval import vqa_soft_accuracy
from src.inference.generate import generate_single

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def evaluate_textvqa(
    num_samples: int = 500,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """
    运行 TextVQA 评测。

    Args:
        num_samples: 评测样本数
        output_dir: 结果输出目录

    Returns:
        评测结果字典
    """
    cfg = get_config()
    logger.info(f"Starting TextVQA evaluation on {num_samples} samples")

    correct = 0
    total = 0
    predictions: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    t_start = time.time()

    for i, sample in enumerate(load_textvqa_stream(max_samples=num_samples)):
        total += 1
        question = sample["question"]
        answers = sample["answers"]

        try:
            t_infer = time.time()
            pred_answer, _ = generate_single(
                sample["image"],
                question,
                scene="document_scene",  # TextVQA 涉及文字识别，用文档场景 Prompt
            )
            infer_time = time.time() - t_infer

            is_correct = vqa_soft_accuracy(pred_answer, answers)
            if is_correct:
                correct += 1

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
                logger.info(f"  Progress: {i+1}/{num_samples}, acc={acc:.3f}, {elapsed:.0f}s elapsed")

        except Exception as e:
            logger.error(f"TextVQA sample {i} failed: {e}")
            continue

    elapsed = time.time() - t_start
    accuracy = correct / total if total > 0 else 0.0

    results: dict[str, Any] = {
        "dataset": "TextVQA",
        "num_samples": num_samples,
        "num_evaluated": total,
        "overall_accuracy": round(accuracy, 4),
        "correct": correct,
        "elapsed_seconds": round(elapsed, 1),
        "avg_infer_time_seconds": round(
            sum(p["infer_time"] for p in predictions) / len(predictions) if predictions else 0,
            2,
        ),
        "error_count": len(errors),
        "timestamp": datetime.now().isoformat(),
    }

    if output_dir is None:
        output_dir = str(_project_root / "outputs" / datetime.now().strftime("%Y-%m-%d-textvqa"))

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(output_dir) / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open(Path(output_dir) / "predictions.json", "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    with open(Path(output_dir) / "errors.json", "w", encoding="utf-8") as f:
        json.dump(errors, f, ensure_ascii=False, indent=2)

    logger.info(f"TextVQA Accuracy: {accuracy:.4f} ({correct}/{total})")
    return results


def main() -> None:
    """命令行入口。"""
    evaluate_textvqa(num_samples=500)


if __name__ == "__main__":
    main()
