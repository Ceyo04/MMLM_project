"""
custom_eval.py - 自建中文数据集评测工具

对自建中文图文问答集进行模型推理和人工打分辅助。
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

from PIL import Image

from src.config.settings import get_config
from src.inference.generate import generate_single

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def run_inference_on_custom_dataset(
    dataset_path: str,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """
    对自建中文数据集执行模型推理。

    Args:
        dataset_path: 数据集 JSON 文件路径
        output_dir: 结果输出目录

    Returns:
        {results, predictions} 字典
    """
    cfg = get_config()

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    logger.info(f"Loaded custom dataset: {len(dataset)} images")

    predictions: list[dict[str, Any]] = []
    total_qa = 0

    t_start = time.time()

    # 数据集文件所在目录，用于解析相对路径
    dataset_dir = Path(dataset_path).resolve().parent

    for item in dataset:
        image_rel = item["image"]
        category = item.get("category", "auto")
        qa_pairs = item.get("qa_pairs", [])

        # 解析图片路径：优先绝对路径，否则相对 dataset 目录
        image_path = Path(image_rel)
        if not image_path.is_absolute():
            image_path = dataset_dir / image_rel

        if not image_path.exists():
            logger.warning(f"Image not found: {image_path}")
            continue

        for qa in qa_pairs:
            question = qa["question"]
            reference_answer = qa["answer"]

            try:
                t_infer = time.time()
                predicted, _ = generate_single(
                    str(image_path),
                    question,
                    scene=category,
                )
                infer_time = time.time() - t_infer

                predictions.append({
                    "image": str(image_path),
                    "category": category,
                    "question": question,
                    "predicted": predicted,
                    "reference_answer": reference_answer,
                    "human_score": None,  # 待人工打分
                    "infer_time": round(infer_time, 2),
                })
                total_qa += 1

            except Exception as e:
                logger.error(f"Inference failed for {image_path}: {e}")
                continue

    elapsed = time.time() - t_start

    if output_dir is None:
        output_dir = str(_project_root / "outputs" / datetime.now().strftime("%Y-%m-%d-custom"))

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 输出待人工打分的预测文件
    scoring_file = Path(output_dir) / "predictions_for_scoring.json"
    with open(scoring_file, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    results = {
        "dataset": "custom_chinese",
        "num_images": len(dataset),
        "num_qa_pairs": total_qa,
        "elapsed_seconds": round(elapsed, 1),
        "avg_infer_time_seconds": round(
            sum(p["infer_time"] for p in predictions) / len(predictions) if predictions else 0,
            2,
        ),
        "output_file": str(scoring_file),
        "timestamp": datetime.now().isoformat(),
    }

    with open(Path(output_dir) / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(f"Custom dataset inference complete: {total_qa} QA pairs")
    logger.info(f"Results saved to {output_dir}")
    logger.info(f"Scoring file: {scoring_file} (请打开文件逐条填写 human_score)")

    return results


def compute_human_scores(scoring_file: str) -> dict[str, Any]:
    """
    从人工打分后的文件中计算评分统计。

    Args:
        scoring_file: 包含 human_score 字段的预测文件路径

    Returns:
        统计结果
    """
    with open(scoring_file, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    scored = [p for p in predictions if p.get("human_score") is not None]
    if not scored:
        logger.warning("No human scores found in file")
        return {"error": "no scores"}

    scores = [p["human_score"] for p in scored]
    mean_score = sum(scores) / len(scores)
    variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
    std_dev = variance ** 0.5

    # 按类别统计
    by_category: dict[str, list[float]] = {}
    for p in scored:
        cat = p.get("category", "unknown")
        by_category.setdefault(cat, []).append(p["human_score"])

    stats = {
        "num_scored": len(scored),
        "total_qa": len(predictions),
        "mean_score": round(mean_score, 2),
        "std_dev": round(std_dev, 2),
        "score_distribution": {
            str(i): len([s for s in scores if s == i]) for i in range(1, 6)
        },
        "by_category": {
            cat: {
                "count": len(s),
                "mean": round(sum(s) / len(s), 2),
                "std": round((sum((x - sum(s)/len(s))**2 for x in s) / len(s)) ** 0.5, 2),
            }
            for cat, s in by_category.items()
        },
    }

    output_path = Path(scoring_file).parent / "scoring_stats.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    logger.info(f"Mean human score: {mean_score:.2f} ± {std_dev:.2f} (n={len(scored)})")
    logger.info(f"Stats saved to {output_path}")

    return stats


def main() -> None:
    """命令行入口。"""
    import argparse
    parser = argparse.ArgumentParser(description="自建中文数据集评测")
    parser.add_argument("--dataset", type=str, required=True, help="数据集 JSON 文件路径")
    parser.add_argument("--score", type=str, help="已完成人工打分的预测文件路径（计算统计）")
    args = parser.parse_args()

    if args.score:
        compute_human_scores(args.score)
    else:
        run_inference_on_custom_dataset(args.dataset)


if __name__ == "__main__":
    main()
