"""
vqa_v2_eval.py - VQA-v2 评测

参考 VQA 标准评测流程：
1. 强制英文短答案 Prompt
2. 简单清洗（去标点 + 小写）
3. 标准匹配策略（标注答案中出现 >=3 次且与预测一致则算正确）
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

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import torch
from qwen_vl_utils import process_vision_info

from src.config.settings import Config, get_config
from src.data.vqa_v2_loader import load_vqa_v2_stream
from src.inference.model_loader import load_model

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# VQA 评测专用参数
MAX_NEW_TOKENS = 20  # 强制短答案

# VQA 评测专用 Prompt —— 强制单词/短语输出
VQA_SYSTEM_PROMPT = (
    "Answer the question using a single word or short phrase."
)


def clean_answer(text: str) -> str:
    """清洗答案：去标点、转小写、去多余空格。"""
    text = text.lower().strip()
    text = re.sub(r"[.,;:!?\"\'()\[\]{}]", "", text)
    text = text.strip()
    return text


def is_correct(predicted: str, gt_answers: list[str]) -> bool:
    """
    VQA 标准匹配策略。

    规则：统计每个标注答案在清洗后的出现次数，
    若预测答案与出现 >=3 次的标注答案一致，则判为正确。
    """
    pred_clean = clean_answer(predicted)

    answer_counts: dict[str, int] = {}
    for a in gt_answers:
        a_clean = clean_answer(a)
        answer_counts[a_clean] = answer_counts.get(a_clean, 0) + 1

    for ans, count in answer_counts.items():
        if count >= 3 and pred_clean == ans:
            return True
    return False


def generate_vqa_answer(
    image, question: str, model, processor, cfg: Config,
) -> str:
    """VQA 评测专用推理：短答案 Prompt + 贪心解码。"""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": f"{VQA_SYSTEM_PROMPT}\nQuestion: {question}"},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for out_ids, in_ids in zip(generated_ids, inputs["input_ids"])
    ]
    answer = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return answer.strip()


def evaluate_vqa_v2(
    num_samples: int = 100,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """
    运行 VQA-v2 评测。

    Args:
        num_samples: 评测样本数
        output_dir: 结果输出目录

    Returns:
        评测结果字典
    """
    cfg = get_config()
    logger.info(f"VQA-v2 Evaluation: {num_samples} samples, model={cfg.model_name}")

    model, processor = load_model(cfg)

    results_details: list[dict[str, Any]] = []
    correct_count = 0
    total_count = 0
    total_infer_time = 0.0
    per_category: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})

    t_start = time.time()

    for i, sample in enumerate(load_vqa_v2_stream(max_samples=num_samples)):
        total_count += 1
        question = sample["question"]
        answers = sample["answers"]
        question_id = sample.get("question_id", i)

        try:
            t0 = time.time()
            predicted = generate_vqa_answer(sample["image"], question, model, processor, cfg)
            infer_time = time.time() - t0
            total_infer_time += infer_time

            correct = is_correct(predicted, answers)
            if correct:
                correct_count += 1

            # 按问题类型分类
            q_lower = question.lower()
            if q_lower.startswith(("is ", "are ", "does ", "do ", "can ", "was ", "were ")):
                category = "yes/no"
            elif any(w in q_lower for w in ["how many", "count", "number"]):
                category = "number"
            else:
                category = "other"

            per_category[category]["total"] += 1
            if correct:
                per_category[category]["correct"] += 1

            results_details.append({
                "question_id": question_id,
                "question": question,
                "predicted": predicted,
                "gt_answers": answers,
                "correct": correct,
                "infer_time": round(infer_time, 4),
            })

            if (i + 1) % 20 == 0:
                acc = correct_count / total_count * 100 if total_count > 0 else 0
                elapsed = time.time() - t_start
                logger.info(f"  [{i+1}/{num_samples}] acc={acc:.1f}% ({correct_count}/{i+1}), "
                           f"{elapsed:.0f}s elapsed")

        except Exception as e:
            logger.error(f"Sample {i} error: {e}")
            continue

    elapsed = time.time() - t_start
    accuracy = correct_count / total_count * 100 if total_count > 0 else 0.0
    avg_time = total_infer_time / total_count if total_count > 0 else 0

    # 构建结果
    results: dict[str, Any] = {
        "config": {
            "num_samples": num_samples,
            "model": cfg.model_name,
            "max_new_tokens": MAX_NEW_TOKENS,
            "max_pixels": cfg.max_pixels_actual,
        },
        "summary": {
            "accuracy": round(accuracy, 2),
            "correct_count": correct_count,
            "total_samples": total_count,
            "avg_infer_time": round(avg_time, 4),
            "total_infer_time": round(total_infer_time, 2),
        },
        "per_category": {
            cat: {
                "accuracy": round(data["correct"] / data["total"] * 100, 1) if data["total"] > 0 else 0,
                "correct": data["correct"],
                "total": data["total"],
            }
            for cat, data in per_category.items()
        },
        "details": results_details,
        "timestamp": datetime.now().isoformat(),
    }

    # 保存
    if output_dir is None:
        output_dir = str(_project_root / "outputs" / datetime.now().strftime("%Y-%m-%d-vqa-v2"))
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    output_file = Path(output_dir) / f"evaluate_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 打印摘要
    logger.info("=" * 60)
    logger.info(f"VQA-v2 评测完成")
    logger.info(f"  总样本: {total_count}")
    logger.info(f"  正确:   {correct_count}")
    logger.info(f"  准确率: {accuracy:.1f}%")
    logger.info(f"  平均推理: {avg_time:.2f}s")
    logger.info(f"  总耗时: {elapsed:.0f}s")
    for cat, data in per_category.items():
        cat_acc = data["correct"] / data["total"] * 100 if data["total"] > 0 else 0
        logger.info(f"  {cat}: {cat_acc:.1f}% ({data['correct']}/{data['total']})")
    logger.info(f"  结果保存: {output_file}")

    return results


def main() -> None:
    """命令行入口。"""
    evaluate_vqa_v2(num_samples=100)


if __name__ == "__main__":
    main()
