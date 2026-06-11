"""
textvqa_eval.py - TextVQA 评测

参考 VQA 标准评测流程，适配 TextVQA 数据集。
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import torch
from qwen_vl_utils import process_vision_info

from src.config.settings import Config, get_config
from src.data.textvqa_loader import load_textvqa_stream
from src.inference.model_loader import load_model

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

MAX_NEW_TOKENS = 20

TEXTVQA_PROMPT = "Answer the question using a single word or short phrase."


def clean_answer(text: str) -> str:
    """清洗答案文本。"""
    text = text.lower().strip()
    text = re.sub(r"[.,;:!?\"\'()\[\]{}]", "", text)
    text = text.strip()
    return text


def is_correct(predicted: str, gt_answers: list[str]) -> bool:
    """VQA 标准匹配策略。"""
    pred_clean = clean_answer(predicted)
    answer_counts: dict[str, int] = {}
    for a in gt_answers:
        a_clean = clean_answer(a)
        answer_counts[a_clean] = answer_counts.get(a_clean, 0) + 1
    for ans, count in answer_counts.items():
        if count >= 3 and pred_clean == ans:
            return True
    return False


def evaluate_textvqa(
    num_samples: int = 100,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """运行 TextVQA 评测。"""
    cfg = get_config()
    logger.info(f"TextVQA Evaluation: {num_samples} samples")

    model, processor = load_model(cfg)
    results_details: list[dict] = []
    correct_count = 0
    total_count = 0

    t_start = time.time()

    for i, sample in enumerate(load_textvqa_stream(max_samples=num_samples)):
        total_count += 1
        question = sample["question"]
        answers = sample["answers"]

        try:
            t0 = time.time()
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": sample["image"]},
                    {"type": "text", "text": f"{TEXTVQA_PROMPT}\nQuestion: {question}"},
                ],
            }]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text], images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt",
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)

            generated_trimmed = [out[len(inp):] for out, inp in zip(generated_ids, inputs["input_ids"])]
            predicted = processor.batch_decode(generated_trimmed, skip_special_tokens=True,
                                                clean_up_tokenization_spaces=False)[0].strip()
            infer_time = time.time() - t0

            correct = is_correct(predicted, answers)
            if correct:
                correct_count += 1

            results_details.append({
                "question_id": sample.get("question_id", i),
                "question": question,
                "predicted": predicted,
                "gt_answers": answers,
                "correct": correct,
                "infer_time": round(infer_time, 4),
            })

            if (i + 1) % 20 == 0:
                acc = correct_count / total_count * 100 if total_count else 0
                logger.info(f"  [{i+1}/{num_samples}] acc={acc:.1f}%")

        except Exception as e:
            logger.error(f"TextVQA sample {i} error: {e}")
            continue

    elapsed = time.time() - t_start
    accuracy = correct_count / total_count * 100 if total_count else 0

    if output_dir is None:
        output_dir = str(_project_root / "outputs" / datetime.now().strftime("%Y-%m-%d-textvqa"))
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    results = {
        "config": {"num_samples": num_samples, "model": cfg.model_name},
        "summary": {
            "accuracy": round(accuracy, 2),
            "correct_count": correct_count,
            "total_samples": total_count,
        },
        "details": results_details,
        "timestamp": datetime.now().isoformat(),
    }

    output_file = Path(output_dir) / f"evaluate_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(f"TextVQA Accuracy: {accuracy:.1f}% ({correct_count}/{total_count})")
    logger.info(f"Results: {output_file}")
    return results


def main() -> None:
    evaluate_textvqa(num_samples=100)


if __name__ == "__main__":
    main()
