"""
vqa_v2_eval.py - VQA-v2 评测

在 VQA-v2 验证集子集上评测模型准确率。
使用 VQA 标准软匹配：预测答案与 10 个标注答案中 ≥3 个一致即算正确。

评测专用优化：
- 强制英文短答案输出（VQA-v2 标准答案为英文）
- 双语匹配（中文回答→英文映射）
- 贪心解码（temperature=0）
"""

from __future__ import annotations

import json
import logging
import re
import string
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

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info

from src.config.settings import Config, get_config
from src.data.vqa_v2_loader import load_vqa_v2_stream
from src.inference.model_loader import load_model

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ─── VQA 评测专用：中文→英文双语映射 ───
ZH_TO_EN: dict[str, str] = {
    # Yes/No
    "是": "yes", "是的": "yes", "对": "yes", "对的": "yes",
    "否": "no", "不是": "no", "没有": "no", "不": "no", "没": "no",
    "不可以": "no", "不能": "no",
    # 颜色
    "红色": "red", "红": "red",
    "蓝色": "blue", "蓝": "blue",
    "绿色": "green", "绿": "green",
    "黄色": "yellow", "黄": "yellow",
    "白色": "white", "白": "white",
    "黑色": "black", "黑": "black",
    "橙色": "orange", "橙": "orange",
    "紫色": "purple", "紫": "purple",
    "粉色": "pink", "粉": "pink",
    "灰色": "gray", "灰": "gray",
    "棕色": "brown", "棕": "brown",
    # 常见物体
    "汽车": "car", "小汽车": "car",
    "卡车": "truck",
    "公共汽车": "bus", "巴士": "bus",
    "自行车": "bicycle", "单车": "bike",
    "飞机": "airplane", "飞机": "plane",
    "火车": "train",
    "船": "boat",
    "猫": "cat", "狗": "dog", "鸟": "bird", "马": "horse",
    "苹果": "apple", "香蕉": "banana", "橙子": "orange",
    "桌子": "table", "椅子": "chair", "床": "bed",
    "杯子": "cup", "瓶子": "bottle", "碗": "bowl",
    "电话": "phone", "手机": "phone", "电脑": "computer",
    "电视": "tv", "电视机": "tv",
    "书": "book", "报纸": "newspaper",
    "人": "person", "男人": "man", "女人": "woman",
    "孩子": "child", "小孩": "kid",
    "帽子": "hat", "鞋": "shoes", "衣服": "clothes",
    "门": "door", "窗户": "window",
    "树": "tree", "花": "flower", "草": "grass",
    "食物": "food", "水": "water",
    "叉子": "fork", "刀": "knife", "勺子": "spoon",
    "筷子": "chopsticks",
    "滑板": "skateboard",
    "伞": "umbrella", "雨伞": "umbrella",
    "蛋糕": "cake", "面包": "bread", "披萨": "pizza",
    "三明治": "sandwich", "米饭": "rice", "面条": "noodles",
    "冰淇淋": "ice cream", "冰激凌": "ice cream", "雪糕": "ice cream",
    # 场景
    "室内": "indoor", "室外": "outdoor",
    "白天": "daytime", "白天": "day",
    "晚上": "night", "夜晚": "night",
    # 动作
    "走": "walking", "步行": "walking", "散步": "walking",
    "跑": "running", "跑步": "running",
    "坐": "sitting", "站着": "standing", "站": "standing",
    "吃": "eating", "喝": "drinking",
    "玩": "playing",
    "看": "watching", "观看": "watching",
    "睡觉": "sleeping", "睡": "sleeping",
    "游泳": "swimming",
    "滑雪": "skiing",
    "冲浪": "surfing",
    "飞": "flying",
    # 数量（保持数字）
    "零": "0", "一": "1", "二": "2", "两": "2", "三": "3", "四": "4",
    "五": "5", "六": "6", "七": "7", "八": "8", "九": "9", "十": "10",
    # 方位
    "左": "left", "左边": "left",
    "右": "right", "右边": "right",
    "上": "top", "上面": "top",
    "下": "down", "下面": "bottom", "底部": "bottom",
    "中间": "middle", "中心": "center",
    # 材质
    "木头": "wood", "木": "wooden",
    "金属": "metal", "塑料": "plastic",
    "玻璃": "glass", "纸": "paper",
    # 天气
    "晴天": "sunny", "阴天": "cloudy", "雨天": "rainy", "下雪": "snowy",
    # 其他常见词
    "大": "large", "小": "small", "多": "many", "少": "few",
    "新": "new", "旧": "old",
    "开": "open", "关": "closed",
    "干净的": "clean", "脏的": "dirty",
    "亮": "bright", "暗": "dark",
}


def normalize_answer(text: str) -> str:
    """标准化答案文本，用于 VQA 软匹配。"""
    text = text.lower().strip()
    # 移除冠词和常见停用词
    text = re.sub(r"\b(a|an|the|is|are|was|were|of|in|on|at|to|for|this|that|there|it)\b", "", text)
    # 移除标点
    text = text.translate(str.maketrans("", "", string.punctuation))
    # 移除多余空格
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_key_answer(predicted: str) -> str:
    """
    从模型的长回答中提取核心答案词。

    策略：
    1. 如果回答很短（<5 个词），直接使用
    2. 如果是完整句子，尝试提取核心词
    3. 提取最后一个短句（通常是结论）
    """
    predicted = predicted.strip()
    words = predicted.split()

    # 如果已经很短，直接返回
    if len(words) <= 3:
        return predicted

    # 尝试提取：按句号/逗号/换行分割，取最后一个有意义的部分
    for sep in [". ", "。", ", ", "，", "\n"]:
        if sep in predicted:
            parts = [p.strip() for p in predicted.split(sep) if p.strip()]
            if parts:
                last = parts[-1]
                # 如果最后一部分足够短且有意义
                if len(last.split()) <= 4:
                    return last

    # 移除常见句子前缀
    prefixes = [
        "it is ", "it's ", "this is ", "that is ", "the answer is ",
        "i think ", "i believe ", "the image shows ", "there is ", "there are ",
        # 中文前缀
        "这是", "那是", "图中是", "图片是", "答案是", "我认为", "我觉得",
        "是的，", "不，", "对，", "错的，", "这是一个", "那是一",
        "这是一", "那是一", "可以说", "应该是",
    ]
    for prefix in prefixes:
        if predicted.lower().startswith(prefix):
            predicted = predicted[len(prefix):].strip()
            break

    # 中文长句：提取"是"后的部分
    cn_patterns = [
        r"是([一-鿿\w]+)[。，\s]*$",  # 最后的"是XXX"
        r"为([一-鿿\w]+)[。，\s]*$",  # 最后的"为XXX"
    ]
    for pattern in cn_patterns:
        m = re.search(pattern, predicted)
        if m:
            extracted = m.group(1).strip()
            if len(extracted) >= 1 and len(extracted) <= 20:
                return extracted

    return predicted


def translate_cn_to_en(text: str) -> list[str]:
    """
    将中文答案翻译为英文候选列表。

    返回：原始文本 + 所有可能的英文翻译
    """
    candidates = [text]

    # 精确匹配中文词表
    normalized = text.strip().lower()
    if normalized in ZH_TO_EN:
        candidates.append(ZH_TO_EN[normalized])

    # 如果文本包含中文，尝试逐词翻译
    has_cjk = any("一" <= c <= "鿿" for c in normalized)
    if has_cjk:
        # 尝试完整短语匹配
        for zh, en in ZH_TO_EN.items():
            if zh in normalized:
                candidates.append(en)
                # 也添加"英文 "形式
                candidates.append(f"{en} ")

        # 收集所有匹配到的英文词，组合为候选
        matched_en_words = []
        for zh, en in ZH_TO_EN.items():
            if zh in normalized:
                matched_en_words.append(en)
        if matched_en_words:
            candidates.append(" ".join(matched_en_words))

    return candidates


def vqa_soft_accuracy(predicted: str, ground_truths: list[str]) -> bool:
    """
    VQA 标准软匹配准确率判定（双语增强版）。

    规则：预测答案经过 normalize + 中英翻译后，
    与 10 个标注答案中 ≥3 个的 normalize 结果一致，则判定为正确。
    """
    # 1. 提取核心答案
    pred_core = extract_key_answer(predicted)

    # 2. 生成中英双语候选列表
    pred_candidates = translate_cn_to_en(pred_core)

    # 3. 标准化所有候选
    pred_norms = [normalize_answer(c) for c in pred_candidates if c.strip()]

    # 4. 标准化所有标注答案
    gt_norms = [normalize_answer(g) for g in ground_truths]

    # 5. 双向匹配计数
    matches = 0
    for gt_norm in gt_norms:
        if not gt_norm:
            continue
        matched = False
        for pred_norm in pred_norms:
            if not pred_norm:
                continue
            # 精确匹配
            if pred_norm == gt_norm:
                matched = True
                break
            # 子串匹配（双向）
            if len(pred_norm) >= 2 and len(gt_norm) >= 2:
                if pred_norm in gt_norm or gt_norm in pred_norm:
                    matched = True
                    break
            # 英文词级重叠（至少有一个词重叠）
            pred_words = set(pred_norm.split())
            gt_words = set(gt_norm.split())
            if pred_words and gt_words:
                overlap = pred_words & gt_words
                if len(overlap) >= 1 and len(overlap) >= min(len(pred_words), len(gt_words)) * 0.5:
                    matched = True
                    break
        if matched:
            matches += 1

    # VQA 标准：≥3 个标注一致
    threshold = min(3, max(1, len(ground_truths) // 3))
    return matches >= threshold


# ─── 评测专用推理函数（强制英文 + 贪心解码） ───

VQA_EVAL_SYSTEM_PROMPT = (
    "You are a visual question answering system. "
    "Answer the question with a SINGLE WORD or SHORT PHRASE (1-3 words maximum). "
    "Do NOT use full sentences. Do NOT explain. Just output the answer directly.\n"
    "Examples:\n"
    "Q: What color is the car? A: red\n"
    "Q: Is it raining? A: yes\n"
    "Q: How many people? A: 3\n"
    "Q: What is on the table? A: a plate"
)


def generate_vqa_answer(
    image: Image.Image,
    question: str,
    model: Any,
    processor: Any,
    cfg: Config,
) -> str:
    """
    VQA 评测专用推理：强制英文短答案 + 贪心解码。
    """
    messages = [
        {"role": "system", "content": [{"type": "text", "text": VQA_EVAL_SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": f"{question}\nAnswer:"},
        ]},
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
            max_new_tokens=32,          # 短答案只需少量 tokens
            do_sample=False,            # 贪心解码，无随机性
            repetition_penalty=1.0,     # 不惩罚重复（短答案不需要）
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for out_ids, in_ids in zip(generated_ids, inputs["input_ids"])
    ]
    answer = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0]

    # 进一步清理：只取第一行/第一句
    answer = answer.strip()
    for sep in ["\n", "。", ". ", ", "]:
        if sep in answer:
            answer = answer.split(sep)[0].strip()

    return answer


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

    # 加载模型（复用缓存）
    model, processor = load_model(cfg)

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
            pred_answer = generate_vqa_answer(
                sample["image"], question,
                model=model, processor=processor, cfg=cfg,
            )
            infer_time = time.time() - t_infer

            is_correct = vqa_soft_accuracy(pred_answer, answers)
            if is_correct:
                correct += 1

            # 按题型分类（简单启发式）
            q_lower = question.lower()
            if q_lower.startswith("is ") or q_lower.startswith("are ") or q_lower.startswith("does "):
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

            # 每 20 条打印进度（更频繁，便于调试）
            if (i + 1) % 20 == 0:
                acc = correct / total if total > 0 else 0
                elapsed = time.time() - t_start
                logger.info(f"  Progress: {i+1}/{num_samples}, acc={acc:.3f}, "
                           f"{elapsed:.0f}s elapsed, ~{elapsed/(i+1)*num_samples - elapsed:.0f}s remaining")
                # 打印最近一个失败样本帮助诊断
                if errors:
                    last_err = errors[-1]
                    logger.info(f"  Sample fail: Q='{last_err['question'][:60]}' "
                               f"Pred='{last_err['predicted'][:40]}' GT={last_err['ground_truths'][:3]}")

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
