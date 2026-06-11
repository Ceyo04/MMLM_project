"""
03_evaluate.py - VQA-v2 评测脚本
功能：
  1. 从VQA-v2验证集中随机抽取指定数量的图片
  2. 使用Qwen2.5-VL-7B-Instruct进行批量推理
  3. 按VQA标准策略计算准确率（答案出现>=3次算正确）
  4. 输出评测报告和典型案例
"""

import json
import os
import random
import time
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info


# ==================== 配置 ====================
# 路径
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "Qwen2.5-VL-7B-Instruct"
DATA_DIR = BASE_DIR / "data" / "VQA"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 评测参数
NUM_SAMPLES = 100
RANDOM_SEED = 42
MAX_NEW_TOKENS = 20
IMAGE_DIR = DATA_DIR / "val2014"
QUESTIONS_FILE = DATA_DIR / "v2_OpenEnded_mscoco_val2014_questions.json"
ANNOTATIONS_FILE = DATA_DIR / "v2_mscoco_val2014_annotations.json"


def load_vqa_data():
    """加载VQA-v2验证集数据，返回 {question_id: {question, image_id, answers}} 的映射"""
    print("加载数据...")
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        questions_data = json.load(f)
    with open(ANNOTATIONS_FILE, "r", encoding="utf-8") as f:
        annotations_data = json.load(f)

    # 构建 question_id -> question 映射
    qid_to_question = {}
    for q in questions_data["questions"]:
        qid_to_question[q["question_id"]] = {
            "question": q["question"],
            "image_id": q["image_id"],
        }

    # 构建 question_id -> answers 映射
    qid_to_answers = {}
    for a in annotations_data["annotations"]:
        qid_to_answers[a["question_id"]] = {
            "answers": [ans["answer"] for ans in a["answers"]],
            "image_id": a["image_id"],
        }

    # 合并：只保留问题和标注都存在的question_id
    valid_qids = set(qid_to_question.keys()) & set(qid_to_answers.keys())
    data = {}
    for qid in valid_qids:
        # 检查图片是否存在
        image_id = qid_to_question[qid]["image_id"]
        image_path = IMAGE_DIR / f"COCO_val2014_{image_id:012d}.jpg"
        if image_path.exists():
            data[qid] = {
                "question": qid_to_question[qid]["question"],
                "image_id": image_id,
                "image_path": str(image_path),
                "answers": qid_to_answers[qid]["answers"],
            }

    print(f"  有效数据: {len(data)} 条（已有图片的）")
    return data


def sample_data(data, num_samples, seed):
    """从数据中随机采样"""
    random.seed(seed)
    sampled_qids = random.sample(list(data.keys()), min(num_samples, len(data)))
    return {qid: data[qid] for qid in sampled_qids}


def load_model():
    """加载模型和处理器（8-bit量化）"""
    from transformers import BitsAndBytesConfig
    print(f"加载模型: {MODEL_PATH}")
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(MODEL_PATH),
        torch_dtype=torch.float16,
        device_map="auto",
        quantization_config=quantization_config,
    )
    processor = AutoProcessor.from_pretrained(str(MODEL_PATH))
    return model, processor


def infer_single(model, processor, image_path, question):
    """对单张图片+问题进行推理，返回模型输出的答案文本"""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": f"Answer the question using a single word or short phrase.\nQuestion: {question}"},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    # 截取生成部分
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text[0].strip()


def clean_answer(text):
    """清洗模型输出，去除标点和多余空格，转为小写"""
    import re
    text = text.lower().strip()
    # 去除句号、逗号等常见标点
    text = re.sub(r"[.,;:!?\"\'()\[\]{}]", "", text)
    text = text.strip()
    return text


def is_correct(predicted, gt_answers):
    """VQA标准匹配策略：预测答案与出现>=3次的ground truth答案匹配则算正确"""
    pred_clean = clean_answer(predicted)
    # 统计每个gt答案的出现次数
    answer_counts = {}
    for a in gt_answers:
        a_clean = clean_answer(a)
        answer_counts[a_clean] = answer_counts.get(a_clean, 0) + 1

    # 检查预测答案是否匹配任何一个出现>=3次的答案
    for ans, count in answer_counts.items():
        if count >= 3 and pred_clean == ans:
            return True
    return False


def run_evaluation():
    """主评测流程"""
    print("=" * 60)
    print("VQA-v2 评测开始")
    print("=" * 60)

    # 1. 加载数据
    data = load_vqa_data()
    sampled = sample_data(data, NUM_SAMPLES, RANDOM_SEED)
    print(f"随机采样: {len(sampled)} 条（种子={RANDOM_SEED}）")

    # 2. 加载模型
    model, processor = load_model()

    # 3. 批量推理
    print("\n开始推理...")
    results = []
    correct_count = 0
    total_infer_time = 0.0

    for qid, item in tqdm(sampled.items(), desc="推理进度"):
        start_time = time.time()
        predicted_answer = infer_single(model, processor, item["image_path"], item["question"])
        infer_time = time.time() - start_time
        total_infer_time += infer_time

        correct = is_correct(predicted_answer, item["answers"])
        if correct:
            correct_count += 1

        results.append({
            "question_id": qid,
            "image_id": item["image_id"],
            "question": item["question"],
            "predicted": predicted_answer,
            "gt_answers": item["answers"],
            "correct": correct,
            "infer_time": infer_time,
        })

    # 4. 统计
    accuracy = correct_count / len(sampled) * 100
    avg_time = total_infer_time / len(sampled)

    print("\n" + "=" * 60)
    print("评测结果")
    print("=" * 60)
    print(f"总样本数: {len(sampled)}")
    print(f"正确数:   {correct_count}")
    print(f"准确率:   {accuracy:.2f}%")
    print(f"平均推理时间: {avg_time:.2f}s")
    print(f"总推理时间:   {total_infer_time:.2f}s")

    # 5. 保存详细结果
    output_file = OUTPUT_DIR / f"evaluate_results_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "num_samples": NUM_SAMPLES,
                "random_seed": RANDOM_SEED,
                "model": str(MODEL_PATH),
            },
            "summary": {
                "accuracy": accuracy,
                "correct_count": correct_count,
                "total_samples": len(sampled),
                "avg_infer_time": avg_time,
                "total_infer_time": total_infer_time,
            },
            "details": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果已保存: {output_file}")

    # 6. 打印典型案例
    print("\n" + "=" * 60)
    print("典型案例（前5个成功 & 前5个失败）")
    print("=" * 60)
    success_cases = [r for r in results if r["correct"]][:5]
    fail_cases = [r for r in results if not r["correct"]][:5]

    print("\n--- 成功案例 ---")
    for i, case in enumerate(success_cases, 1):
        print(f"\n案例{i}:")
        print(f"  问题: {case['question']}")
        print(f"  模型回答: {case['predicted']}")
        print(f"  标注答案: {case['gt_answers'][:3]}...")

    print("\n--- 失败案例 ---")
    for i, case in enumerate(fail_cases, 1):
        print(f"\n案例{i}:")
        print(f"  问题: {case['question']}")
        print(f"  模型回答: {case['predicted']}")
        print(f"  标注答案: {case['gt_answers'][:3]}...")

    return results


if __name__ == "__main__":
    run_evaluation()