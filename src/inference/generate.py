"""
generate.py - 推理生成

提供单轮和多轮对话推理 API 及命令行入口。
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

# 确保项目根目录在 sys.path 中（支持直接执行脚本或作为模块导入）
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info

from src.config.settings import get_config
from src.inference.model_loader import load_model

logger = logging.getLogger(__name__)

# 对话历史类型
Message = dict[str, Any]
History = list[Message]


def generate_single(
    image: Image.Image | str,
    question: str,
    history: History | None = None,
    scene: str = "auto",
) -> tuple[str, History]:
    """
    单轮推理：根据图片和问题生成回答。

    Args:
        image: PIL Image 或图片路径
        question: 用户问题
        history: 可选的对话历史
        scene: 场景类型 (auto / natural_scene / document_scene)

    Returns:
        (answer, updated_history) 元组
    """
    cfg = get_config()
    model, processor = load_model(cfg)

    if isinstance(image, str):
        image = Image.open(image).convert("RGB")

    # 获取 System Prompt
    system_prompt = cfg.prompt_templates.get(scene, cfg.prompt_templates.get("auto", ""))

    # 构造 messages
    if history is None:
        history = [{"role": "system", "content": [{"type": "text", "text": system_prompt}]}]

    user_content: list[dict[str, Any]] = [
        {"type": "image", "image": image},
        {"type": "text", "text": question},
    ]
    history.append({"role": "user", "content": user_content})

    # 应用 chat template
    text = processor.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(history)

    # Tokenize
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # 生成
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature if cfg.do_sample else None,
            top_p=cfg.top_p if cfg.do_sample else None,
            do_sample=cfg.do_sample,
            repetition_penalty=cfg.repetition_penalty,
        )

    # 解码回答
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for out_ids, in_ids in zip(generated_ids, inputs["input_ids"])
    ]
    answer = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    # 更新历史
    history.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})

    return answer, history


def generate_multi(
    image: Image.Image | str,
    question: str,
    history: History | None = None,
    scene: str = "auto",
) -> tuple[str, History]:
    """
    多轮对话推理。

    与 generate_single 接口相同，区别在于首次调用时自动设置 System Prompt。
    后续调用应传入 history 以保持上下文。

    Args:
        image: PIL Image 或图片路径（多轮对话中图片保持不变）
        question: 用户问题
        history: 对话历史，首次调用传 None，后续传入返回值中的 history
        scene: 场景类型

    Returns:
        (answer, updated_history) 元组
    """
    return generate_single(image, question, history, scene)


def reset_history(scene: str = "auto") -> History:
    """创建新的对话历史（带 System Prompt）。"""
    cfg = get_config()
    system_prompt = cfg.prompt_templates.get(scene, cfg.prompt_templates.get("auto", ""))
    return [{"role": "system", "content": [{"type": "text", "text": system_prompt}]}]


def main() -> None:
    """命令行入口。"""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Qwen2.5-VL 图文问答推理")
    parser.add_argument("--image", type=str, required=True, help="图片路径")
    parser.add_argument("--question", type=str, required=True, help="用户问题")
    parser.add_argument("--scene", type=str, default="auto",
                        choices=["auto", "natural_scene", "document_scene"],
                        help="场景类型")
    args = parser.parse_args()

    image_path = args.image
    if not Path(image_path).exists():
        logger.error(f"Image not found: {image_path}")
        return

    answer, _ = generate_single(args.image, args.question, scene=args.scene)
    print(f"问题: {args.question}")
    print(f"回答: {answer}")


if __name__ == "__main__":
    main()
