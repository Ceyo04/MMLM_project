"""
model_loader.py - 模型加载

封装 Qwen2.5-VL 模型加载逻辑，使用 bfloat16 精度。
"""

import logging

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from src.config.settings import Config, get_config

logger = logging.getLogger(__name__)

# 全局缓存的 model 和 processor（进程级单例）
_model = None
_processor = None


def load_model(cfg: Config | None = None) -> tuple[Qwen2_5_VLForConditionalGeneration, AutoProcessor]:
    """
    加载 Qwen2.5-VL 模型和 Processor。

    模型使用 torch.bfloat16 精度加载，适配 8GB VRAM。
    返回 (model, processor) 元组，结果会缓存在模块级变量中。
    """
    global _model, _processor

    if _model is not None and _processor is not None:
        logger.info("Model and processor already loaded, returning cached instances")
        return _model, _processor

    if cfg is None:
        cfg = get_config()

    logger.info(f"Loading model from {cfg.model_path}")
    logger.info(f"Device: {cfg.device}, dtype: bfloat16")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        cfg.model_path,
        torch_dtype=torch.bfloat16,
        device_map=cfg.device,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    logger.info(f"Model loaded on device: {model.device}")

    processor = AutoProcessor.from_pretrained(
        cfg.model_path,
        trust_remote_code=True,
        min_pixels=cfg.min_pixels_actual,
        max_pixels=cfg.max_pixels_actual,
        use_fast=True,
    )
    logger.info(f"Processor loaded, max_pixels={cfg.max_pixels_actual}")

    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        used = total - free
        logger.info(f"VRAM: {used / 1024**3:.1f} GB used / {total / 1024**3:.1f} GB total")

    _model = model
    _processor = processor
    return model, processor


def clear_cache() -> None:
    """清除缓存的模型（释放显存）。"""
    global _model, _processor
    if _model is not None:
        _model.cpu()
        del _model
        _model = None
    if _processor is not None:
        del _processor
        _processor = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Model cache cleared")
