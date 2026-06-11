"""
settings.py - 集中管理所有路径、超参、Prompt 模板

从 configs/ 目录加载 YAML 配置，提供统一的 Config 数据类。
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# 项目根目录（prompt.md 所在目录）
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _load_yaml(rel_path: str) -> dict[str, Any]:
    """加载 YAML 文件，文件不存在时返回空 dict。"""
    path = PROJECT_ROOT / rel_path
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def _deep_merge(base: dict, override: dict) -> dict:
    """递归合并 override 到 base，override 中的值优先。"""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


@dataclass
class Config:
    """全局配置，从 YAML 文件加载。"""
    model_name: str = "Qwen2.5-VL-3B-Instruct"
    model_path: str = ""  # 必须通过 local.yaml 设置

    max_new_tokens: int = 256
    temperature: float = 0.3
    top_p: float = 0.9
    do_sample: bool = False
    repetition_penalty: float = 1.05

    min_pixels: int = 256  # 256 * 28 * 28 = 200704
    max_pixels: int = 640  # 640 * 28 * 28 = 501760

    device: str = "auto"

    prompt_templates: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls) -> "Config":
        """从 configs/ 目录加载并合并配置。"""
        default_data = _load_yaml("configs/default.yaml")
        local_data = _load_yaml("configs/local.yaml")
        prompt_data = _load_yaml("configs/prompt_templates.yaml")

        merged = _deep_merge(default_data, local_data)

        return cls(
            model_name=merged.get("model", {}).get("name", cls.model_name),
            model_path=merged.get("model", {}).get("path", cls.model_path),
            max_new_tokens=merged.get("generation", {}).get("max_new_tokens", cls.max_new_tokens),
            temperature=merged.get("generation", {}).get("temperature", cls.temperature),
            top_p=merged.get("generation", {}).get("top_p", cls.top_p),
            do_sample=merged.get("generation", {}).get("do_sample", cls.do_sample),
            repetition_penalty=merged.get("generation", {}).get("repetition_penalty", cls.repetition_penalty),
            min_pixels=merged.get("image", {}).get("min_pixels", cls.min_pixels),
            max_pixels=merged.get("image", {}).get("max_pixels", cls.max_pixels),
            device=merged.get("device", cls.device),
            prompt_templates=prompt_data,
        )

    @property
    def min_pixels_actual(self) -> int:
        """返回实际像素值（配置值 * 28*28）。"""
        return self.min_pixels * 28 * 28

    @property
    def max_pixels_actual(self) -> int:
        """返回实际像素值（配置值 * 28*28）。"""
        return self.max_pixels * 28 * 28

    def validate(self) -> list[str]:
        """验证配置完整性，返回错误列表。"""
        errors = []
        if not self.model_path:
            errors.append("model_path 未设置，请在 configs/local.yaml 中配置")
        return errors


# 全局单例
_config: Config | None = None


def get_config() -> Config:
    """获取全局配置单例。"""
    global _config
    if _config is None:
        _config = Config.load()
        issues = _config.validate()
        if issues:
            for issue in issues:
                print(f"[WARNING] {issue}")
    return _config
