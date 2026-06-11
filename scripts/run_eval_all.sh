#!/bin/bash
# run_eval_all.sh - 一键运行全部评测

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

source vlm-env/bin/activate

echo "============================================"
echo "  VLM 智能图文问答助手 - 全量评测"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"

# VQA-v2 评测
echo ""
echo "[1/3] VQA-v2 评测..."
python -m src.eval.vqa_v2_eval
echo "VQA-v2 完成"

# TextVQA 评测
echo ""
echo "[2/3] TextVQA 评测..."
python -m src.eval.textvqa_eval
echo "TextVQA 完成"

# 自建中文数据集评测（仅当数据集存在时运行）
echo ""
echo "[3/3] 自建中文数据集评测..."
CUSTOM_DATASET="data/custom/custom_dataset.json"
if [ -f "$CUSTOM_DATASET" ]; then
    python -m src.eval.custom_eval --dataset "$CUSTOM_DATASET"
    echo "自建数据集推理完成，请打开 outputs/*-custom/predictions_for_scoring.json 进行人工打分"
else
    echo "自建数据集 ($CUSTOM_DATASET) 不存在，跳过"
    echo "请先运行: python scripts/build_custom_dataset.py 创建模板并填充数据"
fi

echo ""
echo "============================================"
echo "  全部评测完成"
echo "  结果保存在 outputs/ 目录下"
echo "============================================"
