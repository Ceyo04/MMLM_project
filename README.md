# 基于 VLM 的智能图文问答助手

多模态大模型原理与应用课程 · 期末大作业

基于 Qwen2.5-VL-3B 构建的多模态问答助手，支持自然场景图片和文档/幻灯片截图两类图像的中文问答与多轮对话。

## 功能特性

- 🖼️ 自然场景图片问答（商品图、日常照片、街景等）
- 📄 文档/幻灯片图片问答（讲义截图、PPT、表格等）
- 💬 中文多轮对话（基于同一张图片连续追问）
- 🎯 三种场景模式（自然场景 / 文档场景 / 自动识别）
- 📊 VQA-v2 定量评测（VQA 标准软匹配准确率）
- 🔍 错误分类分析与典型案例研究

## 环境配置

### 系统要求

- Python 3.10+
- GPU：NVIDIA GPU ≥8GB VRAM（RTX 4060 及以上）
- CUDA 11.8+ / 12.x
- 操作系统：Linux / WSL2

### 安装步骤

```bash
# 1. 克隆仓库
git clone <repo-url> && cd 多模态大作业

# 2. 创建虚拟环境
python3 -m venv vlm-env
source vlm-env/bin/activate

# 3. 安装依赖
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
pip install "transformers>=4.49.0" accelerate>=0.26.0
pip install "qwen-vl-utils[decord]==0.0.8"
pip install gradio==4.36.1 pillow numpy opencv-python datasets matplotlib seaborn

# 4. 下载模型
huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct --local-dir ~/models/Qwen2.5-VL-3B-Instruct
# 或使用 ModelScope（国内更快）
pip install modelscope
modelscope download --model qwen/Qwen2.5-VL-3B-Instruct --local_dir ~/models/Qwen2.5-VL-3B-Instruct

# 5. 配置模型路径
# 编辑 configs/local.yaml，设置模型路径：
# model:
#   path: "/home/yourname/models/Qwen2.5-VL-3B-Instruct"
```

## 运行说明

### 启动 Web UI

```bash
source vlm-env/bin/activate
python src/ui/app.py
# 打开浏览器访问 http://localhost:7860
# WSL2 用户：Gradio 已配置 server_name="0.0.0.0"，可直接从 Windows 浏览器访问
```

### 命令行推理

```bash
source vlm-env/bin/activate
python src/inference/generate.py --image <图片路径> --question "你的问题" --scene natural_scene
```

### 运行评测

```bash
# VQA-v2 评测（200 条样本，约 4 分钟）
source vlm-env/bin/activate
python -m src.eval.vqa_v2_eval

# TextVQA 评测（500 条样本）
python -m src.eval.textvqa_eval

# 自建中文数据集评测
python scripts/build_custom_dataset.py  # 生成模板
# 填充数据和图片后运行：
python -m src.eval.custom_eval --dataset data/custom/custom_dataset.json

# 一键运行所有评测
bash scripts/run_eval_all.sh
```

### 错误分析

```bash
python -m src.eval.error_analysis \
  --predictions outputs/<日期>/predictions.json \
  --output outputs/<日期>/error_analysis/
```

## 项目结构

```
.
├── src/
│   ├── config/settings.py      # 集中配置管理
│   ├── inference/               # 模型推理
│   │   ├── model_loader.py      # 模型加载（bfloat16）
│   │   └── generate.py          # 推理生成（API + CLI）
│   ├── ui/app.py                # Gradio Web UI
│   ├── eval/                    # 评测脚本
│   │   ├── vqa_v2_eval.py       # VQA-v2 评测
│   │   ├── textvqa_eval.py      # TextVQA 评测
│   │   ├── custom_eval.py       # 自建数据集评测
│   │   └── error_analysis.py    # 错误分类与可视化
│   └── data/                    # 流式数据加载器
├── configs/
│   ├── default.yaml             # 默认参数
│   ├── local.yaml               # 本地配置（不提交）
│   └── prompt_templates.yaml    # System Prompt 模板
├── scripts/
│   ├── build_custom_dataset.py  # 自建数据集模板
│   └── run_eval_all.sh          # 全量评测脚本
├── report/final_report.md       # 最终报告
├── slides/defense_presentation.md  # 答辩 PPT（Marp 格式）
└── demo/demo_video_script.md    # Demo 视频脚本
```

## 评测结果

| 数据集 | 样本数 | 准确率 | 备注 |
|--------|--------|--------|------|
| VQA-v2 | 50 | 78.0% | Yes/No 100%, 计数 66.7% |
| TextVQA | 100 | 72.0% | OCR/文字阅读 |
| 自建中文集 | - | 待评测 | 人工打分 1-5 |

## Demo 视频

<!-- 上传后将链接替换到这里 -->
[Demo 视频链接]（待上传）

## 致谢

- Qwen2.5-VL 模型：[QwenLM/Qwen-VL](https://github.com/QwenLM/Qwen-VL)
- VQA-v2 数据集：[Visual Question Answering](https://visualqa.org/)
- 第三方工具：HuggingFace Transformers, Gradio, PyTorch
