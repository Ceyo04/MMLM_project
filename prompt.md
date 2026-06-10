# 多模态大作业 主控提示词

## 身份声明（必读）

你当前通过 **Claude Code CLI** 运行，底层大模型为 DeepSeek V4 Pro。你是具备本地文件系统读写和 Bash 执行权限的 **单一 AI 智能体**。你不需要、也不应该输出"指导 XXX 执行"之类的文本——所有代码编写、命令执行、文件操作全部由你**亲自调用工具**完成。

---

## 一、任务定义

### 1.1 题目

**基于 VLM 的智能图文问答助手**

### 1.2 核心目标

构建一个"看图/看文档能聊天"的多模态助手，支持用户上传图片并输入文本问题，系统返回准确的自然语言回答，且支持多轮对话。

### 1.3 场景要求

系统必须覆盖至少两类图像场景：

| 场景类别 | 示例 | 典型问题类型 |
|---------|------|-------------|
| **自然场景** | 商品图、日常照片、街景 | "图中有几个苹果？"、"这是什么品牌？" |
| **文档/幻灯片** | 讲义截图、PPT 截图、表格 | "第3页的结论是什么？"、"表格中最大值是哪个？" |

### 1.4 最低验收标准（必须全部满足）

1. 支持上传图片 + 文本问题，返回多轮对话式回答
2. 实现 Web UI（Gradio）交互界面
3. 在至少一个公开数据集子集上评测，报告准确率或人工打分结果
4. 提供 5-10 个典型成功/失败案例分析

---

## 二、技术方案

### 2.1 模型选择

**主模型：Qwen2.5-VL-7B-Instruct**

- 开源、可本地部署、中文能力强
- 支持图像+文本多模态输入
- 在 VQA、TextVQA、文档理解等任务上表现优秀
- 7B 规模可在单张 24GB GPU（RTX 3090/4090）上以 bfloat16 运行

**备选方案（任一均可，若主模型不可用）：**
- LLaMA 3.2 Vision（11B，需 24GB+）
- Gemma 3-Vision
- GLM-4.6V-Flash（API 调用）

### 2.2 推理框架

```
# Qwen2.5-VL 要求 transformers >= 4.49.0（4.45.0 不含 Qwen2_5_VL 模型定义，会报 KeyError: 'qwen2_5_vl'）
transformers >= 4.49.0
torch >= 2.3.0
qwen-vl-utils[decord]==0.0.8   # Qwen 官方工具库
accelerate >= 0.26.0            # 混合精度推理/设备映射
```

不使用 vLLM（Qwen2.5-VL 的 vLLM 支持取决于版本兼容性，优先使用 transformers 原生推理以降低集成风险）。

### 2.3 界面框架

**Gradio 4.x** — 理由：
- 原生支持 multimodal 输入（Image + Textbox）
- 内置 ChatInterface 组件，可直接构建多轮对话
- 一行 `demo.launch()` 即可部署

**WSL2 用户必须注意**（见 2.6 节）。

### 2.4 推理精度

**统一使用 `torch.bfloat16` 加载模型。**

在 RTX 3090/4090（24GB）上，Qwen2.5-VL-7B 以 bfloat16 加载约占用 15-16GB 显存，安全裕度充足。**不使用 bitsandbytes 4-bit 量化**——在当前 transformers 版本下，4-bit 量化作用于视觉编码器存在已知精度雪崩 Bug，会导致中文输出乱码，且 24GB 显存完全不需要量化。

### 2.5 微调方案（可选，加分项）

若算力允许，使用 **LoRA（PEFT）** 在自建中文图文问答数据上微调：

```
peft >= 0.11.0
```

训练策略：
- 冻结视觉编码器（ViT），仅微调 LLM 部分的 Q/K/V 投影层
- rank=16, alpha=32, target_modules=["q_proj","k_proj","v_proj","o_proj"]
- 学习率 2e-4，batch_size=2，gradient_accumulation_steps=4
- 训练 1-2 epoch，使用 bfloat16 混合精度，在单张 24GB GPU 上完成

### 2.6 WSL2 环境特别注意事项

当前工作目录位于 `/mnt/d/`（Windows DrvFs 挂载），存在以下风险，必须处理：

**I/O 性能：** 通过 `/mnt/d/` 访问 Windows 文件系统的 I/O 显著慢于 WSL2 原生 ext4（`~/`）。模型权重（~15GB）的首次加载会较慢，但这是一次性开销，可接受。图片数据集的频繁小文件读写应避免落盘。

**Gradio 网络可达性：** WSL2 默认 NAT 模式下，服务绑定 `127.0.0.1` 会导致 Windows 宿主机浏览器无法访问。**所有 Gradio 启动必须使用：**
```python
demo.launch(server_name="0.0.0.0", server_port=7860)
```
然后通过 Windows 浏览器访问 `http://localhost:7860`。

**VHDX 磁盘膨胀：** WSL2 虚拟磁盘文件（.vhdx）写入大文件后会膨胀，且删除文件后不会自动回收空间。因此评测数据集**禁止下载原始图片落盘**，必须使用流式加载（见阶段三）。

### 2.7 评测数据集

| 数据集 | 用途 | 指标 | 规模（子集） |
|--------|------|------|-------------|
| VQA-v2 | 自然场景通用 VQA | Accuracy | 1000-2000 条验证集 |
| TextVQA | 图像中文字理解 | Accuracy | 1000 条验证集 |
| 自建中文集 | 文档/幻灯片场景 | 人工打分(1-5) | 50-100 条 |

---

## 三、目录结构

```
多模态大作业/
├── CLAUDE.md                    # 项目规范
├── prompt.md                    # 本文件
├── README.md                    # 项目说明（含环境配置、运行说明）
├── .gitignore                   # 忽略 data/ models/ outputs/ *.env local.yaml
│
├── src/
│   ├── __init__.py
│   │
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py          # 集中管理所有路径、超参、Prompt 模板
│   │
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── model_loader.py      # 模型加载（bfloat16 精度）
│   │   ├── processor.py         # 图像预处理 + 文本 tokenize
│   │   └── generate.py          # 推理生成（单轮 + 多轮）
│   │
│   ├── ui/
│   │   ├── __init__.py
│   │   └── app.py               # Gradio Web UI（多轮对话、历史管理）
│   │
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── vqa_v2_eval.py       # VQA-v2 评测
│   │   ├── textvqa_eval.py      # TextVQA 评测
│   │   └── custom_eval.py       # 自建数据集人工打分辅助工具
│   │
│   └── data/
│       ├── __init__.py
│       ├── vqa_v2_loader.py     # VQA-v2 流式数据加载
│       ├── textvqa_loader.py    # TextVQA 流式数据加载
│       └── custom_dataset.py    # 自建数据集构造脚本
│
├── configs/
│   ├── default.yaml             # 默认配置（模型名、max_tokens、temperature）
│   ├── local.yaml               # 本地配置（模型路径、设备），不提交 git
│   └── prompt_templates.yaml    # 系统提示词模板（中文）
│
├── scripts/
│   ├── build_custom_dataset.py  # 自建中文数据集生成脚本
│   └── run_eval_all.sh          # 一键执行全部评测
│
├── outputs/                     # 不进 git
│   └── YYYY-MM-DD-<描述>/       # 每次实验独立目录
│       ├── results.json
│       ├── error_analysis/
│       └── figures/
│
├── report/
│   ├── final_report.md          # 最终报告 Markdown 源文件
│   ├── references.bib           # 参考文献
│   └── figures/                 # 报告插图（架构图、实验结果图、案例分析图）
│
├── slides/
│   └── defense_presentation.md  # 答辩演示文稿（Marp / Reveal.js 格式）
│
└── demo/
    └── demo_video_script.md     # Demo 视频脚本（3-5 分钟）
```

---

## 四、实施步骤

> **【强制协议】每完成一个阶段并自行运行验证通过后，必须停止调用任何工具，输出以下格式的等待指令：**
> ```
> ✅ 阶段 X 已完成。验证结果：[简述]。
> 请回复「进入下一阶段」以继续。
> ```
> **未收到用户确认前，严禁提前生成下一阶段的代码或文件。**

---

### 阶段零：环境搭建

**步骤 0.1** 创建 conda 环境
```bash
conda create -n vlm-qa python=3.10 -y
conda activate vlm-qa
```

**步骤 0.2** 安装依赖
```bash
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
pip install "transformers>=4.49.0" accelerate>=0.26.0
pip install "qwen-vl-utils[decord]==0.0.8"
pip install gradio==4.36.1 pillow numpy opencv-python
pip install datasets  # HuggingFace datasets，用于流式加载
pip install matplotlib seaborn pandas  # 可视化
# 仅当决定做 LoRA 微调时安装：
pip install peft
```

**步骤 0.3** 下载模型

由于 `/mnt/d/` 跨文件系统 I/O 较慢，模型权重放在 WSL2 原生 ext4 文件系统内（`~/.cache/huggingface/` 或 `~/models/`），不要放在项目目录下的 `./models/` 中。然后在 `configs/local.yaml` 中配置路径。

```bash
# 使用 modelscope（国内下载更快）
pip install modelscope
modelscope download --model qwen/Qwen2.5-VL-7B-Instruct --local_dir ~/models/Qwen2.5-VL-7B-Instruct

# 或使用 huggingface-cli
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir ~/models/Qwen2.5-VL-7B-Instruct
```

**步骤 0.4** 验证环境
```bash
# 验证 transformers 版本与 Qwen2.5-VL 支持
python -c "
import transformers
print(f'transformers: {transformers.__version__}')
from transformers import Qwen2_5_VLForConditionalGeneration
print('Qwen2_5_VLForConditionalGeneration import OK')
"
```

**阶段零验证标准：** 上述 import 无报错；`nvidia-smi` 显示 ≥20GB 可用显存。

---

### 阶段一：核心推理管线

**目标：** 加载模型 → 输入图片+问题 → 输出回答。整个系统的基石。

**步骤 1.1** 实现 `src/config/settings.py`
- 定义 `Config` dataclass：模型路径、设备、max_tokens、temperature、top_p
- 从 `configs/default.yaml` 加载默认值，从 `configs/local.yaml` 覆盖本地值
- 加载 `configs/prompt_templates.yaml` 中的 System Prompt 模板

**步骤 1.2** 实现 `src/inference/model_loader.py`
- 封装模型加载逻辑，使用 `torch.bfloat16` 精度
- 调用 `Qwen2_5_VLForConditionalGeneration.from_pretrained()` 和 `AutoProcessor.from_pretrained()`
- 返回 `(model, processor)` 元组

**步骤 1.3** 实现 `src/inference/processor.py`
- 图像预处理：resize、normalize（使用 `qwen_vl_utils` 工具函数）
- 文本 tokenize + chat template 应用
- 构建符合 Qwen2.5-VL 格式的 messages 列表

**步骤 1.4** 实现 `src/inference/generate.py`
- 提供命令行接口和 Python API 两种调用方式
- `generate_single(image, question, history=None) -> tuple[str, list]`
- `generate_multi(image, question, history) -> tuple[str, list]`
- 命令行用法：
  ```bash
  python src/inference/generate.py --image data/test/sample.jpg --question "这是什么？"
  ```

**步骤 1.5** 准备测试图片
- 在 `data/test/` 下放 2-3 张测试图片（自然场景 1 张 + 文档/幻灯片 1 张）

**阶段一验证标准：** 命令行执行推理，对自然图片和文档图片均输出有意义的中文回答。显存占用 <20GB。

---

### 阶段二：交互界面

**目标：** 用 Gradio 构建 Web UI，支持多轮对话、图片上传、回答展示、历史管理。

**步骤 2.1** 实现 `src/ui/app.py`

Gradio Blocks 布局设计：

```
┌──────────────────────────────────────────┐
│           VLM 智能图文问答助手            │
├────────────────────┬─────────────────────┤
│                    │                     │
│   图片上传区域      │   对话历史区域       │
│   (gr.Image)       │   (gr.Chatbot)      │
│                    │                     │
│   问：_____________│                     │
│    [发送] [清除]   │                     │
│                    │                     │
├────────────────────┴─────────────────────┤
│  场景选择：[自然场景] [文档/幻灯片] [自动] │
│  状态栏：模型已就绪 | 显存占用: xx GB     │
└──────────────────────────────────────────┘
```

核心功能点：
- 图片上传后保持到会话状态（`gr.State`）
- 问题输入支持中文
- "清除对话"按钮重置历史
- 场景选择器影响 System Prompt 模板（自然场景 vs 文档场景 vs 自动）
- 底部状态栏显示模型加载状态和显存占用

**步骤 2.2** 多轮对话实现
- 使用 `gr.Chatbot` 组件，每个消息为 `(user_msg, bot_msg)` 元组
- history 维护在 `gr.State` 中，每轮追加新问答对
- 图片在首轮上传后保持不变，后续轮次复用

**步骤 2.3** 启动命令

Gradio 启动代码必须包含：

```python
demo.launch(server_name="0.0.0.0", server_port=7860)
```

然后通过 Windows 浏览器访问 `http://localhost:7860`。

**阶段二验证标准：** 浏览器打开 Gradio 页面，上传图片 → 输入问题 → 得到回答 → 追问 → 多轮对话正常。

---

### 阶段三：数据集与评测

**目标：** 在 VQA-v2、TextVQA 子集上评测模型，并自建中文数据集。

> **【数据存储强制约束】公开数据集评测严禁下载原始图片到磁盘。必须使用 HuggingFace `datasets` 库的 `streaming=True` 模式，图片在内存中解码为 PIL 对象后直接送入模型，评测完即丢弃。这是为了防止 WSL2 虚拟磁盘（.vhdx）因写入数 GB 图片后无法回收空间而永久膨胀。**

**步骤 3.1** VQA-v2 评测

- 数据加载：`src/data/vqa_v2_loader.py`
  ```python
  from datasets import load_dataset
  # 流式加载，不落盘
  dataset = load_dataset("HuggingFaceM4/VQAv2", split="validation", streaming=True)
  # 取前 1000-2000 条，在内存中处理
  ```
- 评测：`src/eval/vqa_v2_eval.py`
  - 逐条处理：从 URL/bytes 加载图片 → 构造问题 → 模型生成 → 与标准答案比对
  - VQA-v2 软匹配规则：预测答案与 10 个标注答案中 ≥3 个一致算正确
  - 输出：Overall Accuracy、Per-category Accuracy、Yes/No Accuracy

**步骤 3.2** TextVQA 评测

- 数据加载：`src/data/textvqa_loader.py`
  ```python
  dataset = load_dataset("textvqa", split="validation", streaming=True)
  # 取前 1000 条
  ```
- 评测方式同 VQA-v2，额外关注 OCR 相关问题的准确率

**步骤 3.3** 自建中文图文问答集

- `scripts/build_custom_dataset.py` 和 `src/data/custom_dataset.py`
- 数据来源：手动收集 30-50 张中文文档/幻灯片截图（可自行截图或拍照）
- 针对每张图片设计 1-3 个中文问题，标注标准答案
- 数据格式：
  ```json
  {
    "image": "custom_001.jpg",
    "category": "document",
    "qa_pairs": [
      {"question": "这份文档的标题是什么？", "answer": "2024年度工作总结"},
      {"question": "第3点提到的关键指标是多少？", "answer": "增长15%"}
    ]
  }
  ```
- 规模：≥50 条问答对
- 自建数据集的图片可以落盘（规模小，总共几十张）

**步骤 3.4** 人工打分评测工具

- `src/eval/custom_eval.py`
- 功能：逐条展示图片、问题、模型回答、标准答案
- 打分：1-5 分（1=完全错误，5=完全正确）
- 输出：平均分、标准差、标记典型样本（用于后续案例分析）

**步骤 3.5** 评测结果汇总

- `scripts/run_eval_all.sh` 一键运行所有评测
- 结果输出到 `outputs/<日期>/results.json`
- 自动生成 Markdown 格式评测报告

**阶段三验证标准：**
- VQA-v2 子集（≥1000条）准确率 >50%
- TextVQA 子集（≥500条）准确率 >40%
- 自建数据集（≥50条）人工打分均值 >3.0/5.0

---

### 阶段四：错误分析与案例研究

**目标：** 深入分析模型表现，为最终报告提供素材。

**步骤 4.1** 错误分类框架

将错误回答分入以下类别：

| 错误类型 | 定义 | 示例 |
|---------|------|------|
| 视觉理解错误 | 模型未正确识别图中物体/文字 | 把"苹果"识别成"橙子" |
| OCR 错误 | 图中文字识别错误或遗漏 | 表格数字读错 |
| 推理错误 | 识别正确但逻辑推理错误 | 能数出物体但关系判断错 |
| 知识缺失 | 需要外部知识但模型不具备 | 不知道某个品牌 logo 含义 |
| 语言表达 | 回答含糊、不完整或答非所问 | 信息正确但表达混乱 |

**步骤 4.2** 产出物

- `outputs/<日期>/error_analysis/` 目录：
  - 每个错误案例一个 JSON 条目：`{image_ref, question, predicted, ground_truth, error_type, notes}`
  - 错误类型分布图（饼图/柱状图，用 matplotlib 生成）
  - 错误类型 × 数据集交叉分析

**步骤 4.3** 成功案例分析

- 挑选 5-7 个模型回答准确的案例，覆盖不同场景
- 每个案例：原图 + 问题 + 模型回答 + 标准答案 + 简短分析（为什么成功）

**步骤 4.4** 失败案例分析

- 挑选 5-7 个模型回答错误的案例
- 每个案例：同成功案例格式 + 错误分类 + 改进建议

**阶段四验证标准：** 两类案例各 ≥5 个，错误分布图生成成功。

---

### 阶段五：最终报告撰写

**目标：** 撰写 15-20 页最终报告。

**报告格式：** 使用 Markdown 撰写 `report/final_report.md`。这是 Agent 可直接生成的格式。

生成 PDF 的方式（按优先级尝试）：
1. 若有 `pandoc`：`pandoc report/final_report.md -o report/final_report.pdf --pdf-engine=xelatex -C --metadata link-citations=true`
2. 若 pandoc 不可用但有 Python：使用 `weasyprint` 或 `md2pdf` 转换
3. 若以上均不可用：保留 `final_report.md` 文件，**不要反复尝试安装 texlive（约需数 GB 且耗时极长）**，告知用户手动转换

**报告结构（严格按此大纲）：**

```
1. 摘要（半页）

2. 引言（1-2 页）
   2.1 任务背景：视觉问答的应用价值
   2.2 任务定义：给定图片+问题，输出回答
   2.3 挑战：多模态对齐、中文问答、多场景泛化
   2.4 本文工作概览

3. 相关工作（2-3 页）
   至少引用 5 篇代表性工作，分类讨论：
   3.1 视觉-语言预训练模型（CLIP, BLIP-2）
   3.2 开源多模态对话模型（LLaVA, MiniGPT-4, Qwen-VL）
   3.3 视觉问答数据集与评测方法（VQA-v2, TextVQA）
   3.4 参数高效微调（LoRA, QLoRA）

4. 系统设计（2-3 页）
   4.1 系统架构图（必须：用 Mermaid 或 ASCII art 画一张清晰的架构图）
   4.2 模型选择与理由
   4.3 推理管线设计
   4.4 Web UI 设计
   4.5 Prompt 工程（System Prompt 设计）

5. 实验（3-4 页）
   5.1 实验设置（数据集、评测指标、计算资源）
   5.2 VQA-v2 评测结果（表格 + 简要分析）
   5.3 TextVQA 评测结果（表格 + 简要分析）
   5.4 自建中文数据集评测（人工打分统计 + 分析）
   5.5 消融实验（可选加分项：不同 Prompt 对比、不同参数对比）

6. 案例分析（2-3 页）
   6.1 成功案例展示（3-5 个）
   6.2 失败案例分析与错误分类（3-5 个）
   6.3 错误类型分布统计

7. 反思与展望（1-2 页）
   7.1 模型局限
   7.2 改进方向
   7.3 AGI 视角下的多模态理解：潜力与瓶颈

8. 结论（半页）

9. 参考文献
   使用 GB/T 7714 格式（中文论文标准）
```

**第 3 节最低引用清单：**
1. Radford et al., "Learning Transferable Visual Models From Natural Language Supervision", ICML 2021. (CLIP)
2. Li et al., "BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models", ICML 2023.
3. Liu et al., "Visual Instruction Tuning", NeurIPS 2023. (LLaVA)
4. Zhu et al., "MiniGPT-4: Enhancing Vision-Language Understanding with Advanced Large Language Models", 2023.
5. Bai et al., "Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond", 2024.
6. Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022.
7. Goyal et al., "Making the V in VQA Matter: Elevating the Role of Image Understanding in Visual Question Answering", CVPR 2017. (VQA-v2)
8. Singh et al., "Towards VQA Models That Can Read", CVPR 2019. (TextVQA)

**阶段五验证标准：** `report/final_report.md` 包含全部 9 个章节，≥15 页等量内容，8 篇引用格式正确。

---

### 阶段六：答辩演示文稿与 Demo 视频

**目标：** 制作答辩演示文稿和 Demo 视频脚本。

**演示文稿格式：** 使用 Markdown 撰写 `slides/defense_presentation.md`，采用 Marp 或 Reveal.js 格式。

Marp 格式示例：
```markdown
---
marp: true
theme: default
---

# 基于 VLM 的智能图文问答助手

姓名 | 学号 | 计算机科学与技术

---

## 任务定义

- 构建多模态图文问答助手
- 支持自然场景 + 文档/幻灯片两类图像
- 中文问答 + 多轮对话
```

**PPT 原则（参考 CLAUDE.md 约定）：**
- 写精简要点，不照搬报告全文
- 每页 3-5 个要点，配合图表
- 控制节奏：15 页以内，5 分钟汇报节奏

**PPT 大纲（12-15 页）：**
1. 封面（题目、姓名、学号）
2. 任务定义
3. 系统架构图（核心页）
4. 技术选型与理由
5. Prompt 设计
6-7. Web UI Demo 截图与说明
8. 实验设计
9-10. 实验结果（数据表格/柱状图）
11. 成功案例展示
12. 失败案例与错误分析
13. 反思与展望
14. 总结与致谢

**Demo 视频脚本（3-5 分钟）：**
- 脚本文件：`demo/demo_video_script.md`
- 录制流程：
  1. 打开 Web UI（10s）
  2. 自然场景问答演示（60s）：上传商品图/日常照片，2-3 个问题，多轮对话
  3. 文档场景问答演示（60s）：上传幻灯片截图，文档相关问题，推理能力
  4. 成功/失败案例对比（60s）
  5. 总结（30s）

**阶段六验证标准：** 演示文稿 Markdown 文件完整，Demo 脚本覆盖全部 5 个环节，总时长估算合理。

---

## 五、工程规范

### 5.1 代码规范（强制）

1. **Python 3.10+**，所有函数签名必须有类型注解：
   ```python
   def generate(image: Image.Image, question: str, history: list[dict] | None = None) -> tuple[str, list[dict]]:
       ...
   ```

2. **文件头格式：**
   ```python
   """
   module_name.py - 一句话描述
   """
   ```

3. **配置集中管理：** 硬编码路径、模型名、超参数必须定义在 `configs/` 中，代码通过 `src/config/settings.py` 读取。

4. **错误处理：** 模型推理失败不应导致 UI 崩溃，需捕获异常并返回友好提示。

5. **日志：** 使用 `logging` 模块而非 `print()`，关键节点（模型加载、推理耗时、评测进度）必须记录。

### 5.2 Git 规范

- Commit 粒度：每个功能模块完成即提交
- Commit 消息格式：
  ```
  feat: 添加 VQA-v2 评测脚本
  fix: 修复多轮对话历史拼接错误
  docs: 更新 README 运行说明
  ```

### 5.3 文档规范

- `README.md` 必须包含：
  - 项目简介
  - 环境配置步骤（从零到能跑）
  - 运行说明（如何启动 UI、跑评测）
  - 项目结构说明
  - Demo 视频链接
- 第三方代码/数据来源必须在 README 和报告中标注

### 5.4 安全规范

- `.env`、密钥、token 不进代码、不进 git
- `configs/local.yaml` 包含本地路径，在 `.gitignore` 中
- 安装新全局依赖或系统级包（如 texlive）前必须询问用户
- 不提交大文件（模型权重、数据集原始图片）

---

## 六、验收自检清单

### 6.1 系统实现（30% 权重）

- [ ] 模型以 bfloat16 正确加载并推理，无 CUDA OOM
- [ ] 支持自然场景图片输入和中文问答
- [ ] 支持文档/幻灯片图片输入和中文问答
- [ ] 多轮对话功能正常（上下文记忆）
- [ ] Web UI 可访问，上传图片 → 输入问题 → 查看回答流程顺畅
- [ ] UI 有基本的错误处理（如未上传图片时提示）
- [ ] Gradio 绑定 `0.0.0.0`，WSL2 外可访问

### 6.2 实验评测（20% 权重）

- [ ] VQA-v2 验证集子集评测完成（≥1000 条），流式加载，图片未落盘
- [ ] TextVQA 验证集子集评测完成（≥500 条），流式加载
- [ ] 自建中文数据集 ≥50 条，人工打分完成
- [ ] 至少 1-2 项定量指标（Accuracy、人工打分均值）
- [ ] 可视化：至少包含准确率柱状图和错误类型分布图

### 6.3 案例分析

- [ ] 5-7 个成功案例分析，包含原图+问题+回答+分析
- [ ] 5-7 个失败案例分析，包含原图+问题+回答+错误分类+改进建议
- [ ] 错误类型分布统计（饼图或柱状图）

### 6.4 报告与展示（30% 权重）

- [ ] 最终报告 Markdown 文件 ≥15 页等量内容
- [ ] 报告包含全部 9 个必要章节
- [ ] 参考文献 ≥8 篇，GB/T 7714 格式
- [ ] 架构图（Mermaid 或 ASCII art）
- [ ] 答辩演示文稿 Markdown 文件 12-15 页
- [ ] Demo 视频脚本完整
- [ ] README 完整

### 6.5 代码质量

- [ ] 代码有类型注解
- [ ] 目录结构符合本文约定
- [ ] 无硬编码路径（本地路径在 `local.yaml`）
- [ ] 无大文件误提交
- [ ] Commit 历史干净，消息有意义

---

## 七、常见问题与应对

### Q1：24GB 显存不够怎么办？

bfloat16 加载 Qwen2.5-VL-7B 约 15-16GB，正常情况下有 6-8GB 裕度。若仍 OOM：
1. 关闭其他占用显存的进程（`nvidia-smi` 查看）
2. 使用 `device_map="auto"` 让 accelerate 自动分配
3. 如果以上无效，切换更小的模型（Qwen2.5-VL-2B）
4. **不使用 4-bit 量化**（精度雪崩风险）

### Q2：模型生成中文回答质量差？

1. 确认 System Prompt 为中文（见 `configs/prompt_templates.yaml`）
2. 确认 `processor.apply_chat_template` 正确应用 chat template
3. 检查 temperature 和 top_p 参数，中文任务建议 temperature=0.3-0.5
4. 如仍不理想，考虑用少量中文数据做 LoRA 微调

### Q3：VQA-v2 准确率异常低？

1. 确认答案匹配逻辑正确（VQA-v2 使用 10 选 ≥3 软匹配，不是完全匹配）
2. 检查图片预处理尺寸是否与模型要求一致（Qwen2.5-VL 使用动态分辨率）
3. 确认使用了正确的 chat template

### Q4：Gradio 界面在 WSL2 中无法从浏览器访问？

1. 确认启动时使用 `server_name="0.0.0.0"` 而非默认的 `127.0.0.1`
2. 在 Windows 浏览器访问 `http://localhost:7860`（不是 WSL2 的 IP）
3. 如果仍不可达，检查 Windows 防火墙是否拦截了 7860 端口

### Q5：`transformers` 导入 Qwen2.5-VL 报 KeyError？

确认版本：`pip show transformers | grep Version`。必须 ≥4.49.0。如果已安装旧版本：
```bash
pip install --upgrade "transformers>=4.49.0"
```

### Q6：WSL2 磁盘空间不足？

1. 检查 `.vhdx` 文件大小：在 Windows PowerShell 中运行 `Get-ChildItem $env:LOCALAPPDATA\Packages\*Canonical*\LocalState\*.vhdx | Select Name, Length`
2. 如果过大，在 WSL2 内删除不需要的文件后，在 Windows PowerShell 中使用 `Optimize-VHD` 压缩
3. 这证实了为什么评测数据集必须流式加载、不能落盘

---

## 八、启动指令

开始执行时，按以下顺序逐阶段推进：

```
阶段零（环境搭建）→ 阶段一（核心推理）→ 阶段二（Web UI）→ 阶段三（评测）→ 阶段四（错误分析）→ 阶段五（报告）→ 阶段六（演示文稿+视频脚本）
```

每个阶段开始前，先确认：
- 前置阶段已完成并验证通过
- 可用的显存和磁盘空间
- 所需的依赖已安装

**每个阶段验证通过后，必须停止并等待用户确认「进入下一阶段」，才能继续。**
