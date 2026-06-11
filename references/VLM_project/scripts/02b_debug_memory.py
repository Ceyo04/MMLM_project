# 保存为 02b_debug_memory.py，放在 scripts/ 下

import torch
from pathlib import Path
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "Qwen2.5-VL-7B-Instruct"

print("=" * 50)
print("显存诊断 (8-bit)")
print("=" * 50)

# 1. 加载前显存
print("\n加载前:")
print(f"  GPU 已分配: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"  GPU 缓存:   {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

# 2. 加载模型（8-bit量化）
print("\n加载模型 (8-bit)...")
quantization_config = BitsAndBytesConfig(load_in_8bit=True)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    str(MODEL_PATH),
    torch_dtype=torch.float16,
    device_map="auto",
    quantization_config=quantization_config,
)
processor = AutoProcessor.from_pretrained(str(MODEL_PATH))

print("\n加载后:")
print(f"  GPU 已分配: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"  GPU 缓存:   {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
print(f"  模型参数量: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

# 3. 推理一次后显存
print("\n进行一次推理...")
from qwen_vl_utils import process_vision_info
test_image = str(BASE_DIR / "data" / "test" / "1.png")

messages = [{"role": "user", "content": [
    {"type": "image", "image": test_image},
    {"type": "text", "text": "描述这张图片"},
]}]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
inputs = inputs.to(model.device)

with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=50, do_sample=False)

print("\n推理后:")
print(f"  GPU 已分配: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"  GPU 缓存:   {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
print(f"  GPU 峰值分配: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")