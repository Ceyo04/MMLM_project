"""
基础推理测试：验证模型加载和单张图片问答
"""
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ============ 1. 加载模型 ============
print("正在加载模型...")
model_path = "../models/Qwen2.5-VL-7B-Instruct"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
)

processor = AutoProcessor.from_pretrained(
    model_path,
    trust_remote_code=True,
)
print("模型加载完成！显存占用约", torch.cuda.max_memory_allocated() / 1024**3, "GB")

# ============ 2. 准备测试 ============
# 先用你的测试图片路径
image_path = "../data/test/1.png"  # 改成你实际的图片路径
question = "请详细描述这张图片的内容。"

# 构建消息
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": question},
        ],
    }
]

# ============ 3. 推理 ============
print(f"\n用户问题: {question}")

# 处理输入
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)

inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
).to(model.device)

# 生成回答
with torch.no_grad():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True,
    )

# 解码（去掉输入部分）
generated_ids_trimmed = [
    out_ids[len(in_ids):]
    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]

response = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)[0]

print(f"模型回答: {response}")