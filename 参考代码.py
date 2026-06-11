"""
Gradio Web UI - 支持多轮对话的图文问答助手（最终修正版）
"""
import gradio as gr
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
import os
import time

class VLMChatbot:
    def __init__(self, model_path="../models/Qwen2.5-VL-7B-Instruct"):
        print("正在加载模型（8-bit量化）...")
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            quantization_config=quantization_config,
        )
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        print(f"模型加载完成！显存占用: {torch.cuda.max_memory_allocated() / 1024**3:.1f} GB")
        
    def predict(self, image, history, user_text):
        
        messages = []
        for msg in history:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        current_content = []
        if image is not None:
            # 限制图片最大边长，大幅降低视觉编码开销
            w, h = image.size
            max_side = max(w, h)
            if max_side > 448:
                scale = 448 / max_side
                image = image.resize((int(w * scale), int(h * scale)))
            temp_path = "temp_image.png"
            image.save(temp_path)
            current_content.append({"type": "image", "image": temp_path})
        
        current_content.append({"type": "text", "text": user_text})
        messages.append({"role": "user", "content": current_content})
        
        try:
            t0 = time.time()
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.model.device)
            t1 = time.time()
            print(f"预处理耗时: {t1 - t0:.2f}s")
            print(f"模型设备: {self.model.device}")
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                )
            t2 = time.time()
            print(f"模型推理耗时: {t2 - t1:.2f}s")
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            response = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            
            if image is not None and os.path.exists("temp_image.png"):
                os.remove("temp_image.png")
            
            torch.cuda.empty_cache()
            return response
            
        except Exception as e:
            return f"生成回复时出错: {str(e)}"

# 创建界面
with gr.Blocks(title="VLM智能图文问答助手") as demo:
    gr.Markdown("""
    # 🖼️ VLM智能图文问答助手
    ### 基于Qwen2.5-VL-7B的多模态对话系统
    """)
    
    chatbot_model = VLMChatbot()
    
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="上传图片")
            
        with gr.Column(scale=2):
            # 不使用type参数，但history会用字典格式管理
            chatbot = gr.Chatbot(label="对话记录", height=500)
            
            with gr.Row():
                msg = gr.Textbox(label="输入问题", placeholder="请输入问题...", scale=4)
                send_btn = gr.Button("发送", variant="primary", scale=1)
            
            clear_btn = gr.Button("清空对话")
    
    def respond(message, history, image):
        """处理用户消息"""
        if not message:
            return "", history, image
        
        # 获取模型回复
        bot_response = chatbot_model.predict(image, history, message)
        
        # 更新history - 使用字典格式
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": bot_response})
        
        return "", history, image
    
    def clear_history():
        """清空对话历史"""
        return [], None
    
    # 绑定事件
    send_btn.click(respond, [msg, chatbot, image_input], [msg, chatbot, image_input])
    msg.submit(respond, [msg, chatbot, image_input], [msg, chatbot, image_input])
    clear_btn.click(clear_history, None, [chatbot, image_input])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)