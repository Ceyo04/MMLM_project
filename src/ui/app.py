"""
app.py - Gradio Web UI

基于 Qwen2.5-VL 的多模态图文问答助手。
参考 Gradio 6.x 官方 Chatbot 用法：history 为 [{"role":"user","content":...}, ...] 格式。
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import gradio as gr
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info

from src.config.settings import get_config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class VLMChatbot:
    """封装模型加载与推理。"""

    def __init__(self) -> None:
        cfg = get_config()
        logger.info(f"Loading model from {cfg.model_path}...")

        self.model = self._load_model(cfg)
        self.processor = self._load_processor(cfg)
        self.cfg = cfg

        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            used = total - free
            logger.info(f"Model loaded. VRAM: {used / 1024**3:.1f} GB / {total / 1024**3:.1f} GB")

    @staticmethod
    def _load_model(cfg):  # noqa: ANN205
        from transformers import Qwen2_5_VLForConditionalGeneration
        return Qwen2_5_VLForConditionalGeneration.from_pretrained(
            cfg.model_path,
            torch_dtype=torch.bfloat16,
            device_map=cfg.device,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        ).eval()

    @staticmethod
    def _load_processor(cfg):  # noqa: ANN205
        from transformers import AutoProcessor
        return AutoProcessor.from_pretrained(
            cfg.model_path,
            trust_remote_code=True,
            min_pixels=cfg.min_pixels_actual,
            max_pixels=cfg.max_pixels_actual,
        )

    def predict(self, image: Image.Image | None, history: list[dict], user_text: str) -> str:
        """
        根据图片、对话历史和用户输入生成回答。

        Args:
            image: 用户上传的 PIL Image（可为 None）
            history: [{"role":"user","content":...}, {"role":"assistant","content":...}, ...]
            user_text: 当前轮用户问题文本

        Returns:
            模型回答字符串
        """
        # 从 history 重建 messages（用于多轮对话上下文）
        messages: list[dict] = []
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})

        # 构造当前轮用户消息
        current_content: list[dict] = []
        if image is not None:
            current_content.append({"type": "image", "image": image})
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
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            t1 = time.time()
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.cfg.max_new_tokens,
                    do_sample=self.cfg.do_sample,
                    temperature=self.cfg.temperature if self.cfg.do_sample else None,
                    top_p=self.cfg.top_p if self.cfg.do_sample else None,
                    repetition_penalty=self.cfg.repetition_penalty,
                )
            t2 = time.time()

            generated_ids_trimmed = [
                out_ids[len(in_ids):] for out_ids, in_ids in zip(generated_ids, inputs["input_ids"])
            ]
            response = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            logger.info(f"Inference: preprocess={t1-t0:.2f}s, generate={t2-t1:.2f}s, total={t2-t0:.2f}s")
            torch.cuda.empty_cache()
            return response

        except Exception as e:
            logger.error(f"Inference error: {e}")
            return f"⚠️ 推理出错：{str(e)}"


def build_ui() -> gr.Blocks:
    """构建 Gradio Blocks 界面。"""
    cfg = get_config()
    chatbot_model = VLMChatbot()

    with gr.Blocks(title="VLM 智能图文问答助手") as demo:
        gr.Markdown(f"""
        # 🖼️ VLM 智能图文问答助手
        ### 基于 {cfg.model_name} 的多模态对话系统
        """)

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="📷 上传图片")

            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="📝 对话记录", height=500)

                with gr.Row():
                    msg = gr.Textbox(
                        label="💬 输入问题",
                        placeholder="例如：图中有几个人？这是什么颜色？",
                        scale=4,
                    )
                    send_btn = gr.Button("🚀 发送", variant="primary", scale=1)

                clear_btn = gr.Button("🗑️ 清空对话")

        def respond(message: str, history: list[dict], image: Image.Image | None) -> tuple[str, list[dict], Image.Image | None]:
            """处理用户消息。"""
            if not message.strip():
                return "", history, image

            bot_response = chatbot_model.predict(image, history, message)

            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": bot_response})

            return "", history, image

        def clear_history() -> tuple[list, None]:
            """清空对话历史。"""
            return [], None

        # 事件绑定
        send_btn.click(respond, [msg, chatbot, image_input], [msg, chatbot, image_input])
        msg.submit(respond, [msg, chatbot, image_input], [msg, chatbot, image_input])
        clear_btn.click(clear_history, None, [chatbot, image_input])

    return demo


def main() -> None:
    """启动 Gradio Web UI。"""
    demo = build_ui()
    logger.info("Starting Gradio Web UI on 0.0.0.0:7860")
    try:
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=False,
        )
    except ValueError as e:
        if "localhost is not accessible" in str(e):
            logger.warning("WSL2: localhost check failed, retrying with share=True")
            demo.launch(server_name="0.0.0.0", server_port=7860, share=True, show_error=False)
        else:
            raise


if __name__ == "__main__":
    main()
