"""
app.py - Gradio Web UI

提供多模态图文问答的 Web 交互界面：
- 图片上传
- 中文文本输入
- 多轮对话
- 场景选择
- 模型状态显示
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import gradio as gr
import torch
from PIL import Image

from src.config.settings import get_config
from src.inference.generate import generate_single, reset_history

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class ChatSession:
    """管理单个会话的状态（图片、对话历史、场景）。"""

    def __init__(self) -> None:
        self.image: Image.Image | None = None
        self.history: list[dict] = []
        self.scene: str = "auto"
        self._cfg = get_config()

    def set_image(self, img: Image.Image | None) -> None:
        self.image = img
        # 换图时重置对话历史
        self.history = reset_history(self.scene)

    def set_scene(self, scene: str) -> None:
        self.scene = scene
        # 切换场景时重置对话历史
        self.history = reset_history(self.scene)

    def chat(self, message: str, history: list[list[str | None]]) -> tuple[list[list[str | None]], "ChatSession"]:
        """
        处理一轮对话。

        Args:
            message: 用户输入文本
            history: Gradio Chatbot 格式的历史：[[user, bot], ...]

        Returns:
            (updated_gradio_history, updated_session)
        """
        if self.image is None:
            history.append([message, "⚠️ 请先上传图片！"])
            return history, self

        try:
            answer, self.history = generate_single(
                self.image,
                message,
                history=self.history,
                scene=self.scene,
            )
            history.append([message, answer])
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            history.append([message, f"⚠️ 推理出错：{str(e)}"])

        return history, self


def get_vram_info() -> str:
    """获取当前显存使用情况。"""
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        used = total - free
        return f"模型已就绪 | 显存: {used / 1024**3:.1f} GB / {total / 1024**3:.1f} GB"
    return "模型已就绪 | 使用 CPU 推理"


def build_ui() -> gr.Blocks:
    """构建 Gradio Blocks 界面。"""
    cfg = get_config()

    css = """
    .status-bar { font-size: 0.85em; color: #666; text-align: center; padding: 8px; }
    .title { text-align: center; font-size: 1.5em; font-weight: bold; margin-bottom: 10px; }
    """

    with gr.Blocks(css=css, title="VLM 智能图文问答助手") as demo:
        gr.Markdown(
            '<div class="title">🖼️ VLM 智能图文问答助手</div>'
            f'<div style="text-align:center;color:#888;margin-bottom:16px;">模型: {cfg.model_name} | 支持自然场景 & 文档/幻灯片问答</div>'
        )

        session_state = gr.State(ChatSession())

        with gr.Row():
            # 左侧：图片上传和问题输入
            with gr.Column(scale=1):
                image_input = gr.Image(
                    type="pil",
                    label="📷 上传图片",
                    height=320,
                )
                scene_selector = gr.Radio(
                    choices=["auto", "natural_scene", "document_scene"],
                    value="auto",
                    label="🎯 场景模式",
                    info="auto=自动判断 | natural_scene=自然场景 | document_scene=文档/幻灯片",
                )
                question_input = gr.Textbox(
                    label="💬 输入你的问题",
                    placeholder="例如：图中有几个人？这张表格的第三行数据是什么？",
                    lines=2,
                )
                with gr.Row():
                    send_btn = gr.Button("🚀 发送", variant="primary")
                    clear_btn = gr.Button("🗑️ 清除对话")

            # 右侧：对话历史
            with gr.Column(scale=1):
                chatbot = gr.Chatbot(
                    label="📝 对话记录",
                    height=500,
                    show_copy_button=True,
                )

        # 状态栏
        status_bar = gr.Markdown(
            f'<div class="status-bar">⏳ 模型加载中... | 请稍候</div>'
        )

        # --- 事件绑定 ---

        def on_image_change(img: Image.Image | None, session: ChatSession) -> tuple[list, ChatSession, str]:
            """图片上传后的处理：重置对话、更新状态。"""
            session.set_image(img)
            reset_chat = []
            if img is None:
                return reset_chat, session, "⚠️ 未上传图片，请先上传"
            status = f"✅ 图片已上传 | {get_vram_info()}"
            return reset_chat, session, status

        image_input.change(
            on_image_change,
            inputs=[image_input, session_state],
            outputs=[chatbot, session_state, status_bar],
        )

        def on_scene_change(scene: str, session: ChatSession) -> tuple[list, ChatSession]:
            """场景切换后的处理。"""
            session.set_scene(scene)
            return [], session

        scene_selector.change(
            on_scene_change,
            inputs=[scene_selector, session_state],
            outputs=[chatbot, session_state],
        )

        def on_send(message: str, history: list, session: ChatSession) -> tuple[str, list, ChatSession, str]:
            """发送消息的处理函数。"""
            if not message.strip():
                return "", history, session, "⚠️ 请输入问题"
            if session.image is None:
                return "", history, session, "⚠️ 请先上传图片！"
            new_history, new_session = session.chat(message, history)
            status = get_vram_info()
            return "", new_history, new_session, status

        send_btn.click(
            on_send,
            inputs=[question_input, chatbot, session_state],
            outputs=[question_input, chatbot, session_state, status_bar],
        )
        question_input.submit(
            on_send,
            inputs=[question_input, chatbot, session_state],
            outputs=[question_input, chatbot, session_state, status_bar],
        )

        def on_clear(session: ChatSession) -> tuple[list, ChatSession, str]:
            """清除对话的处理函数。"""
            session.history = reset_history(session.scene)
            message = "✅ 对话已清除" if session.image else "⚠️ 未上传图片"
            return [], session, message

        clear_btn.click(
            on_clear,
            inputs=[session_state],
            outputs=[chatbot, session_state, status_bar],
        )

        # 页面加载时更新状态栏
        demo.load(
            lambda: get_vram_info(),
            outputs=[status_bar],
        )

    return demo


def main() -> None:
    """启动 Gradio Web UI。"""
    demo = build_ui()
    logger.info("Starting Gradio Web UI on 0.0.0.0:7860")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
