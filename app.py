import os
import time
import asyncio
import httpx
import gradio as gr
from openai import AsyncOpenAI

"""Optimized asynchronous chat interface for local LLMs via LM Studio."""

BASE_URL = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1")
API_KEY = os.getenv("LM_STUDIO_KEY", "lm-studio")

client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)

_MODELS_CACHE = {"timestamp": 0.0, "models": []}


async def fetch_models() -> list[str]:
    """Fetch available model IDs with simple caching."""
    if time.time() - _MODELS_CACHE["timestamp"] < 60:
        return _MODELS_CACHE["models"]
    try:
        async with httpx.AsyncClient(timeout=10) as http_client:
            response = await http_client.get(f"{BASE_URL}/models")
            response.raise_for_status()
            models = [m["id"] for m in response.json().get("data", [])]
    except Exception as exc:  # noqa: BLE001
        models = [f"⚠️ Error fetching models: {exc}"]
    _MODELS_CACHE["timestamp"] = time.time()
    _MODELS_CACHE["models"] = models
    return models


async def generate_chat(user_msg, history, model_id, temperature, max_tokens, system_prompt):
    """Generate a response from the model and update history."""
    if not any(msg.get("role") == "system" for msg in history):
        history.insert(0, {"role": "system", "content": system_prompt})

    history.append({"role": "user", "content": user_msg})

    try:
        response = await client.chat.completions.create(
            model=model_id,
            messages=history,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )
        assistant_msg = response.choices[0].message.content
    except Exception as exc:  # noqa: BLE001
        assistant_msg = f"❌ Error: {exc}"

    history.append({"role": "assistant", "content": assistant_msg})
    return history, history


def refresh_models():
    """Clear cache and update the dropdown choices."""
    _MODELS_CACHE["timestamp"] = 0
    return gr.update(choices=asyncio.run(fetch_models()))


def clear_chat():
    """Reset the conversation."""
    return [], []


def build_ui() -> gr.Blocks:
    """Construct the Gradio Blocks interface."""
    with gr.Blocks(title="Optimized Local Chat") as demo:
        gr.Markdown("## 🤖 Optimized Local LLM Chat")

        with gr.Row():
            with gr.Column(scale=1, min_width=260):
                with gr.Accordion("Parameters", open=True):
                    model_dropdown = gr.Dropdown(label="Model", choices=asyncio.run(fetch_models()))
                    temperature_slider = gr.Slider(0.1, 1.5, value=0.7, step=0.1, label="Temperature")
                    max_tokens_slider = gr.Slider(64, 2048, value=512, step=64, label="Max Tokens")
                    system_prompt_box = gr.Textbox(
                        label="System Prompt",
                        value="You are a helpful assistant running locally.",
                        lines=3,
                        max_lines=5,
                        interactive=True,
                        show_copy_button=True,
                    )
                    refresh_button = gr.Button("🔄 Refresh Models")
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="Chat")
                user_message = gr.Textbox(placeholder="Say something", label="Your Message")

        state = gr.State([])

        user_message.submit(
            lambda *args: asyncio.run(generate_chat(*args)),
            inputs=[user_message, state, model_dropdown, temperature_slider, max_tokens_slider, system_prompt_box],
            outputs=[chatbot, state],
        )
        refresh_button.click(refresh_models, None, model_dropdown)
        refresh_button.click(clear_chat, None, [chatbot, state])

    return demo


if __name__ == "__main__":
    build_ui().launch()
