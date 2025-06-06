import os
import functools
import requests
import gradio as gr
from openai import OpenAI

"""Gradio interface for chatting with locally hosted LLMs via LM Studio."""

BASE_URL = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1")
API_KEY = os.getenv("LM_STUDIO_KEY", "lm-studio")  # dummy key for compatibility

# OpenAI client configured for the local LM Studio endpoint
client = OpenAI(base_url=BASE_URL, api_key=API_KEY)


@functools.lru_cache(maxsize=1)
def fetch_models():
    """Return a list of available model IDs from the LM Studio API."""
    try:
        response = requests.get(f"{BASE_URL}/models", timeout=10)
        response.raise_for_status()
        models = response.json().get("data", [])
        return [model["id"] for model in models]
    except Exception as exc:  # noqa: BLE001
        return [f"\u26a0\ufe0f Error fetching models: {exc}"]


def generate_chat_response(user_message, chat_history, model_id, temperature, max_tokens, system_prompt):
    """Generate a chat completion and update the chat history."""
    if not any(msg.get("role") == "system" for msg in chat_history):
        chat_history.insert(0, {"role": "system", "content": system_prompt})

    chat_history.append({"role": "user", "content": user_message})

    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=chat_history,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )
        assistant_message = response.choices[0].message.content
    except Exception as exc:  # noqa: BLE001
        assistant_message = f"\u274c Error: {exc}"

    chat_history.append({"role": "assistant", "content": assistant_message})
    return chat_history, chat_history


def clear_models():
    """Refresh the model dropdown choices."""
    fetch_models.cache_clear()
    return gr.update(choices=fetch_models())


def clear_chat():
    """Clear the conversation history and UI."""
    return [], []


def build_ui():
    """Create the Gradio UI for chatting with a local model."""
    with gr.Blocks() as demo:
        gr.Markdown("## \U0001F916 Local LM Studio Chatbot")

        with gr.Row():
            with gr.Column(scale=1, min_width=280):
                gr.Markdown("### Settings")
                with gr.Accordion("Model & Parameters", open=True):
                    model_dropdown = gr.Dropdown(
                        choices=fetch_models(),
                        label="Select Model",
                        interactive=True,
                    )
                    temperature_slider = gr.Slider(
                        minimum=0.1,
                        maximum=1.5,
                        value=0.7,
                        step=0.1,
                        label="Temperature",
                    )
                    max_tokens_slider = gr.Slider(
                        minimum=64,
                        maximum=2048,
                        value=512,
                        step=64,
                        label="Max Tokens",
                    )
                with gr.Accordion("System Prompt", open=False):
                    system_prompt_box = gr.Textbox(
                        value="You are a helpful AI assistant running locally via LM Studio.",
                        label="System Prompt",
                        lines=3,
                        max_lines=5,
                        interactive=True,
                        show_copy_button=True,
                    )
                clear_button = gr.Button("\ud83d\udd04 Refresh + Clear")

            with gr.Column(scale=3):
                chatbot = gr.Chatbot(type="messages", label="Chat with Local AI")
                user_message = gr.Textbox(
                    placeholder="Type your message here...",
                    label="Your Message",
                    submit_btn="Send",
                    autofocus=True,
                    autoscroll=True,
                )

        chat_state = gr.State([])

        user_message.submit(
            generate_chat_response,
            inputs=[user_message, chat_state, model_dropdown, temperature_slider, max_tokens_slider, system_prompt_box],
            outputs=[chatbot, chat_state],
        )
        clear_button.click(clear_models, None, model_dropdown)
        clear_button.click(clear_chat, None, [chatbot, chat_state])

    return demo


if __name__ == "__main__":
    app = build_ui()
    app.launch()
