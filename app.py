import os
import httpx
import gradio as gr
from openai import OpenAI

"""Simplified chat interface for local LLMs via LM Studio."""

BASE_URL = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1")
API_KEY = os.getenv("LM_STUDIO_KEY", "lm-studio")

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)


def fetch_models() -> list[str]:
    """Return available model IDs or an error message list."""
    try:
        response = httpx.get(f"{BASE_URL}/models", timeout=5)
        response.raise_for_status()
        return [m["id"] for m in response.json().get("data", [])]
    except Exception as exc:  # noqa: BLE001
        return [f"⚠️ Error fetching models: {exc}"]


def generate_chat(
    user_msg,
    history,
    model_id,
    temperature,
    max_tokens,
    system_prompt,
    top_p,
    freq_penalty,
    pres_penalty,
    seed,
):
    """Generate a response from the model and update history."""
    if not any(msg.get("role") == "system" for msg in history):
        history.insert(0, {"role": "system", "content": system_prompt})

    history.append({"role": "user", "content": user_msg})

    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=history,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=freq_penalty,
            presence_penalty=pres_penalty,
            seed=seed,
            stream=False,
        )
        assistant_msg = response.choices[0].message.content
    except Exception as exc:  # noqa: BLE001
        assistant_msg = f"❌ Error: {exc}"

    history.append({"role": "assistant", "content": assistant_msg})
    return history, history


def regenerate_last(
    history,
    model_id,
    temperature,
    max_tokens,
    system_prompt,
    top_p,
    freq_penalty,
    pres_penalty,
    seed,
):
    """Regenerate the assistant's last response."""
    if not history:
        return history, history

    if history[-1].get("role") == "assistant":
        history.pop()

    if not any(msg.get("role") == "system" for msg in history):
        history.insert(0, {"role": "system", "content": system_prompt})

    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=history,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=freq_penalty,
            presence_penalty=pres_penalty,
            seed=seed,
            stream=False,
        )
        assistant_msg = response.choices[0].message.content
    except Exception as exc:  # noqa: BLE001
        assistant_msg = f"❌ Error: {exc}"

    history.append({"role": "assistant", "content": assistant_msg})
    return history, history


def refresh_models():
    """Update the model dropdown choices."""
    return gr.update(choices=fetch_models())


def clear_chat():
    """Reset the conversation."""
    return [], []


def sync_history(history):
    """Update internal state after a user edits messages."""
    return history, history


def build_ui() -> gr.Blocks:
    """Construct the Gradio Blocks interface."""
    with gr.Blocks(title="Optimized Local Chat") as demo:
        gr.Markdown("## 🤖 Optimized Local LLM Chat")

        with gr.Row():
            with gr.Column(scale=1, min_width=260):
                with gr.Accordion("Parameters", open=True):
                    model_dropdown = gr.Dropdown(label="Model", choices=fetch_models())
                    temperature_slider = gr.Slider(0.1, 1.5, value=0.7, step=0.1, label="Temperature")
                    max_tokens_slider = gr.Slider(64, 2048, value=512, step=64, label="Max Tokens")
                    top_p_slider = gr.Slider(0.0, 1.0, value=1.0, step=0.05, label="Top P")
                    freq_penalty_slider = gr.Slider(-2.0, 2.0, value=0.0, step=0.1, label="Frequency Penalty")
                    pres_penalty_slider = gr.Slider(-2.0, 2.0, value=0.0, step=0.1, label="Presence Penalty")
                    seed_number = gr.Number(value=0, precision=0, label="Seed")
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
                chatbot = gr.Chatbot(label="Chat", type="messages", resizable=True, editable="all")
                user_message = gr.Textbox(placeholder="Say something", label="Your Message", submit_btn="Send")
                regenerate_button = gr.Button("🔁 Regenerate")

        state = gr.State([])

        user_message.submit(
            generate_chat,
            inputs=[
                user_message,
                state,
                model_dropdown,
                temperature_slider,
                max_tokens_slider,
                system_prompt_box,
                top_p_slider,
                freq_penalty_slider,
                pres_penalty_slider,
                seed_number,
            ],
            outputs=[chatbot, state],
        )
        regenerate_button.click(
            regenerate_last,
            inputs=[
                state,
                model_dropdown,
                temperature_slider,
                max_tokens_slider,
                system_prompt_box,
                top_p_slider,
                freq_penalty_slider,
                pres_penalty_slider,
                seed_number,
            ],
            outputs=[chatbot, state],
        )
        chatbot.change(sync_history, chatbot, state)
        refresh_button.click(refresh_models, None, model_dropdown)
        refresh_button.click(clear_chat, None, [chatbot, state])

    return demo


if __name__ == "__main__":
    build_ui().launch()
