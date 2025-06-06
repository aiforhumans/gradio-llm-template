# Simple LLM Chat App

This project provides a minimal Gradio interface to chat with a local language model served via LM Studio.  
It now supports message editing, regenerating the last response and several advanced tuning parameters.

## Setup

1. **Create a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. **Install dependencies**
   ```bash
   pip install --upgrade "gradio>=4.44.1,<5" -r requirements.txt
   ```
   This project requires **Gradio 4.x**. The command above installs the latest 4.x release.
3. **Run the app**
   ```bash
   python app.py
   ```

While chatting you can edit any message and the conversation state will update automatically. Use the "Regenerate" button to re-run the last user request. The parameter panel exposes temperature, top-p, frequency and presence penalties as well as a seed value for repeatable results.

The LM Studio server URL can be configured with the `LM_STUDIO_URL` environment variable. The default is `http://localhost:1234/v1`.

## Using an OpenAI-compatible endpoint

If you have a chat server that exposes an OpenAI API (for example Ollama), you can run a chat interface in a single line. Make sure the `openai` package is installed and then start the interface with `gr.load_chat`:

```python
import gradio as gr

gr.load_chat("http://localhost:11434/v1/", model="llama3.2", token="***").launch()
```
