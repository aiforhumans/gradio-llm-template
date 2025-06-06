# Simple LLM Chat App

This project provides a minimal Gradio interface to chat with a local language model served via LM Studio.

## Setup

1. **Create a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the app**
   ```bash
   python app.py
   ```

The LM Studio server URL can be configured with the `LM_STUDIO_URL` environment variable. The default is `http://localhost:1234/v1`.

## Using an OpenAI-compatible endpoint

If you have a chat server that exposes an OpenAI API (for example Ollama), you can run a chat interface in a single line. Make sure the `openai` package is installed and then start the interface with `gr.load_chat`:

```python
import gradio as gr

gr.load_chat("http://localhost:11434/v1/", model="llama3.2", token="***").launch()
```
