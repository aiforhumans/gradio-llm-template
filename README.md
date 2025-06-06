# Optimized LLM Chat App

This project provides a simple Gradio interface to chat with a local language model served via LM Studio. It uses asynchronous requests for improved responsiveness and caches the available model list.

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
