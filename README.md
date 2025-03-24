---
title: test
app_file: app.py
sdk: gradio
sdk_version: 5.22.0
---
# 🧠 Gradio LLM Local Project

This project uses Gradio to build local user interfaces for LLMs like those run in LM Studio or Ollama.

---

## 🛠️ Setup Instructions

### 1. Clone the project

```bash
git clone https://github.com/yourusername/gradio-llm-project.git
cd gradio-llm-project
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the virtual environment

- **Windows:**
  ```bash
  .venv\Scripts\activate
  ```
- **Mac/Linux:**
  ```bash
  source .venv/bin/activate
  ```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Gradio app

```bash
python app.py
```

---

## ✅ Notes

- Keep `.venv/` out of version control
- Add new packages with `pip install <package>` and update `requirements.txt` with `pip freeze > requirements.txt`
