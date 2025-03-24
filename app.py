import gradio as gr

def chatbot(message):
    return "You said: " + message

demo = gr.ChatInterface(fn=chatbot, title="Local LLM Chat")

if __name__ == "__main__":
    demo.launch()
