import os
import gradio as gr
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Set the API key for PenAI (replace with your actual API key)
os.environ["PENAI_API_KEY"] = "your_openai_api_key"

# Load the chat interface from a local endpoint with streaming enabled
demo = gr.load_chat(
    "http://localhost:1234/v1",
    model="gemma-3-1b-it",
    system_message= "Welcome to the chat interface! Ask me anything.",
    streaming=True
)

# Launch the Gradio interface with public sharing enabled
demo.launch(pwa=True, share=True)
