"""ChatGPT-style PyMind-A web chat interface (Gradio)."""

import os

import gradio as gr

from pymind.chat import LocalChatEngine


ENGINE = LocalChatEngine(os.environ.get("PYMIND_CHECKPOINT", "checkpoints/tiny_gpt.pt"))


def respond(message, history, temperature, top_k):
    if not message or not message.strip():
        return "", history
    try:
        answer = ENGINE.reply(
            message,
            history=history,
            temperature=temperature,
            top_k=int(top_k),
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        answer = f"**PyMind setup issue:** {exc}"
    updated = list(history or [])
    updated.append({"role": "user", "content": message})
    updated.append({"role": "assistant", "content": answer})
    return "", updated


with gr.Blocks(title="PyMind-A", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# 🧠 PyMind-A\n"
        "Your Python-first AI chat workspace.\n\n"
        "> Early prototype: replies come from your locally trained TinyGPT checkpoint. "
        "It is not yet comparable to ChatGPT."
    )
    chatbot = gr.Chatbot(label="Conversation", type="messages", height=520)
    with gr.Row():
        message = gr.Textbox(
            placeholder="Message PyMind... / PyMind-க்கு message அனுப்பு",
            show_label=False,
            scale=8,
            lines=2,
        )
        send = gr.Button("Send", variant="primary", scale=1)
    with gr.Row():
        temperature = gr.Slider(0.2, 1.5, value=0.8, step=0.1, label="Creativity")
        top_k = gr.Slider(0, 100, value=40, step=1, label="Top-K (0 = off)")
    clear = gr.Button("＋ New chat")

    send.click(
        respond,
        inputs=[message, chatbot, temperature, top_k],
        outputs=[message, chatbot],
    )
    message.submit(
        respond,
        inputs=[message, chatbot, temperature, top_k],
        outputs=[message, chatbot],
    )
    clear.click(lambda: ([], ""), outputs=[chatbot, message])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")))
