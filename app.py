"""PyMind-A browser workspace: local TinyGPT chat and prompt-to-site builder."""
from __future__ import annotations

import html
import os
from pathlib import Path
import tempfile
import zipfile

import gradio as gr

from pymind.chat import LocalChatEngine
from pymind.website_builder import _slug, generate_website, write_website


ENGINE = LocalChatEngine(os.environ.get("PYMIND_CHECKPOINT", "checkpoints/tiny_gpt.pt"))


def respond(message, history, temperature, top_k):
    if not message or not message.strip():
        return "", history
    try:
        answer = ENGINE.reply(message, history=history, temperature=temperature, top_k=int(top_k))
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        answer = f"**PyMind setup issue:** {exc}"
    updated = list(history or [])
    updated.extend([{"role": "user", "content": message}, {"role": "assistant", "content": answer}])
    return "", updated


def build_site(prompt: str):
    try:
        project = generate_website(prompt)
        with tempfile.TemporaryDirectory(prefix="pymind-site-") as folder:
            site_dir = write_website(project, Path(folder) / "website")
            archive = Path(tempfile.gettempdir()) / f"{site_dir.name}.zip"
            with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
                for file in site_dir.iterdir():
                    bundle.write(file, arcname=f"{_slug(project.name)}/{file.name}")
        preview = project.files["index.html"].replace(
            '<link rel="stylesheet" href="styles.css">',
            f"<style>{project.files['styles.css']}</style>",
        ).replace(
            '<script src="script.js"></script>',
            f"<script>{project.files['script.js']}</script>",
        )
        # The preview is isolated from the Gradio page in a sandboxed iframe.
        iframe = (
            '<iframe title="Generated website preview" '
            'sandbox="allow-scripts" referrerpolicy="no-referrer" '
            f'srcdoc="{html.escape(preview, quote=True)}" '
            'style="width:100%;height:680px;border:1px solid #dbe2ea;border-radius:14px;background:white"></iframe>'
        )
        status = (
            f"Generated **{project.name}** ({project.kind} template) with "
            "index.html, styles.css, script.js and README.txt. "
            "This is a template-assisted starter, not autonomous TinyGPT code generation."
        )
        return status, project.files["index.html"], project.files["styles.css"], project.files["script.js"], iframe, str(archive)
    except (ValueError, OSError) as exc:
        message = f"Website generation failed: {exc}"
        return message, "", "", "", "", None


with gr.Blocks(title="PyMind-A", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# 🧠 PyMind-A\n"
        "Python-first AI workspace · Tamil + English learning project\n\n"
        "> The TinyGPT chat model is an educational prototype. Website Builder currently uses "
        "safe, deterministic templates; it does not claim the tiny model can write production code."
    )
    with gr.Tabs():
        with gr.Tab("AI Chat"):
            chatbot = gr.Chatbot(label="Conversation", type="messages", height=470)
            with gr.Row():
                message = gr.Textbox(placeholder="Message PyMind... / PyMind-க்கு message அனுப்பு", show_label=False, scale=8, lines=2)
                send = gr.Button("Send", variant="primary", scale=1)
            with gr.Row():
                temperature = gr.Slider(0.2, 1.5, value=0.8, step=0.1, label="Creativity")
                top_k = gr.Slider(0, 100, value=40, step=1, label="Top-K (0 = off)")
            clear = gr.Button("＋ New chat")
            send.click(respond, inputs=[message, chatbot, temperature, top_k], outputs=[message, chatbot])
            message.submit(respond, inputs=[message, chatbot, temperature, top_k], outputs=[message, chatbot])
            clear.click(lambda: ([], ""), outputs=[chatbot, message])
        with gr.Tab("Website Builder"):
            gr.Markdown("### Describe the website you want")
            prompt = gr.Textbox(
                label="Website brief",
                placeholder="Example: Create a modern portfolio website for a Python developer.",
                lines=4,
                max_lines=8,
            )
            build = gr.Button("Generate website", variant="primary")
            status = gr.Markdown()
            preview = gr.HTML(label="Sandboxed preview")
            with gr.Accordion("Generated source code", open=False):
                html_code = gr.Code(label="index.html", language="html")
                css_code = gr.Code(label="styles.css", language="css")
                js_code = gr.Code(label="script.js", language="javascript")
            download = gr.File(label="Download website ZIP")
            build.click(build_site, inputs=[prompt], outputs=[status, html_code, css_code, js_code, preview, download])
        with gr.Tab("Capabilities"):
            gr.Markdown(
                "### Available in this repository\n"
                "- **Text chat:** local TinyGPT checkpoint (small educational model).\n"
                "- **Website builder:** responsive starter templates with isolated preview and ZIP export.\n"
                "- **Python code inspection:** static analysis; inspected code is not executed.\n"
                "- **Blender:** procedural scene-to-GLB script; Blender must be installed separately.\n\n"
                "### Not connected yet\n"
                "- **Image generation:** requires a dedicated image model/runtime.\n"
                "- **Video generation:** requires a dedicated video model/runtime.\n"
                "- **General AI code authoring:** needs a capable code model; the current TinyGPT is not reliable for this.\n\n"
                "No API keys are required for the deterministic website templates. External media models may need "
                "separate hardware, downloads, or service credentials."
            )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")))
