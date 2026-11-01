# PyMind-A Website Builder

## What this phase delivers

The Gradio workspace now has a Website Builder tab. Enter a brief to generate a responsive starter website, inspect the HTML/CSS/JavaScript, preview it in a sandboxed iframe, and download a ZIP.

Supported starter types are business landing pages, developer portfolios, product/store landing pages, and a procedural Three.js 3D car viewer.

## Important model boundary

The current PyMind-A TinyGPT is a small byte-level educational language model. It is **not** currently capable of reliably writing arbitrary production website code. To make the first usable builder available without falsely claiming model capabilities, this phase uses deterministic Python templates selected from the user's brief. The generated site is editable and serves as a foundation for a later code-model integration.

## Run

```bash
pip install -r requirements-ui.txt
python app.py
```

Train a checkpoint first if you also want to use the local chat tab:

```bash
python -m pymind.train --data data/pymind_training.txt --max-iters 150
```

The Website Builder itself does not need a model checkpoint or API key.

## Generated files

- `index.html`: semantic responsive page
- `styles.css`: responsive styling
- `script.js`: small interactions; the 3D template initializes a Three.js scene
- `README.txt`: generated project notes

The preview runs in an iframe with the `sandbox="allow-scripts"` attribute and no same-origin permission. Treat downloaded source code as editable project output and review it before deployment.

## Image, video, and 3D roadmap

The existing Blender integration can create simple procedural scenes and export GLB when Blender is installed. The 3D website template currently demonstrates browser-based primitive geometry; it does not create photorealistic assets.

Image and video generation are not implemented by this phase. They need separate image/video models and compatible compute or a hosted service. This repository does not silently call paid APIs or pretend those engines are available.
