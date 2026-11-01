# PyMind-A

**A Python-first open-source learning project for language models and AI tool integrations.**

PyMind-A is an educational decoder-only GPT project built with Python and PyTorch. It learns next-byte prediction from text datasets you provide. The long-term goal is a modular AI workspace that can connect language models to coding and creative tools. It is not currently comparable to ChatGPT and does not natively generate images or videos.

## Current working components

- Small decoder-only Transformer with causal self-attention
- UTF-8 byte tokenizer (256 IDs), including Tamil, English, and emoji
- Configurable training data and model parameters
- Train/validation loss history in CSV format
- Reproducible initialization seed
- CPU-first training and generation
- Checkpoint save/load, including legacy Phase 1 character-tokenizer checkpoints
- JSON-driven Blender scene creation and GLB export script (requires Blender installed)
- Unit tests and GitHub Actions CI

## Repository layout

```text
PyMind-A/
├── data/input.txt
├── examples/blender_scene.json
├── output/                         # generated GLB files; do not commit large outputs
├── checkpoints/                    # generated weights; git-ignored
├── logs/                           # generated CSV; git-ignored
├── pymind/
│   ├── model.py
│   ├── tokenizer.py
│   ├── train.py
│   ├── generate.py
│   └── integrations/blender/
│       ├── scene_spec.py
│       └── create_scene.py
├── tests/
├── requirements.txt
└── README.md
```

## Quick start

Use Python 3.10 or newer.

```bash
python -m venv .venv
```

Activate the environment and install dependencies:

```bash
# Linux / macOS
source .venv/bin/activate

# Windows PowerShell
.venv\\Scripts\\Activate.ps1

pip install -r requirements.txt
```

The default dependencies use CPU-only PyTorch, so the basic workflow does not require a CUDA GPU.

## Train and generate text

```bash
python -m pymind.train --data data/input.txt --max-iters 1000
python -m pymind.generate --checkpoint checkpoints/tiny_gpt.pt --prompt "வணக்கம்"
```

For a small CPU experiment:

```bash
python -m pymind.train --data data/input.txt --max-iters 20 --eval-interval 10 --batch-size 4 --block-size 32 --n-embd 32 --n-head 4 --n-layer 1
```

Training writes `checkpoints/tiny_gpt.pt` and `logs/training_metrics.csv` by default. The CSV contains `step`, `train_loss`, and `val_loss`.

## Blender scene to GLB

Install Blender separately. Then run from the repository root:

```bash
blender --background --python pymind/integrations/blender/create_scene.py -- --spec examples/blender_scene.json --output output/pymind_scene.glb
```

This creates simple procedural geometry from a JSON specification and exports a GLB. It is a deterministic scene builder—not yet an AI text-to-3D model. The language model is not yet connected to this integration.

Supported primitives: `cube`, `sphere`, and `cylinder`. Each object can specify `name`, `location`, `scale`, and RGBA `color`.

## Tests

```bash
python -m pytest -q
python -m compileall -q pymind tests
```

## Reality check and roadmap

A single small GPT cannot realistically overtake ChatGPT in one coding attempt. Frontier models require large datasets, extensive training compute, research, evaluations, and safety work. PyMind-A can still become a useful personal, modular AI platform.

1. Stabilize the language model and CI.
2. Improve evaluation, sampling, data handling, and model quality.
3. Connect the model to safe Python tools and code workflows.
4. Build Blender scene generation and validate GLB exports.
5. Add optional image/video model integrations only when a suitable local or hosted engine is available.
6. Build an Android-friendly web interface.

Image/video generation, advanced reasoning, and ChatGPT-level performance are not implemented today. No paid API key is required for the current CPU-only training and Blender JSON workflow; image/video engines may have their own hardware or service requirements.

## License

No license has been added yet. Add a license before accepting external contributions or distributing the project under specific reuse terms.
