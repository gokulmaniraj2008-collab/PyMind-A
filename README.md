# PyMind-A

**A Python-first open-source learning project for language models and AI tool integrations.**

PyMind-A is an educational decoder-only GPT project built with Python and PyTorch. It learns next-byte prediction from text datasets you provide. The long-term goal is a modular AI workspace that can connect language models to coding and creative tools. It is not currently comparable to ChatGPT and does not natively generate images or videos.

## Current working components

- Small decoder-only Transformer with causal self-attention
- UTF-8 byte tokenizer (256 IDs), including Tamil, English, and emoji
- Configurable training data and model parameters
- Validation-based best-checkpoint saving, optional early stopping, and gradient accumulation
- Configurable evaluation batches and AdamW weight decay
- Train/validation loss history in CSV format
- UTF-8 dataset cleaner with Unicode NFC normalization and duplicate-line removal
- Held-out dataset evaluation with next-byte loss and perplexity
- Small original Tamil + English starter corpus for pipeline tests (not a capable training corpus)
- Reproducible initialization seed
- CPU-first training and generation
- Top-K and Top-P sampling, temperature, repetition penalty, and generation seed
- Checkpoint save/load, including legacy Phase 1 character-tokenizer checkpoints
- JSON-driven Blender scene creation and GLB export script (requires Blender installed)
- Offline Python code inspector: syntax, imports, functions/classes, and heuristic risk checks
- Unit tests and GitHub Actions CI

## Repository layout

```text
PyMind-A/
├── app.py                            # Gradio chat website
├── requirements-ui.txt               # Optional browser UI dependencies
├── data/input.txt
├── data/starter_ta_en.txt
├── data/sources.example.csv
├── data/README.md
├── examples/blender_scene.json
├── output/                         # generated GLB files; do not commit large outputs
├── checkpoints/                    # generated weights; git-ignored
├── logs/                            # generated CSV; git-ignored
├── pymind/
│   ├── model.py
│   ├── tokenizer.py
│   ├── train.py
│   ├── generate.py
│   └── integrations/blender/
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

## Prepare and evaluate datasets

The included `data/starter_ta_en.txt` is a tiny original Tamil + English starter sample for testing the pipeline only; it is not large enough for meaningful language ability. Read [the data guide](data/README.md) before adding data.

Clean a UTF-8 text file (normalizes Unicode to NFC, removes duplicate non-empty lines, and preserves paragraph breaks):

```bash
python -m pymind.data_clean --input data/starter_ta_en.txt --output data/cleaned.txt
```

After training, evaluate on a separate held-out text file:

```bash
python -m pymind.evaluate --checkpoint checkpoints/tiny_gpt.pt --data path/to/held_out.txt
```

The evaluator reports next-byte loss and perplexity. Use data from different documents/sources for evaluation where possible; never evaluate on the same text used for training.

## Train and generate text

```bash
python -m pymind.train --data data/input.txt --max-iters 1000
python -m pymind.generate --checkpoint checkpoints/tiny_gpt.pt --prompt "வணக்கம்"
```

Generation controls:

```bash
python -m pymind.generate \
  --checkpoint checkpoints/tiny_gpt.pt \
  --prompt "வணக்கம்" \
  --max-new-tokens 120 \
  --temperature 0.8 \
  --top-k 40 \
  --top-p 0.9 \
  --repetition-penalty 1.1 \
  --seed 1337
```

- **Temperature:** lower values make sampling more conservative; higher values increase randomness.
- **Top-K:** limits each next-token choice to the K most likely tokens. Use `--top-k 0` to disable it.
- **Top-P:** nucleus sampling keeps the smallest high-probability set whose cumulative probability reaches P.
- **Repetition penalty:** discourages reuse of tokens already present in the sequence.
- **Seed:** makes sampling reproducible on the same software/device setup.

For a small CPU experiment:

```bash
python -m pymind.train --data data/input.txt --max-iters 20 --eval-interval 10 --batch-size 4 --block-size 32 --n-embd 32 --n-head 4 --n-layer 1
```

Training saves the checkpoint with the lowest observed validation loss to `checkpoints/tiny_gpt.pt` and writes `logs/training_metrics.csv` by default. The CSV contains `step`, `train_loss`, `val_loss`, and `best_val_loss`. Optional `--early-stopping-patience 5` stops after five evaluation rounds without improvement; `0` disables it. `--grad-accum-steps 2` accumulates gradients over two batches before each optimizer update. See [the training data guide](data/README.md) before adding a larger corpus.

## ChatGPT-style web chat prototype

A browser chat interface is available through Gradio. It supports a conversation transcript, new-chat reset, Tamil/English text input, and sampling controls. Replies are generated by the local TinyGPT checkpoint; the interface is a prototype and the model is not instruction-tuned or comparable to ChatGPT.

Install the optional UI dependencies and train a byte-tokenizer checkpoint:

```bash
pip install -r requirements-ui.txt
python -m pymind.train --data data/starter_ta_en.txt --max-iters 100
python app.py
```

Open the local address printed by Gradio (normally `http://127.0.0.1:7860`). For another checkpoint path, set `PYMIND_CHECKPOINT`. The included starter dataset is only for a quick pipeline demonstration; meaningful chat quality requires a much larger, legally usable, high-quality instruction dataset and substantial training compute.

## Offline Python coding assistant tools

PyMind-A now includes a deterministic static Python inspector. It parses source code without executing it, summarizes imports/functions/classes, checks syntax, and flags selected risky patterns such as `eval`, `exec`, `os.system`, `subprocess.run(shell=True)`, pickle loading, hard-coded secret-like assignments, bare exception handlers, and mutable defaults.

```bash
python -m pymind.coding path/to/your_script.py
python -m pymind.coding path/to/your_script.py --format json
python -m pymind.coding path/to/your_script.py --fail-on-high
```

This is an offline code-review helper, not yet a conversational coding model and not a complete security audit. It never runs the inspected file. The tiny GPT is not yet reliable enough to author or repair production code autonomously.

## Blender scene to GLB

Install Blender separately. Then run from the repository root:

```bash
blender --background --python pymind/integrations/blender/create_scene.py -- --spec examples/blender_scene.json --output output/pymind_scene.glb
```

This creates simple procedural geometry from a JSON specification and exports a GLB. It is a deterministic scene builder—not yet an AI text-to-3D model. The language model is not yet connected to this integration.

Supported primitives: `cube`, `sphere`, and `cylinder`. Each object can specify `name`, `location`, `scale`, and RGBA `color`.

## Prompt-to-website builder

The Gradio workspace includes a Website Builder tab. Describe a business landing page, portfolio, product page, or 3D viewer; inspect generated HTML/CSS/JavaScript, preview it in an isolated sandboxed iframe, and download a ZIP. The first version uses deterministic Python templates rather than claiming the tiny GPT can autonomously author reliable websites. See [the Website Builder guide](docs/WEBSITE_BUILDER.md).

Run the browser workspace:

```bash
pip install -r requirements-ui.txt
python app.py
```

The website builder does not need a model checkpoint. The chat tab still needs a trained checkpoint. Image/video generation are not connected yet and require separate models or services.

## Tests

```bash
python -m pytest -q
python -m compileall -q pymind tests
```

## Reality check and roadmap

A single small GPT cannot realistically overtake ChatGPT in one coding attempt. Frontier models require large datasets, extensive training compute, research, evaluations, and safety work. PyMind-A can still become a useful personal, modular AI platform.

1. Stabilize and evaluate the language model on larger datasets.
2. Improve tokenizer and text generation quality.
3. Connect the model to safe Python tools and code workflows.
4. Build Blender scene generation and validate GLB exports.
5. Add optional image/video model integrations only when a suitable local or hosted engine is available.
6. Build an Android-friendly web interface.

Image/video generation, advanced reasoning, and ChatGPT-level performance are not implemented today. No paid API key is required for the current CPU-only training and Blender JSON workflow; image/video engines may have their own hardware or service requirements.

## License

No license has been added yet. Add a license before accepting external contributions or distributing the project under specific reuse terms.
