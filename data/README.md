# Training data guide

The bundled `data/input.txt` is a tiny demonstration corpus for smoke tests and learning. It is not large or diverse enough to produce useful general-purpose language generation.

## Build a better dataset

1. Collect text that you own, wrote yourself, or have permission to use.
2. Keep a consistent subject and language for each experiment (for example, Python tutorials or Tamil agriculture notes).
3. Use plain UTF-8 text. Preserve meaningful punctuation and paragraph boundaries.
4. Remove duplicated pages, navigation menus, broken markup, personal data, and irrelevant boilerplate.
5. Keep a record of each source and its license/permission outside the training text.
6. Split evaluation material by document or source when possible, so near-duplicate passages do not leak into validation.
7. Start with a few hundred KB of clean text and increase gradually; monitor both training and validation loss.

Do not scrape private content or assume that publicly visible text is automatically licensed for model training. Do not put passwords, API keys, personal records, or confidential documents in the dataset.

## Run a small experiment

```bash
python -m pymind.train --data data/input.txt --max-iters 100 --eval-interval 10 --eval-batches 5 --batch-size 4 --block-size 32 --n-embd 32 --n-head 4 --n-layer 1 --grad-accum-steps 2 --early-stopping-patience 3
```

The trainer saves the checkpoint with the lowest observed validation loss, not simply the final step. The CSV log records train loss, validation loss, and best validation loss. Validation loss is a useful signal, not a guarantee of factual accuracy or human-quality text.
