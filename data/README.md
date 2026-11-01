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


## Included bilingual starter file

`data/starter_ta_en.txt` contains a small set of original Tamil and English educational sentences. It exists to test Unicode handling, cleaning, evaluation, and CI. It is deliberately not described as a large corpus and will not make the tiny model fluent.

## Dataset cleaning and evaluation commands

```bash
python -m pymind.data_clean --input raw.txt --output cleaned.txt
python -m pymind.train --data cleaned.txt --max-iters 500
python -m pymind.evaluate --checkpoint checkpoints/tiny_gpt.pt --data held_out.txt
```

Keep `held_out.txt` separate from training text. Compare validation/evaluation loss across experiments using the same held-out set. Perplexity values are only comparable when tokenizer and evaluation data are the same.


## Record dataset provenance

Use `data/sources.example.csv` as a template. Replace the example row with the real source title, URL, license, permission basis, language, and collection date. Do not treat the example URL as a real training source. Only add material that you created, have permission to use, or whose license clearly permits the intended use. Keep personal or confidential information out of datasets.
