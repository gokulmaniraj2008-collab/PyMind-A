"""Conversation prompt utilities, FAQ retrieval and local TinyGPT inference."""

from difflib import SequenceMatcher
from pathlib import Path

import torch

from pymind.model import GPTConfig, TinyGPT
from pymind.tokenizer import ByteTokenizer


_STOPWORDS = {
    "a", "an", "and", "are", "can", "could", "do", "does", "for", "how",
    "i", "is", "it", "me", "of", "please", "the", "to", "tell", "what",
    "when", "where", "which", "who", "why", "with", "you", "your", "about",
}


def normalize_question(text: str) -> str:
    cleaned = "".join(
        character if character.isalnum() or character.isspace() else " "
        for character in text.casefold()
    )
    return " ".join(cleaned.split())


def build_chat_prompt(message: str, history: list | None = None, max_turns: int = 4) -> str:
    """Build a bounded role-labelled prompt matching the training corpus format."""
    if not isinstance(message, str) or not message.strip():
        raise ValueError("message must be a non-empty string")
    if max_turns < 0:
        raise ValueError("max_turns cannot be negative")
    lines = []
    valid_history = []
    for item in (history or []):
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
            valid_history.append((role.title(), content.strip()))
    for role, content in valid_history[-max_turns:] if max_turns else []:
        lines.append(f"{role}: {content}")
    lines.append(f"User: {message.strip()}")
    lines.append("Assistant:")
    return "\n".join(lines)


class LocalChatEngine:
    """Local TinyGPT with bounded FAQ retrieval for known beginner questions."""

    def __init__(
        self,
        checkpoint_path: str = "checkpoints/tiny_gpt.pt",
        corpus_path: str = "data/pymind_training.txt",
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.corpus_path = Path(corpus_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = ByteTokenizer()
        self.model = None
        self.faq_pairs = self._load_faq_pairs()

    def _load_faq_pairs(self):
        if not self.corpus_path.is_file():
            return []
        pairs = []
        question = None
        for line in self.corpus_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("User:"):
                question = line[len("User:"):].strip()
            elif line.startswith("Assistant:") and question:
                answer = line[len("Assistant:"):].strip()
                if answer:
                    pairs.append((question, answer))
                question = None
        return pairs

    def retrieve_answer(self, message: str) -> str | None:
        """Find an exact or strongly overlapping question in the training corpus."""
        query = normalize_question(message)
        if not query:
            return None
        for question, answer in self.faq_pairs:
            if query == normalize_question(question):
                return answer

        query_terms = set(query.split()) - _STOPWORDS
        if not query_terms:
            return None
        best_score = 0.0
        best_answer = None
        for question, answer in self.faq_pairs:
            candidate = normalize_question(question)
            candidate_terms = set(candidate.split()) - _STOPWORDS
            if not candidate_terms:
                continue
            overlap = len(query_terms & candidate_terms) / len(query_terms | candidate_terms)
            sequence = SequenceMatcher(None, query, candidate).ratio()
            score = max(overlap, sequence)
            if score > best_score:
                best_score = score
                best_answer = answer
        return best_answer if best_score >= 0.62 else None

    def load(self):
        if self.model is not None:
            return
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {self.checkpoint_path}. "
                "Train the model first using python -m pymind.train."
            )
        checkpoint = torch.load(
            self.checkpoint_path, map_location=self.device, weights_only=False
        )
        if checkpoint.get("tokenizer", {}).get("type") != ByteTokenizer.tokenizer_type:
            raise ValueError("Chat UI currently requires a UTF-8 byte-tokenizer checkpoint.")
        config = GPTConfig(**checkpoint["config"])
        self.model = TinyGPT(config).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

    @torch.inference_mode()
    def reply(
        self,
        message: str,
        history: list | None = None,
        temperature: float = 0.65,
        top_k: int = 20,
        max_new_tokens: int = 80,
    ) -> str:
        retrieved = self.retrieve_answer(message)
        if retrieved:
            return retrieved
        self.load()
        prompt = build_chat_prompt(message, history)
        token_ids = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        output = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=max(temperature, 0.05),
            top_k=top_k if top_k > 0 else None,
            top_p=0.9,
            repetition_penalty=1.1,
        )[0].tolist()
        decoded = self.tokenizer.decode(output)
        answer = decoded[len(prompt):].strip()
        return answer or "I am still learning. Try a short question about Python, AI, or Blender."
