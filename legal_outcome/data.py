"""Ordered span segmentation and packed token-to-span alignment."""
import re
import torch
from .cues import CueExtractor


def segment_document(text, max_spans=8):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Document text must be nonempty")
    if not 1 <= max_spans <= 8:
        raise ValueError("max_spans must be between 1 and 8")
    # Retain original whitespace and order. Boundaries occur only between words.
    words = list(re.finditer(r"\S+\s*", text))
    n = min(max_spans, len(words))
    spans = []
    for i in range(n):
        lo, hi = i * len(words) // n, (i + 1) * len(words) // n
        spans.append(text[words[lo].start():words[hi - 1].end()])
    return spans


class DocumentProcessor:
    def __init__(self, tokenizer, config, cue_extractor=None):
        self.tokenizer, self.config = tokenizer, config
        self.cues = cue_extractor or CueExtractor()
        for name in ("cls_token_id", "sep_token_id", "pad_token_id"):
            if getattr(tokenizer, name, None) is None:
                raise ValueError(f"Tokenizer requires {name}")

    def __call__(self, text):
        spans = segment_document(text, self.config.max_spans)
        budget = self.config.max_tokens - 2
        # Use leading truncation, as in the source. Drop spans with no retained
        # tokens. Cues and MiniLM see exactly the retained token text.
        ids, mapping, retained = [], [], []
        for span in spans:
            tokens = self.tokenizer.encode(span, add_special_tokens=False)[:budget - len(ids)]
            if not tokens:
                continue
            mapping.extend([len(retained)] * len(tokens))
            ids.extend(tokens)
            retained.append(self.tokenizer.decode(tokens, skip_special_tokens=True))
            if len(ids) == budget:
                break
        if not retained:
            raise ValueError("Document has no usable tokens")
        length = len(ids) + 2
        padding = self.config.max_tokens - length
        # -1 marks special tokens and padding; span indices are zero-based.
        return {
            "input_ids": torch.tensor([self.tokenizer.cls_token_id, *ids, self.tokenizer.sep_token_id] + [self.tokenizer.pad_token_id] * padding),
            "attention_mask": torch.tensor([1] * length + [0] * padding),
            "span_ids": torch.tensor([-1, *mapping, -1] + [-1] * padding),
            "spans": retained,
            "cue_features": self.cues(retained),
        }


def collate_documents(items, semantic_encoder):
    if not items:
        raise ValueError("Cannot collate an empty batch")
    n = max(len(x["spans"]) for x in items)
    flat = [s for x in items for s in x["spans"]]
    with torch.no_grad():
        embeddings = semantic_encoder.encode(flat, convert_to_tensor=True, normalize_embeddings=True)
    embeddings = torch.as_tensor(embeddings).detach().cpu().float()
    if embeddings.ndim != 2 or embeddings.shape[0] != len(flat):
        raise ValueError("Semantic encoder returned an invalid shape")
    batch = {k: torch.stack([x[k] for x in items]) for k in ("input_ids", "attention_mask", "span_ids")}
    batch["span_mask"] = torch.zeros(len(items), n, dtype=torch.bool)
    batch["cue_features"] = torch.zeros(len(items), n, 12)
    batch["semantic_embeddings"] = torch.zeros(len(items), n, embeddings.shape[-1])
    offset = 0
    for i, item in enumerate(items):
        count = len(item["spans"])
        batch["span_mask"][i, :count] = True
        batch["cue_features"][i, :count] = item["cue_features"]
        batch["semantic_embeddings"][i, :count] = embeddings[offset:offset + count]
        offset += count
    if all("label" in x for x in items):
        batch["labels"] = torch.tensor([x["label"] for x in items])
        if batch["labels"].ndim == 2:
            batch["labels"] = batch["labels"].float()
    return batch
