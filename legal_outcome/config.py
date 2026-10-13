from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class ModelConfig:
    labels: tuple[str, ...]
    legalbert_model: str = "nlpaueb/legal-bert-base-uncased"
    minilm_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_tokens: int = 384
    max_spans: int = 8
    dropout: float = 0.1
    graph_layers: int = 2
    order_decay: float = 1.0
    cue_scale: float = 1.0
    # The equations figure marks layers 5–12 trainable. Default to all layers,
    # following the prose and first figure. This option supports that variant.
    freeze_lower_layers: int = 0

    def __post_init__(self):
        object.__setattr__(self, "labels", tuple(self.labels))
        if len(self.labels) < 2 or len(set(self.labels)) != len(self.labels):
            raise ValueError("Use at least two distinct labels")
        if any(not isinstance(x, str) or not x.strip() for x in self.labels):
            raise ValueError("Labels must be nonempty strings")
        if not 1 <= self.max_spans <= 8 or not 3 <= self.max_tokens <= 384:
            raise ValueError("Require 1–8 spans and 3–384 tokens")
        if not 0 <= self.dropout < 1 or self.graph_layers < 1:
            raise ValueError("Invalid dropout or graph layer count")
        if self.order_decay <= 0 or self.cue_scale < 0:
            raise ValueError("Invalid refinement or order scale")
        if not 0 <= self.freeze_lower_layers <= 12:
            raise ValueError("freeze_lower_layers must be between 0 and 12")

    def to_dict(self):
        return asdict(self)
