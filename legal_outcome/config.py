from dataclasses import dataclass, asdict
import math


@dataclass(frozen=True)
class ModelConfig:
    labels: tuple[str, ...]
    legalbert_model: str = "nlpaueb/legal-bert-base-uncased"
    minilm_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    legalbert_revision: str | None = None
    minilm_revision: str | None = None
    max_tokens: int = 384
    max_spans: int = 8
    dropout: float = 0.1
    task_type: str = "single_label"
    threshold: float = 0.5
    positive_weights: tuple[float, ...] | None = None
    graph_layers: int = 2
    order_decay: float = 1.0
    cue_scale: float = 1.0
    # The equations figure marks layers 5–12 trainable. Default to all layers,
    # following the prose and first figure. This option supports that variant.
    freeze_lower_layers: int = 0

    def __post_init__(self):
        object.__setattr__(self, "labels", tuple(self.labels))
        if self.task_type not in ("single_label", "multi_label"):
            raise ValueError("Unknown task type")
        if not 0 < self.threshold < 1:
            raise ValueError("Threshold must be between zero and one")
        if self.positive_weights is not None:
            weights = tuple(self.positive_weights)
            if self.task_type != "multi_label" or len(weights) != len(self.labels) or any(not math.isfinite(w) or w <= 0 for w in weights):
                raise ValueError("Positive weights require one positive finite value per multi-label class")
            object.__setattr__(self, "positive_weights", weights)
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
