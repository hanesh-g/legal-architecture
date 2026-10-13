"""Trainable token/span model. MiniLM features enter only the graph priors."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .config import ModelConfig


class LabelAttention(nn.Module):
    def __init__(self, labels, hidden):
        super().__init__()
        self.queries = nn.Parameter(torch.empty(labels, hidden))
        nn.init.normal_(self.queries, std=0.02)
        self.key = nn.Linear(hidden, hidden)
        self.value = nn.Linear(hidden, hidden)

    def forward(self, states, mask):
        scores = torch.einsum("ld,btd->blt", self.queries, self.key(states)) / math.sqrt(states.shape[-1])
        scores = scores.masked_fill(~mask[:, None, :], float("-inf"))
        weights = scores.softmax(-1)
        return weights @ self.value(states), weights


class DualTokenAttention(nn.Module):
    """Two independent label queries, combined by a learned elementwise gate."""
    def __init__(self, labels, hidden):
        super().__init__()
        self.first = LabelAttention(labels, hidden)
        self.second = LabelAttention(labels, hidden)
        self.gate = nn.Linear(2 * hidden, hidden)

    def forward(self, states, mask):
        first, a = self.first(states, mask)
        second, b = self.second(states, mask)
        gate = torch.sigmoid(self.gate(torch.cat([first, second], -1)))
        return gate * first + (1 - gate) * second, torch.stack([a, b], 1)


def mean_span_pool(states, span_ids, span_mask):
    n = span_mask.shape[1]
    membership = (span_ids[:, :, None] == torch.arange(n, device=states.device)[None, None, :])
    membership = membership & span_mask[:, None, :]
    weights = membership.to(states.dtype)
    return (weights.transpose(1, 2) @ states) / weights.sum(1).clamp_min(1).unsqueeze(-1)


class CueRefinement(nn.Module):
    def __init__(self, hidden, scale):
        super().__init__()
        self.gate = nn.Linear(hidden + 12, hidden)
        self.projection = nn.Linear(12, hidden, bias=False)
        self.scale = scale

    def forward(self, pooled, cues, mask):
        # Raw regex counts implement f_cue in the supplied equation.
        gate = torch.sigmoid(self.gate(torch.cat([pooled, cues], -1)))
        refined = pooled + self.scale * gate * self.projection(cues)
        return refined * mask.unsqueeze(-1)


class AdaptiveSpanGraph(nn.Module):
    def __init__(self, hidden, layers, order_decay, dropout):
        super().__init__()
        self.prior_logits = nn.Parameter(torch.zeros(3))
        self.layers = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.dropout = nn.Dropout(dropout)
        self.order_decay = order_decay

    def adjacency(self, semantic, mask):
        # Frozen MiniLM output is used only for cosine-based edges.
        unit = F.normalize(semantic.detach(), dim=-1)
        similarity = (unit @ unit.transpose(1, 2)).clamp_min(0)
        n = mask.shape[1]
        indices = torch.arange(n, device=semantic.device)
        order = torch.exp(-self.order_decay * (indices[:, None] - indices[None, :]).abs().to(semantic.dtype))
        identity = torch.eye(n, device=semantic.device, dtype=semantic.dtype)
        mixing = self.prior_logits.softmax(0)
        a = mixing[0] * similarity + mixing[1] * order + mixing[2] * identity
        a = a * (mask[:, :, None] & mask[:, None, :])
        # Symmetric degree normalization. Identity already supplies self loops.
        inv_degree = a.sum(-1).clamp_min(1e-8).rsqrt()
        return inv_degree[:, :, None] * a * inv_degree[:, None, :]

    def forward(self, states, semantic, mask):
        a = self.adjacency(semantic, mask)
        for layer, norm in zip(self.layers, self.norms):
            states = norm(states + self.dropout(F.gelu(layer(a @ states))))
            states = states * mask.unsqueeze(-1)
        return states, a


class LegalOutcomeModel(nn.Module):
    def __init__(self, config: ModelConfig, encoder=None):
        super().__init__()
        self.config = config
        if encoder is None:
            from transformers import AutoModel
            encoder = AutoModel.from_pretrained(config.legalbert_model)
        self.encoder = encoder
        hidden = encoder.config.hidden_size
        labels = len(config.labels)
        if config.freeze_lower_layers:
            if not hasattr(encoder, "encoder") or not hasattr(encoder.encoder, "layer"):
                raise ValueError("Layer freezing requires a BERT-compatible encoder")
            if config.freeze_lower_layers > len(encoder.encoder.layer):
                raise ValueError("Requested more frozen layers than encoder provides")
            for block in [encoder.embeddings, *encoder.encoder.layer[:config.freeze_lower_layers]]:
                for parameter in block.parameters():
                    parameter.requires_grad = False
        self.token_attention = DualTokenAttention(labels, hidden)
        self.refinement = CueRefinement(hidden, config.cue_scale)
        self.graph = AdaptiveSpanGraph(hidden, config.graph_layers, config.order_decay, config.dropout)
        self.span_attention = LabelAttention(labels, hidden)
        self.fusion_gate = nn.Linear(hidden * 3, hidden)
        self.dropout = nn.Dropout(config.dropout)
        self.head_weight = nn.Parameter(torch.empty(labels, hidden * 3))
        self.head_bias = nn.Parameter(torch.zeros(labels))
        nn.init.xavier_uniform_(self.head_weight)

    def forward(self, input_ids, attention_mask, span_ids, span_mask, cue_features,
                semantic_embeddings, labels=None):
        if input_ids.ndim != 2 or input_ids.shape != attention_mask.shape or input_ids.shape != span_ids.shape:
            raise ValueError("Token inputs must have matching [batch, tokens] shapes")
        b, t = input_ids.shape
        if t > self.config.max_tokens or span_mask.ndim != 2 or span_mask.shape[0] != b:
            raise ValueError("Invalid token or span dimensions")
        n = span_mask.shape[1]
        if not 1 <= n <= self.config.max_spans or not span_mask.bool().any(-1).all():
            raise ValueError("Each document requires a valid span")
        if cue_features.shape != (b, n, 12) or semantic_embeddings.ndim != 3 or semantic_embeddings.shape[:2] != (b, n):
            raise ValueError("Invalid cue or semantic embedding dimensions")
        if not torch.isfinite(cue_features).all() or not torch.isfinite(semantic_embeddings).all():
            raise ValueError("Cue features and semantic embeddings must be finite")
        span_mask = span_mask.bool()
        token_mask = attention_mask.bool()
        if not token_mask[:, 0].all() or not token_mask.any(-1).all():
            raise ValueError("CLS must be visible in every document")
        if ((span_ids < -1) | (span_ids >= n)).any() or ((span_ids >= 0) & ~token_mask).any():
            raise ValueError("Invalid token-to-span mapping")
        membership = span_ids[:, :, None] == torch.arange(n, device=span_ids.device)
        if not torch.equal(membership.any(1), span_mask):
            raise ValueError("Span mask must match populated token spans")
        states = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        context = states[:, 0]
        # All visible tokens (including CLS/SEP) participate in Token-LCA.
        token, token_weights = self.token_attention(states, token_mask)
        pooled = mean_span_pool(states, span_ids, span_mask)
        refined = self.refinement(pooled, cue_features, span_mask)
        enhanced, adjacency = self.graph(refined, semantic_embeddings, span_mask)
        span, span_weights = self.span_attention(enhanced, span_mask)
        expanded = context[:, None, :].expand(-1, len(self.config.labels), -1)
        gate = torch.sigmoid(self.fusion_gate(torch.cat([token, span, expanded], -1)))
        fused = self.dropout(torch.cat([token, gate * span, expanded], -1))
        logits = torch.einsum("bld,ld->bl", fused, self.head_weight) + self.head_bias
        output = {"logits": logits, "probabilities": logits.softmax(-1), "predictions": logits.argmax(-1),
                  "token_attention": token_weights, "span_attention": span_weights,
                  "span_gate": gate, "adjacency": adjacency}
        if labels is not None:
            output["loss"] = F.cross_entropy(logits, labels)
        return output
