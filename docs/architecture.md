# Architecture decisions

Source: the three-page PDF supplied by the user, *Proposed Architecture for Legal Outcome.pdf*. Page 1 shows the flow. Page 2 shows equations. Page 3 explains the flow in prose. The implementation treats the document as the architecture specification.

## Tensor contract

Let B be batch size, T be packed token count, N be retained span count, D be LegalBERT hidden size, and L be class count. T is at most 384. N is at most 8. Cue count K is 12.

| Tensor | Shape |
| --- | --- |
| Token IDs, token mask, span IDs | B × T |
| Token states | B × T × D |
| Span mask | B × N |
| Cue counts | B × N × 12 |
| MiniLM embeddings | B × N × 384 for the default MiniLM model |
| Pooled, refined, and graph states | B × N × D |
| Token and span evidence | B × L × D |
| Fused evidence | B × L × 3D |
| Scores and probabilities | B × L |

The normal encoder hidden size is 768. Tests inject smaller encoders. MiniLM dimensions do not set the classifier hidden size.

## Equations implemented

Attention uses `softmax(Q K(H)^T / sqrt(D)) V(H)` with masked positions excluded. Two token branches use separate queries and projections. A learned sigmoid gate combines their evidence.

Span pooling takes the mean of content token states in each span.

Cue refinement follows the source: `g = sigmoid(Wg [Hpool; cues] + bg)` and `Href = Hpool + alpha * g * phi(cues)`. The default alpha is 1. Cue counts are raw counts. `phi` is a learned linear projection with no bias.

Graph priors are nonnegative cosine similarity, `exp(-beta * abs(i-j))` for order, and the identity matrix for self connections. Learned mixing weights use a three-way softmax. The combined graph is normalized with `D^(-1/2) A D^(-1/2)`. Invalid nodes are masked before normalization. Self connections are included once in the self prior. The semantic and order diagonals remain part of their respective priors.

Each graph layer computes `LayerNorm(H + Dropout(GELU(W(Ahat H) + b)))`. Span attention then uses the graph states.

Fusion follows page 2: `z = [token; span; CLS]`, `g = sigmoid(Wg z + bg)`, and `F = [token; g * span; CLS]`. Each class score is the dot product of that class's head weights and its fused vector, plus its bias. Softmax operates across classes.

## Details the PDF does not specify

- **Segmentation:** Use balanced word groups. No sentence or legal section parser is specified.
- **Truncation:** Keep the leading content tokens. Only retained text supplies cue counts and MiniLM embeddings. This prevents discarded text from entering the span path. Long-document coverage is a limitation of the proposed token limit.
- **Dual attention:** Use two independent class-query branches and a learned branch gate. The source names Dual Token-LCA but shows one query tensor. It does not define the dual operation.
- **Cue vocabulary:** Use the 12 documented default patterns. A custom `CueExtractor` can supply another set of exactly 12 named patterns. Jurisdiction-specific patterns need separate validation.
- **Graph mixing:** Use trainable global prior weights. The source gives lambda weights but does not define their parameterization. Edge scores vary per document.
- **Graph propagation:** Use two residual graph layers, GELU, and layer normalization. The source does not state layer count or activation.
- **Token attention mask:** Include visible CLS and SEP tokens. Exclude padding. Span pooling excludes all special tokens.
- **Span IDs:** Use zero-based indices and `-1` for noncontent tokens. The diagram uses a different illustrative numbering scheme.
- **Training:** Use cross-entropy for one mutually exclusive target class per document. Multi-label prediction is outside the source's softmax task.
- **Model selection:** Save the final epoch. No early stopping or hyperparameter search is implied.

## Conflicting source details

Page 1 and the prose state that all LegalBERT layers are trainable. Page 2 marks layers 5 through 12 as trainable. The default follows page 1 and the prose. `freeze_lower_layers=4` supports the page 2 variant and also freezes the embedding layer.

Page 1 has extra arrows from token evidence into the graph area. Page 2 and the prose give a strict span chain and say that MiniLM is used only for semantic edges. The implementation follows page 2 and the prose. Token evidence enters fusion directly. It does not build graph edges.

## Operational limits

No labeled dataset or judgment class list was supplied. The repository provides a trainable implementation. It does not include a trained legal outcome model. Synthetic checks measure software behavior only.

The default MiniLM uses its own tokenizer and length limit. A retained span longer than that limit is truncated by MiniLM. This is separate from LegalBERT packing.

The implementation has no web service, database, or document upload UI. Those components are not defined by the supplied architecture.
