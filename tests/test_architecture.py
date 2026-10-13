from types import SimpleNamespace
import pytest
import torch
from torch import nn
from legal_outcome import ModelConfig, LegalOutcomeModel
from legal_outcome.cues import CueExtractor
from legal_outcome.data import DocumentProcessor, segment_document, collate_documents
from legal_outcome.model import mean_span_pool, AdaptiveSpanGraph
from legal_outcome.runtime import metrics, read_records


class Tokenizer:
    cls_token_id, sep_token_id, pad_token_id = 1, 2, 0

    def encode(self, text, add_special_tokens=False):
        return [3 + i for i, _ in enumerate(text.split())]

    def decode(self, ids, skip_special_tokens=True):
        return ' '.join(f'word{i}' for i in ids)


class SemanticEncoder:
    def encode(self, texts, **kwargs):
        return torch.tensor([[1.0, len(t) / 100, 0.5] for t in texts])


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=12)
        self.embedding = nn.Embedding(64, 12)

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


@pytest.fixture
def prepared():
    config = ModelConfig(('allow', 'deny', 'remand'), max_tokens=24, dropout=0.0)
    processor = DocumentProcessor(Tokenizer(), config)
    items = [processor('the appeal is allowed after review of evidence and section four'), processor('dismiss appeal')]
    batch = collate_documents(items, SemanticEncoder())
    return LegalOutcomeModel(config, Encoder()), batch


def test_segmentation_preserves_order_and_words():
    text = 'one two\nthree four five six seven eight nine ten'
    spans = segment_document(text, 3)
    assert len(spans) == 3
    assert ''.join(spans) == text
    assert ''.join(segment_document(text, 8)) == text


@pytest.mark.parametrize('text', ['', '  '])
def test_empty_document_rejected(text):
    with pytest.raises(ValueError):
        segment_document(text)


def test_packing_truncation_and_special_masks():
    config = ModelConfig(('a', 'b'), max_tokens=6, max_spans=2)
    item = DocumentProcessor(Tokenizer(), config)('one two three four five six seven eight nine ten')
    assert item['input_ids'].shape == (6,)
    assert item['span_ids'].tolist() == [-1, 0, 0, 0, 0, -1]
    assert len(item['spans']) == 1
    assert item['cue_features'].shape == (1, 12)


def test_cues_count_and_case():
    values = CueExtractor()(['SECTION 4. Evidence evidence. Appeal denied.'])
    assert values.shape == (1, 12)
    assert values[0, 0] == 1
    assert values[0, 2] == 2
    assert values[0, 6] == 1


def test_pool_excludes_special_and_padding():
    states = torch.tensor([[[100.0], [2.0], [4.0], [9.0], [1000.0]]])
    pooled = mean_span_pool(states, torch.tensor([[-1, 0, 0, 1, -1]]), torch.tensor([[True, True, False]]))
    assert pooled.squeeze().tolist() == [3.0, 9.0, 0.0]


def test_shapes_probability_and_masked_attention(prepared):
    model, batch = prepared
    output = model(**batch)
    assert output['logits'].shape == (2, 3)
    assert output['token_attention'].shape == (2, 2, 3, 24)
    assert output['span_attention'].shape == (2, 3, 8)
    assert torch.allclose(output['probabilities'].sum(-1), torch.ones(2))
    assert torch.equal(output['predictions'], output['logits'].argmax(-1))
    token_padding = (~batch['attention_mask'].bool())[:, None, None, :].expand_as(output['token_attention'])
    span_padding = (~batch['span_mask'])[:, None, :].expand_as(output['span_attention'])
    assert (output['token_attention'][token_padding] == 0).all()
    assert (output['span_attention'][span_padding] == 0).all()


def test_padding_does_not_change_predictions(prepared):
    model, batch = prepared
    model.eval()
    original = model(**batch)['logits']
    changed = {k: v.clone() for k, v in batch.items()}
    changed['input_ids'][~changed['attention_mask'].bool()] = 60
    changed['cue_features'][~changed['span_mask']] = 100
    changed['semantic_embeddings'][~changed['span_mask']] = 100
    assert torch.allclose(original, model(**changed)['logits'], atol=1e-6)


def test_gradients_reach_all_trainable_paths(prepared):
    model, batch = prepared
    batch['labels'] = torch.tensor([0, 1])
    batch['semantic_embeddings'].requires_grad_()
    loss = model(**batch)['loss']
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    assert batch['semantic_embeddings'].grad is None
    for name in ['encoder.embedding.weight', 'graph.prior_logits', 'token_attention.first.queries',
                 'token_attention.second.queries', 'span_attention.queries', 'fusion_gate.weight']:
        assert dict(model.named_parameters())[name].grad.abs().sum() > 0, name


def test_optimizer_changes_loss(prepared):
    model, batch = prepared
    batch['labels'] = torch.tensor([0, 1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    initial = model(**batch)['loss'].item()
    for _ in range(8):
        optimizer.zero_grad()
        loss = model(**batch)['loss']
        loss.backward()
        optimizer.step()
    assert model(**batch)['loss'].item() < initial


def test_graph_symmetry_padding_and_priors():
    graph = AdaptiveSpanGraph(4, 1, 1.0, 0.0)
    semantic = torch.tensor([[[1., 0.], [0., 1.], [99., 99.]]])
    mask = torch.tensor([[True, True, False]])
    a = graph.adjacency(semantic, mask)
    assert torch.allclose(a, a.transpose(1, 2))
    assert torch.isfinite(a).all()
    assert (a[:, 2, :] == 0).all() and (a[:, :, 2] == 0).all()
    assert (a.diagonal(dim1=1, dim2=2)[:, :2] > 0).all()
    with torch.no_grad():
        graph.prior_logits.copy_(torch.tensor([20., -20., -20.]))
    a_sem = graph.adjacency(semantic, mask)
    with torch.no_grad():
        graph.prior_logits.copy_(torch.tensor([-20., 20., -20.]))
    a_order = graph.adjacency(semantic, mask)
    assert a_order[0, 0, 1] > a_sem[0, 0, 1]


def test_semantic_features_only_reach_graph(prepared):
    model, batch = prepared
    model.eval()
    first = model(**batch)
    altered = dict(batch, semantic_embeddings=torch.randn_like(batch['semantic_embeddings']))
    second = model(**altered)
    assert torch.equal(first['token_attention'], second['token_attention'])
    assert not torch.allclose(first['adjacency'], second['adjacency'])


@pytest.mark.parametrize('field', ['span_ids', 'span_mask', 'cue_features', 'attention_mask'])
def test_invalid_inputs_rejected(prepared, field):
    model, batch = prepared
    if field == 'span_ids':
        batch[field][0, 1] = 9
    elif field == 'span_mask':
        batch[field][0] = False
    elif field == 'cue_features':
        batch[field][0, 0, 0] = float('nan')
    else:
        batch[field][0, 0] = 0
    with pytest.raises(ValueError):
        model(**batch)


def test_single_span_supported(prepared):
    model, _ = prepared
    processor = DocumentProcessor(Tokenizer(), model.config)
    batch = collate_documents([processor('word')], SemanticEncoder())
    assert torch.isfinite(model(**batch)['logits']).all()


def test_metrics_match_known_result():
    result = metrics([0, 0, 1, 1], [0, 1, 1, 1], ['a', 'b'])
    assert result['accuracy'] == 0.75
    assert result['confusion_matrix'] == [[1, 1], [0, 2]]
    assert result['macro_f1'] == pytest.approx((2/3 + 4/5)/2)


def test_dataset_validation(tmp_path):
    path = tmp_path / 'data.jsonl'
    path.write_text('{"text":"appeal","label":"unknown"}\n')
    with pytest.raises(ValueError, match='unknown'):
        read_records(path, ('a', 'b'))
    path.write_text('')
    with pytest.raises(ValueError, match='empty'):
        read_records(path, ('a', 'b'))


@pytest.mark.parametrize('kwargs', [{'labels': ('a', 'a')}, {'max_tokens': 400}, {'max_spans': 9}, {'dropout': 1.0}])
def test_config_bounds(kwargs):
    with pytest.raises(ValueError):
        ModelConfig(**({'labels': ('a', 'b')} | kwargs))


def test_cue_refinement_matches_source_equation():
    from legal_outcome.model import CueRefinement
    refinement = CueRefinement(4, scale=0.5)
    pooled = torch.randn(1, 2, 4)
    cues = torch.ones(1, 2, 12)
    mask = torch.tensor([[True, False]])
    expected_gate = torch.sigmoid(refinement.gate(torch.cat([pooled, cues], -1)))
    expected = (pooled + 0.5 * expected_gate * refinement.projection(cues)) * mask.unsqueeze(-1)
    actual = refinement(pooled, cues, mask)
    assert torch.allclose(actual, expected)
    actual.sum().backward()
    assert refinement.gate.weight.grad.abs().sum() > 0
    assert refinement.projection.weight.grad.abs().sum() > 0
    assert not torch.allclose(actual, refinement(pooled, torch.zeros_like(cues), mask))


def test_fusion_gates_only_span_and_uses_class_specific_heads(prepared):
    model, batch = prepared
    model.eval()
    captured = {}
    hooks = [model.token_attention.register_forward_hook(lambda m, a, o: captured.update(token=o[0])),
             model.span_attention.register_forward_hook(lambda m, a, o: captured.update(span=o[0])),
             model.encoder.register_forward_hook(lambda m, a, o: captured.update(cls=o.last_hidden_state[:, 0]))]
    try:
        output = model(**batch)
    finally:
        for hook in hooks:
            hook.remove()
    context = captured['cls'][:, None, :].expand(-1, 3, -1)
    fused = torch.cat([captured['token'], output['span_gate'] * captured['span'], context], -1)
    expected = torch.einsum('bld,ld->bl', fused, model.head_weight) + model.head_bias
    assert torch.allclose(output['logits'], expected)
