import pytest
import torch
from types import SimpleNamespace
from torch import nn
from legal_outcome import ModelConfig, LegalOutcomeModel
from legal_outcome.data import DocumentProcessor, collate_documents
from legal_outcome.model import mean_span_pool
from test_architecture import SemanticEncoder


class NumberTokenizer:
    cls_token_id, sep_token_id, pad_token_id = 1, 2, 0
    def encode(self, text, **kwargs):
        return [int(token) for token in text.split()]
    def decode(self, ids, **kwargs):
        return ' '.join(str(token) for token in ids)


class LocalEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=12, max_position_embeddings=128)
        self.embedding = nn.Embedding(32, 12)
        self.calls = []
    def forward(self, input_ids, attention_mask):
        self.calls.append(input_ids.shape)
        values = self.embedding(input_ids)
        # Contextual local representation makes the chunk CLS depend on content.
        context = (values * attention_mask.unsqueeze(-1)).sum(1, keepdim=True) / attention_mask.sum(1)[:, None, None]
        return SimpleNamespace(last_hidden_state=values + context)


def test_full_8192_packing_retains_tail_and_all_spans():
    config = ModelConfig(('a', 'b'), max_tokens=8192, chunk_tokens=128)
    tokens = ['3'] * 8063 + ['4'] + ['5'] * 50
    item = DocumentProcessor(NumberTokenizer(), config)(' '.join(tokens))
    assert item['input_ids'].shape == (8192,)
    assert (item['span_ids'] >= 0).sum() == 8064
    assert item['input_ids'][-2].item() == 4
    assert item['input_ids'][-1].item() == 2
    assert item['span_ids'][-2] == 7
    assert len(item['spans']) == 8
    assert config.max_content_tokens == 8064


def test_chunked_encoder_respects_positions_and_padding():
    config = ModelConfig(('a', 'b'), max_tokens=64, chunk_tokens=16, dropout=0)
    encoder = LocalEncoder()
    model = LegalOutcomeModel(config, encoder).eval()
    processor = DocumentProcessor(NumberTokenizer(), config)
    items = [processor(' '.join(['3'] * 40)), processor('3 4')]
    batch = collate_documents(items, SemanticEncoder())
    before = model(**batch)
    assert all(shape[-1] == 16 for shape in encoder.calls)
    assert len(encoder.calls) == 4  # Three chunks in the long case, one in the short.
    altered = {k: v.clone() for k, v in batch.items()}
    altered['input_ids'][~altered['attention_mask'].bool()] = 30
    assert torch.allclose(before['logits'], model(**altered)['logits'], atol=1e-6)
    altered = {k: v.clone() for k, v in batch.items()}
    # Change content beyond the first local chunk.
    position = torch.where(altered['span_ids'][0] >= 0)[0][-1]
    altered['input_ids'][0, position] = 10
    assert not torch.equal(before['logits'][0], model(**altered)['logits'][0])


def test_gradient_checkpointing_and_chunk_gradients():
    from transformers import BertConfig, BertModel
    config = ModelConfig(('a', 'b'), max_tokens=64, chunk_tokens=16, gradient_checkpointing=True, dropout=0)
    encoder = BertModel(BertConfig(vocab_size=32, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                                  intermediate_size=32, max_position_embeddings=16))
    model = LegalOutcomeModel(config, encoder)
    assert model.encoder.is_gradient_checkpointing
    item = DocumentProcessor(NumberTokenizer(), config)(' '.join(['3'] * 35))
    batch = collate_documents([item], SemanticEncoder())
    batch['labels'] = torch.tensor([1])
    output = model(**batch)
    output['loss'].backward()
    assert torch.isfinite(output['loss'])
    assert model.encoder.embeddings.word_embeddings.weight.grad.abs().sum() > 0
    assert model.graph.layers[0].weight.grad.abs().sum() > 0


@pytest.mark.parametrize('kwargs', [{'max_tokens':8192}, {'max_tokens':8192,'chunk_tokens':513},
                                   {'max_tokens':8191,'chunk_tokens':128}, {'max_tokens':16384,'chunk_tokens':128}])
def test_invalid_long_configuration(kwargs):
    with pytest.raises(ValueError):
        ModelConfig(('a', 'b'), **kwargs)
