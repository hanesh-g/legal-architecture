import json
import torch
from transformers import BertConfig, BertModel, BertTokenizerFast
from legal_outcome import ModelConfig, LegalOutcomeModel
from legal_outcome.data import DocumentProcessor, collate_documents
from legal_outcome.runtime import save_checkpoint, load_checkpoint


def test_checkpoint_roundtrip_without_encoder_download(tmp_path, monkeypatch):
    vocabulary = tmp_path / 'vocab.txt'
    vocabulary.write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nappeal\nis\nallowed\n')
    tokenizer = BertTokenizerFast(vocab_file=str(vocabulary))
    config = ModelConfig(('allow', 'deny'), max_tokens=16, dropout=0.0)
    encoder = BertModel(BertConfig(vocab_size=8, hidden_size=16, num_hidden_layers=1,
                                  num_attention_heads=2, intermediate_size=32))
    model = LegalOutcomeModel(config, encoder).eval()
    processor = DocumentProcessor(tokenizer, config)
    class Semantic:
        def __init__(self, *args, **kwargs):
            pass
        def encode(self, texts, **kwargs):
            return torch.ones(len(texts), 8)
    import sentence_transformers
    monkeypatch.setattr(sentence_transformers, 'SentenceTransformer', Semantic)
    batch = collate_documents([processor('appeal is allowed')], Semantic())
    with torch.no_grad():
        before = model(**batch)['logits']
    path = tmp_path / 'checkpoint'
    save_checkpoint(path, model, processor, {'test': True})
    restored, restored_processor, semantic = load_checkpoint(path, 'cpu')
    with torch.no_grad():
        after = restored(**collate_documents([restored_processor('appeal is allowed')], semantic))['logits']
    assert torch.allclose(before, after, atol=1e-6)
    assert restored.config.labels == ('allow', 'deny')
    assert json.loads((path / 'metadata.json').read_text())['cue_patterns'] == processor.cues.patterns
