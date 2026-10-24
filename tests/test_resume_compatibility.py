"""Older recovery files remain valid when optional revision fields are added."""
import torch
from legal_outcome import ModelConfig
from legal_outcome.training import CheckpointSelector, make_optimizer, save_training_state, load_training_state
from test_architecture import Encoder
from legal_outcome import LegalOutcomeModel


def test_old_checkpoint_without_revision_defaults_is_readable(tmp_path):
    config = ModelConfig(('a', 'b'))
    model = LegalOutcomeModel(config, Encoder())
    optimizer = make_optimizer(model, 1e-5, 1e-5, .01)
    selector = CheckpointSelector('last')
    selector.update(model, 1)
    contract = {'config': config.to_dict()}
    file = tmp_path / 'state.pt'
    save_training_state(file, model, optimizer, selector, [{'epoch': 1}], contract)
    payload = torch.load(file, weights_only=True)
    payload['contract']['config'].pop('legalbert_revision')
    payload['contract']['config'].pop('minilm_revision')
    for name in ['chunk_tokens', 'encoder_chunk_batch_size', 'gradient_checkpointing']:
        payload['contract']['config'].pop(name)
    torch.save(payload, file)
    epoch, history = load_training_state(file, model, optimizer, selector, contract)
    assert epoch == 1 and history == [{'epoch': 1}]
