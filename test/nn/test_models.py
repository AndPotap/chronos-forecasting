import torch
from transformers import T5Config

from chronos.nn.t5 import T5ForConditionalGeneration


def test_t5():
    config = T5Config(initializer_factor=0.05)
    model = T5ForConditionalGeneration(config)
    vocab_size = 4096
    pad_token_id = 0
    eos_token_id = 1
    model.resize_token_embeddings(vocab_size)
    model.config.pad_token_id = model.generation_config.pad_token_id = pad_token_id
    model.config.eos_token_id = model.generation_config.eos_token_id = eos_token_id
    bz, N = 13, 20
    x = torch.randint(low=0, high=vocab_size, size=(bz, N)) 
    y = model(x, decoder_input_ids=x)
    assert y is not None
