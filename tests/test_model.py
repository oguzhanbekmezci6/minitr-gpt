import torch

from config import GPTConfig
from model import MiniGPT


def tiny_model() -> MiniGPT:
    torch.manual_seed(7)
    config = GPTConfig(
        vocab_size=20,
        block_size=16,
        n_layer=2,
        n_head=2,
        n_embd=16,
        dropout=0.0,
    )
    return MiniGPT(config)


def test_forward_shape_and_loss() -> None:
    model = tiny_model()
    x = torch.randint(0, 20, (4, 8))
    y = torch.randint(0, 20, (4, 8))
    logits, loss, _ = model(x, y)

    assert logits.shape == (4, 8, 20)
    assert loss is not None
    assert torch.isfinite(loss)


def test_causal_mask_blocks_future_positions() -> None:
    model = tiny_model()
    model.eval()
    x = torch.randint(0, 20, (1, 8))
    _, _, attention_maps = model(x, return_attention=True)
    assert attention_maps is not None

    first_layer = attention_maps[0][0]  # [head, query, key]
    upper_triangle = torch.triu(first_layer, diagonal=1)
    assert torch.allclose(upper_triangle, torch.zeros_like(upper_triangle), atol=1e-7)


def test_generation_adds_requested_tokens() -> None:
    model = tiny_model()
    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)
    generated = model.generate(
        prompt,
        max_new_tokens=5,
        temperature=1.0,
        top_k=5,
    )
    assert generated.shape == (1, 8)
