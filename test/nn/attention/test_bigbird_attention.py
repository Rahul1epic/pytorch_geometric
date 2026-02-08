import torch

from torch_geometric.nn.attention import BigBirdAttention


def test_bigbird_attention():
    x = torch.randn(1, 4096, 16)
    mask = torch.ones([1, 4096], dtype=torch.bool)
    attn = BigBirdAttention(channels=16, n_heads=4, block_size=64,
                            num_rand_blocks=3)
    out = attn(x, mask)
    assert out.shape == (1, 4096, 16)
    assert str(attn) == ('BigBirdAttention(heads=4, '
                         'head_dim=64, '
                         'block_size=64, '
                         'num_rand_blocks=3)')
