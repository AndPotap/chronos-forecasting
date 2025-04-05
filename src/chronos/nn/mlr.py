import torch
import torch.nn as nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class MLRAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        seq_len: int,
        ranks: str,
        block_szs: str,
        dropout: float,
        use_bias: bool,
    ):
        super().__init__()
        self.ranks = list(map(int, ranks.split("|")))
        self.seq_len = seq_len
        if block_szs == "default":
            self.block_szs = [self.seq_len // (2**idx) for idx in range(len(self.ranks))]
        else:
            self.block_szs = list(map(int, block_szs.split("|")))

        assert d_model % n_head == 0
        assert len(self.ranks) == len(self.block_szs)
        assert sum(self.ranks) == d_model // n_head
        for bdx, curr_rank in zip(self.block_szs, self.ranks):
            assert self.seq_len % bdx == 0
            assert bdx >= curr_rank, f"{bdx} < {curr_rank}"

        self.c_attn = nn.Linear(d_model, 3 * d_model, bias=use_bias)
        self.c_proj = nn.Linear(d_model, d_model, bias=use_bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.d_model = d_model
        self.dropout = dropout

        bias = torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len)
        self.register_buffer("bias", bias)

        mlr_block_mask = self.precompute_MLR_block_wise_division_mask()
        self.register_buffer("mlr_block_mask", mlr_block_mask)

    def precompute_MLR_block_wise_division_mask(self):
        mlr_block_mask = torch.zeros(1, 1, self.seq_len, self.seq_len)

        for i, bdx in enumerate(self.block_szs):
            num_blocks = int(self.seq_len / bdx)

            for j in range(num_blocks):
                sl = slice(bdx * j, bdx * (j + 1))
                mlr_block_mask[..., sl, sl] += 1

        return 1.0 / mlr_block_mask

    def forward(self, x):
        B, T, C = x.shape
        assert T == self.seq_len, f"cannot apply MLR to other length than {self.seq_len}, got {T}"
        qrs, keys, vals = self.c_attn(x).split(self.d_model, dim=2)
        device, dtype = qrs.device, qrs.dtype

        keys = keys.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        qrs = qrs.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        vals = vals.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        q_tuples = torch.split(qrs, self.ranks, dim=-1)
        k_tuples = torch.split(keys, self.ranks, dim=-1)

        att_is = []
        for i, bdx in enumerate(self.block_szs):
            num_blocks = self.seq_len // bdx
            att_i = torch.zeros(B, self.n_head, self.seq_len, self.seq_len, device=device, dtype=dtype)

            curr_q = torch.stack(torch.split(q_tuples[i], bdx, dim=-2))  # [num_blocks,B,T,bdx,rank]
            curr_k = torch.stack(torch.split(k_tuples[i], bdx, dim=-2))
            curr_qk_product = torch.matmul(curr_q, curr_k.transpose(-2, -1))
            curr_qk_product = curr_qk_product * (1.0 / self.ranks[i])  # [num_blocks,B,T,bdx,bdx]

            for j in range(num_blocks):
                sl = slice(bdx * j, bdx * (j + 1))
                att_i[..., sl, sl] = curr_qk_product[j]
            att_is.append(att_i)

        att = sum(att_is)
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ vals
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y
