from types import SimpleNamespace as sns

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import T5Config
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration as T5ForC

from chronos.nn.mlr import LayerNorm, MLRAttention
from chronos.nn.t5 import T5ForConditionalGeneration


def test_mlr():
    B, L, D = 13, 16, 16
    config = sns(
        n_head=2,
        d_head=D // 2,
        d_model=D,
        mlr_rank_list="4|3|1",
        mlr_block_size_list="4|8|16",
        block_size=L,
        mlr_divide_by_num_levels=False,
        mlr_block_divide_by_num_levels=False,
        link_function="softmax",
        bias=True,
        dropout=0.1,
        split_qkv=False,
        do_qk_ln=False,
        debug=False,
    )
    x = torch.randn(B, L, D)
    torch.manual_seed(seed=21)
    A = AttMLR(config)
    soln = A(x)
    torch.manual_seed(seed=21)
    MLR = MLRAttention(
        d_model=config.d_model,
        n_head=config.n_head,
        seq_len=config.block_size,
        ranks=config.mlr_rank_list,
        block_szs=config.mlr_block_size_list,
        dropout=config.dropout,
        use_bias=config.bias,
    )
    approx = MLR(x)
    diff = torch.linalg.norm(approx - soln)
    print(f"{diff=:.3e}")
    assert diff < 1e-8


def test_t5_attn_equiv():
    B, L, D, nh = 13, 16, 16, 2
    vocab_size, pad_token_id, eos_token_id = 4096, 0, 1
    config = T5Config(
        d_model=D,
        num_heads=nh,
        initializer_factor=0.05,
        seq_len=L,
        ranks="4|3|1",
        block_szs="4|8|16",
    )
    x = torch.randint(low=0, high=vocab_size, size=(B, L))

    # t5 = T5ForConditionalGeneration(config)
    t5 = T5ForC(config)
    t5.resize_token_embeddings(vocab_size)
    t5.config.pad_token_id = t5.generation_config.pad_token_id = pad_token_id
    t5.config.eos_token_id = t5.generation_config.eos_token_id = eos_token_id

    t5_actual = T5ForC(config)
    t5_actual.resize_token_embeddings(vocab_size)
    t5_actual.config.pad_token_id = t5_actual.generation_config.pad_token_id = pad_token_id
    t5_actual.config.eos_token_id = t5_actual.generation_config.eos_token_id = eos_token_id

    t5_actual.load_state_dict(t5.state_dict())
    t5.eval()
    t5_actual.eval()

    approx = t5(x, decoder_input_ids=x)["logits"]
    soln = t5_actual(x, decoder_input_ids=x)["logits"]
    diff = torch.linalg.norm(approx - soln) / torch.linalg.norm(soln)
    print(f"{diff=:.3e}")
    assert diff < 1e-8


def test_t5():
    B, L, D, nh = 13, 16, 16, 2
    vocab_size, pad_token_id, eos_token_id = 4096, 0, 1
    config = T5Config(
        d_model=D,
        num_heads=nh,
        initializer_factor=0.05,
        seq_len=L,
        ranks="4|3|1",
        block_szs="4|8|16",
    )
    t5 = T5ForConditionalGeneration(config)
    t5.resize_token_embeddings(vocab_size)
    t5.config.pad_token_id = t5.generation_config.pad_token_id = pad_token_id
    t5.config.eos_token_id = t5.generation_config.eos_token_id = eos_token_id

    x = torch.randint(low=0, high=vocab_size, size=(B, L))
    y = t5(x, decoder_input_ids=x)
    assert y is not None


class AttMLR(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.d_model % config.n_head == 0

        self.debug = config.debug
        self.mlr_divide_by_num_levels = config.mlr_divide_by_num_levels
        self.mlr_block_divide_by_num_levels = config.mlr_block_divide_by_num_levels

        # MLR rank distribution
        self.ranks = list(map(int, config.mlr_rank_list.split("|")))
        self.cumsum_ranks = torch.cumsum(torch.tensor([0] + self.ranks), dim=0).tolist()
        self.mlr_block_size_list = config.mlr_block_size_list
        self.block_size = config.block_size

        if self.mlr_block_size_list == "default":
            self.mlr_block_size_list = [self.block_size // (2**idx) for idx in range(len(self.ranks))]
        else:
            self.mlr_block_size_list = list(map(int, config.mlr_block_size_list.split("|")))

        assert len(self.ranks) == len(self.mlr_block_size_list)
        assert sum(self.ranks) == config.d_head

        for curr_block_size, curr_rank in zip(self.mlr_block_size_list, self.ranks):
            assert self.block_size % curr_block_size == 0
            assert curr_block_size >= curr_rank

        assert sum(self.ranks) == config.d_head

        self.split_qkv = config.split_qkv
        self.do_qk_ln = config.do_qk_ln
        self.link_function = config.link_function

        if self.do_qk_ln:
            self.ln_q = LayerNorm(config.d_head, bias=config.bias)
            self.ln_k = LayerNorm(config.d_head, bias=config.bias)

        if self.split_qkv:
            self.c_attn_q = nn.Linear(config.d_model, config.d_model, bias=config.bias)
            self.c_attn_k = nn.Linear(config.d_model, config.d_model, bias=config.bias)
            self.c_attn_v = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        else:
            self.c_attn = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)

        self.c_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.d_model = config.d_model
        self.dropout = config.dropout

        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

        if self.mlr_block_divide_by_num_levels:
            mlr_block_mask = self.precompute_MLR_block_wise_division_mask()
            self.register_buffer(
                "mlr_block_mask",
                mlr_block_mask,
            )

    def precompute_MLR_block_wise_division_mask(self):
        mlr_block_mask = torch.zeros(1, 1, self.block_size, self.block_size)

        for i, curr_block_size in enumerate(self.mlr_block_size_list):
            num_blocks = int(self.block_size / curr_block_size)

            for j in range(num_blocks):
                mlr_block_mask[
                    :,
                    :,
                    curr_block_size * j : curr_block_size * (j + 1),
                    curr_block_size * j : curr_block_size * (j + 1),
                ] += 1

        return 1.0 / mlr_block_mask

    def forward(self, x):
        B, T, C = x.shape

        # TODO: didn't implement inference for MLR Attention
        assert T == self.block_size

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        if self.split_qkv:
            q = self.c_attn_q(x)
            k = self.c_attn_k(x)
            v = self.c_attn_v(x)
        else:
            q, k, v = self.c_attn(x).split(self.d_model, dim=2)

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # QK-layer normalization (per-head)
        if self.do_qk_ln:
            q = self.ln_q(q)
            k = self.ln_k(k)

        # split d_head based on "a|b|c|d"
        q_tuples = torch.split(q, self.ranks, dim=-1)
        k_tuples = torch.split(k, self.ranks, dim=-1)

        # MLR Matrix
        list_of_att_i = []

        for i, curr_block_size in enumerate(self.mlr_block_size_list):
            num_blocks = int(self.block_size / curr_block_size)
            att_i = torch.zeros(B, self.n_head, self.block_size, self.block_size, device=q.device, dtype=q.dtype)

            # split block_size based on curr_block_size
            curr_q = torch.stack(torch.split(q_tuples[i], curr_block_size, dim=-2))
            curr_k = torch.stack(torch.split(k_tuples[i], curr_block_size, dim=-2))

            curr_qk_product = torch.matmul(curr_q, curr_k.transpose(-2, -1)) * (1.0 / self.ranks[i])

            for j in range(num_blocks):
                att_i[
                    :,
                    :,
                    curr_block_size * j : curr_block_size * (j + 1),
                    curr_block_size * j : curr_block_size * (j + 1),
                ] = curr_qk_product[j]

            list_of_att_i.append(att_i)

        if self.mlr_divide_by_num_levels:
            att = sum(list_of_att_i) * (1 / len(list_of_att_i))
        else:
            att = sum(list_of_att_i)

        if self.mlr_block_divide_by_num_levels:
            att = att * self.mlr_block_mask

        if self.link_function == "softmax":
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
        elif self.link_function == "identity":
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, 0)
        else:
            raise ValueError

        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y
