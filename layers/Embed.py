import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math
from einops import rearrange
from collections import OrderedDict


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, : x.size(1)]


class PositionalEmbedding_trainable(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        # Create a parameter tensor of size [max_length, d_model]
        pe = torch.randn(max_len, d_model).float()

        # Register it as a parameter that will be updated during training
        self.pe = nn.Parameter(pe, requires_grad=True)

    def forward(self, x):
        # Just return the first T position embeddings
        return self.pe[None, : x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model, kernel_size=3):
        super(TokenEmbedding, self).__init__()
        padding = (kernel_size - 1) // 2  # `same` padding
        self.tokenConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=padding,
            padding_mode="circular",
            bias=False,
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type="fixed", freq="h"):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4  # 15 minutes
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == "fixed" else nn.Embedding
        if freq == "t":
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = (
            self.minute_embed(x[:, :, 4]) if hasattr(self, "minute_embed") else 0.0
        )
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type="timeF", freq="h"):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {"h": 4, "t": 5, "s": 6, "m": 1, "a": 1, "w": 2, "d": 3, "b": 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type="fixed", freq="h", dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = (
            TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
            if embed_type != "timeF"
            else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = (
                self.value_embedding(x)
                + self.temporal_embedding(x_mark)
                + self.position_embedding(x)
            )
        return self.dropout(x)


class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type="fixed", freq="h", dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.temporal_embedding = (
            TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
            if embed_type != "timeF"
            else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class PatchEmbedding(nn.Module):
    def __init__(
        self, d_model, patch_len, stride, padding, dropout, learnable_position=False
    ):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # Positional embedding
        if learnable_position:
            self.position_embedding = PositionalEmbedding_trainable(d_model)
        else:
            self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)  # (B, C, T) -> (B, C, T+S)
        x = x.unfold(
            dimension=-1, size=self.patch_len, step=self.stride
        )  # (B, C, T+S) -> (B, C, T_p, P)
        x = torch.reshape(
            x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        )  # (B, C, T_p, P) -> (B * C, T_p, P)
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(
            x
        )  # (B * C, T_p, P) -> (B * C, T_p, D)
        return self.dropout(x), n_vars


class PatchEmbedding_temp(nn.Module):
    def __init__(
        self,
        C_t,
        d_model,
        patch_len,
        stride,
        dropout,
        pos_embed_type="none",
        token_embed_type="linear",
        kernel_size=3,
        temporal_embed_type="learned",
        freq="h",
    ):
        super(PatchEmbedding_temp, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, stride))

        # Positional embedding (none, learnable, fixed)
        if pos_embed_type == "none":
            self.position_embedding = None
        elif pos_embed_type == "learnable":  # nn.Parameter
            self.position_embedding = PositionalEmbedding_trainable(d_model)
        else:  # sin/cos
            self.position_embedding = PositionalEmbedding(d_model)

        # Token embedding (linear, conv)
        if token_embed_type == "linear":
            self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        elif token_embed_type == "conv":
            self.value_embedding = TokenEmbedding(
                c_in=patch_len, d_model=d_model, kernel_size=kernel_size
            )

        # Temporal embedding (none, fixed, learned, timeF)
        if temporal_embed_type == "none":
            self.temporal_embedding = None
        else:
            # fixed, learned, timeF
            self.temporal_embedding = (
                TemporalEmbedding(
                    d_model=d_model, embed_type=temporal_embed_type, freq=freq
                )
                if temporal_embed_type != "timeF"
                else TimeFeatureEmbedding(
                    d_model=d_model, embed_type=temporal_embed_type, freq=freq
                )
            )

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_mark=None):
        C = x.shape[2]  # x.shape = (B, T, C)

        # do patching and embedding on tokens
        x = rearrange(x, "B T C -> B C T")  # (B, C, T)
        x = self.padding_patch_layer(x)  # (B, C, T+S)
        x = x.unfold(
            dimension=-1, size=self.patch_len, step=self.stride
        )  # (B, C, T_p, P)
        x = rearrange(x, "B C T_p P -> (B C) T_p P")  # (B * C, T_p, P)
        x = self.value_embedding(x)  # (B * C, T_p, D)

        # do patching and embedding on tokens
        if x_mark is not None and self.temporal_embedding is not None:
            x_mark = rearrange(x_mark, "B T C_t -> B C_t T")  # (B, C_t, T)
            x_mark = self.padding_patch_layer(x_mark)  # (B, C_t, T+S)
            x_mark = x_mark.unfold(
                dimension=-1, size=self.patch_len, step=self.stride
            )  # (B, C_t, T_p, P)
            x_mark = x_mark.unsqueeze(1).repeat(1, C, 1, 1, 1)  # (B, C, C_t, T_p, P)
            x_mark = rearrange(
                x_mark, "B C C_t T_p P -> (B C) T_p P C_t"
            )  # (B * C, T_p, P, C_t)
            x_mark = x_mark[
                :, :, 0, :
            ]  # (B * C, T_p, C_t) # select the first value in each patch
            x_mark = self.temporal_embedding(x_mark)  # (B * C, T_p, D)
        else:
            # Even if we have x_mark, we still need to set it to None
            # if self.temporal_embedding is None
            x_mark = None

        # Add positional embedding
        if self.position_embedding is not None:
            x = x + self.position_embedding(x)

        return self.dropout(x + x_mark) if x_mark is not None else self.dropout(x)


class PatchEmbedding_temp_old(nn.Module):
    def __init__(
        self,
        C_t,
        d_model,
        patch_len,
        stride,
        dropout,
        pos_embed_type="none",
        token_embed_type="linear",
        temporal_embed_type="learnable",
    ):
        super(PatchEmbedding_temp_old, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, stride))

        # Positional embedding (none, learnable, fixed)
        if pos_embed_type == "none":
            self.position_embedding = None
        elif pos_embed_type == "learnable":  # nn.Parameter
            self.position_embedding = PositionalEmbedding_trainable(d_model)
        else:  # sin/cos
            self.position_embedding = PositionalEmbedding(d_model)

        # Token embedding (linear, conv)
        if token_embed_type == "linear":
            self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        elif token_embed_type == "conv":
            self.value_embedding = TokenEmbedding(c_in=patch_len, d_model=d_model)

        # Temporal embedding (none, learnable)
        if temporal_embed_type == "none":
            self.temporal_embedding = None
        elif temporal_embed_type == "learnable":
            self.temporal_embedding = nn.Sequential(
                OrderedDict(
                    [
                        ("flatten", nn.Flatten(start_dim=-2)),
                        ("linear", nn.Linear(patch_len * C_t, d_model, bias=False)),
                    ]
                )
            )

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_mark=None):
        C = x.shape[2]  # x.shape = (B, T, C)

        # do patching and embedding on tokens
        x = rearrange(x, "B T C -> B C T")  # (B, C, T)
        x = self.padding_patch_layer(x)  # (B, C, T+S)
        x = x.unfold(
            dimension=-1, size=self.patch_len, step=self.stride
        )  # (B, C, T_p, P)
        x = rearrange(x, "B C T_p P -> (B C) T_p P")  # (B * C, T_p, P)
        x = self.value_embedding(x)  # (B * C, T_p, D)

        # do patching and embedding on tokens
        if x_mark is not None and self.temporal_embedding is not None:
            x_mark = rearrange(x_mark, "B T C_t -> B C_t T")  # (B, C_t, T)
            x_mark = self.padding_patch_layer(x_mark)  # (B, C_t, T+S)
            x_mark = x_mark.unfold(
                dimension=-1, size=self.patch_len, step=self.stride
            )  # (B, C_t, T_p, P)
            x_mark = x_mark.unsqueeze(1).repeat(1, C, 1, 1, 1)  # (B, C, C_t, T_p, P)
            x_mark = rearrange(
                x_mark, "B C C_t T_p P -> (B C) T_p P C_t"
            )  # (B * C, T_p, P, C_t)
            x_mark = self.temporal_embedding(x_mark)  # (B * C, T_p, D)
        else:
            # Even if we have x_mark, we still need to set it to None
            # if self.temporal_embedding is None
            x_mark = None

        # Add positional embedding
        if self.position_embedding is not None:
            x = x + self.position_embedding(x)

        return self.dropout(x + x_mark) if x_mark is not None else self.dropout(x)
