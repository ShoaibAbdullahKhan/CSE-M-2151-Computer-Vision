"""Recurrent video-indexing heads (RNN / LSTM / GRU) and the two non-recurrent baselines, all on the same
2048-d ResNet-50 frame features."""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

CELLS = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}
GATES = {"RNN": 1, "GRU": 3, "LSTM": 4}          # weight blocks per recurrent layer


class Standardise(nn.Module):
    """Fixed per-dimension feature standardisation, statistics from the training videos only."""
    def __init__(self, dim):
        super().__init__()
        self.register_buffer("mu", torch.zeros(dim)); self.register_buffer("sd", torch.ones(dim))

    def fit(self, x):
        self.mu.copy_(x.mean(0)); self.sd.copy_(x.std(0).clamp_min(1e-3))

    def forward(self, x):
        return (x - self.mu) / self.sd


class RecurrentIndexer(nn.Module):
    """frame features -> dropout -> unidirectional recurrent layer(s) -> last hidden state -> Clean / Dirty logits.
    The three variants differ only in the recurrent cell. readout='mean' (ablation) averages the hidden states of all
    time steps instead of using the last one."""
    def __init__(self, cell="GRU", in_dim=2048, hidden=128, layers=1, dropout=0.3, readout="last"):
        super().__init__()
        self.cell, self.readout = cell, readout
        self.norm = Standardise(in_dim)
        self.drop = nn.Dropout(dropout)
        kw = {"nonlinearity": "tanh"} if cell == "RNN" else {}
        self.rnn = CELLS[cell](in_dim, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0.0, **kw)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 2))

    def forward(self, x, lengths):
        x = self.drop(self.norm(x))
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, h = self.rnn(packed)
        if self.readout == "mean":
            out, L = pad_packed_sequence(out, batch_first=True)
            return self.head(out.sum(1) / L.to(out.device)[:, None])
        h = h[0] if self.cell == "LSTM" else h
        return self.head(h[-1])

    def step_logits(self, x):
        """Logits after every time step (for equal-length sequences): what the model would answer having seen
        only the first t frames."""
        out, _ = self.rnn(self.drop(self.norm(x)))
        if self.readout == "mean":
            out = out.cumsum(1) / torch.arange(1, x.shape[1] + 1, device=x.device)[None, :, None]
        return self.head(out)

    def recurrent_states(self, x):
        out, _ = self.rnn(self.norm(x))
        return out


class FrameMLP(nn.Module):
    """Baseline without recurrence. mode='frame': classify each frame on its own and average the logits
    (single-frame classifier); mode='mean': average the frame features first (order-free temporal pooling)."""
    def __init__(self, mode="frame", in_dim=2048, hidden=128, dropout=0.3):
        super().__init__()
        self.cell, self.mode = f"MLP-{mode}", mode
        self.norm = Standardise(in_dim)
        self.net = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 2))

    def forward(self, x, lengths):
        mask = (torch.arange(x.shape[1], device=x.device)[None] < lengths.to(x.device)[:, None]).float()[..., None]
        x = self.norm(x)
        if self.mode == "mean":
            return self.net((x * mask).sum(1) / mask.sum(1))
        return (self.net(x) * mask).sum(1) / mask.sum(1)

    def step_logits(self, x):
        x = self.norm(x)
        t = torch.arange(1, x.shape[1] + 1, device=x.device)[None, :, None]
        if self.mode == "mean":
            return self.net(x.cumsum(1) / t)
        return self.net(x).cumsum(1) / t


def build(name, **kw):
    if name in CELLS:
        return RecurrentIndexer(name, **kw)
    if name in ("MLP-frame", "MLP-mean"):
        kw.pop("layers", None)
        return FrameMLP(name.split("-")[1], **kw)
    raise ValueError(name)


def n_params(model, trainable=True):
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable)


def recurrent_macs(cell, in_dim=2048, hidden=128, layers=1, T=16):
    """Multiply-accumulates of the recurrent part for one sequence of T frames:
    every step computes GATES[cell] blocks of (input x W + h x U)."""
    g = GATES[cell]
    per_step = g * hidden * (in_dim + hidden) + g * hidden * (hidden + hidden) * (layers - 1)
    return per_step * T
