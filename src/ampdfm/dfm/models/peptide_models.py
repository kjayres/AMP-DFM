from torch import nn
import torch
import numpy as np
from transformers import AutoModel
import esm
from typing import Optional

class GaussianFourierProjection(nn.Module):
    """
    Gaussian random features for encoding time steps.
    """

    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class Dense(nn.Module):
    """
    A fully connected layer that reshapes outputs to feature maps.
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.dense(x)[...]

class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x): 
        return torch.sigmoid(x) * x

class CNNESMModel(nn.Module):
    """CNN head on top of frozen ESM-2 embeddings with optional conditioning."""

    def __init__(self, alphabet_size: int = 4, embed_dim: int = 256, hidden_dim: int = 256, cond_dim: int = 3):
        """
        Args:
            embed_dim (int): Dimensionality of the token and time embeddings.
        """
        super().__init__()
        self.alphabet_size = alphabet_size
        
        # self.token_embedding = nn.Embedding(self.alphabet_size, embed_dim)
        self.esm = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
        self.esm.eval()
        for param in self.esm.parameters():
            param.requires_grad = False

        self.time_embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim)
        )

        # Conditioning projection (species flags etc.) – always created (matches CNNModel style)
        self.cond_proj = nn.Linear(cond_dim, embed_dim)

        self.swish = Swish()
        
        n = hidden_dim
        
        self.linear = nn.Conv1d(embed_dim, n, kernel_size=9, padding=4)
        
        self.blocks = nn.ModuleList([
            nn.Conv1d(n, n, kernel_size=9, padding=4),
            nn.Conv1d(n, n, kernel_size=9, padding=4),
            nn.Conv1d(n, n, kernel_size=9, dilation=4, padding=16),
            nn.Conv1d(n, n, kernel_size=9, dilation=16, padding=64),
            nn.Conv1d(n, n, kernel_size=9, dilation=64, padding=256),
            # nn.Conv1d(n, n, kernel_size=9, padding=4),
            # nn.Conv1d(n, n, kernel_size=9, padding=4),
            # nn.Conv1d(n, n, kernel_size=9, dilation=4, padding=16),
            # nn.Conv1d(n, n, kernel_size=9, dilation=16, padding=64),
            # nn.Conv1d(n, n, kernel_size=9, dilation=64, padding=256),
            # nn.Conv1d(n, n, kernel_size=9, padding=4),
            # nn.Conv1d(n, n, kernel_size=9, padding=4),
            # nn.Conv1d(n, n, kernel_size=9, dilation=4, padding=16),
            # nn.Conv1d(n, n, kernel_size=9, dilation=16, padding=64),
            # nn.Conv1d(n, n, kernel_size=9, dilation=64, padding=256),
            # nn.Conv1d(n, n, kernel_size=9, padding=4),
            # nn.Conv1d(n, n, kernel_size=9, padding=4),
            # nn.Conv1d(n, n, kernel_size=9, dilation=4, padding=16),
            # nn.Conv1d(n, n, kernel_size=9, dilation=16, padding=64),
            # nn.Conv1d(n, n, kernel_size=9, dilation=64, padding=256)
        ])
        
        self.denses = nn.ModuleList([Dense(embed_dim, n) for _ in range(5)])
        self.norms = nn.ModuleList([nn.GroupNorm(1, n) for _ in range(5)])

        self.final = nn.Sequential(
            nn.Conv1d(n, n, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(n, self.alphabet_size, kernel_size=1)
        )


    def forward(self, x, t, cond_vec=None):
        """
        Args:
            x: Tensor of shape (B, L) containing DNA token indices.
            t: Tensor of shape (B,) containing the time steps.
        Returns:
            out: Tensor of shape (B, L, 4) with output logits for each DNA base.
        """
        # x = self.token_embedding(x) # (B, L) -> (B, L, embed_dim)
        with torch.no_grad():
            x = self.esm(input_ids=x).last_hidden_state  # (B, L, embed_dim)

        # Add conditioning embedding if provided
        if cond_vec is not None:
            cond_emb = self.cond_proj(cond_vec.float())  # (B, embed_dim)
            x = x + cond_emb[:, None, :]  # broadcast over sequence length
        time_embed = self.swish(self.time_embed(t))  # (B, embed_dim)
        
        out = x.permute(0, 2, 1)    # (B, L, embed_dim) -> (B, embed_dim, L)
        out = self.swish(self.linear(out))  # (B, n, L)
        
        # Process through convolutional blocks, adding time conditioning via dense layers.
        for block, dense, norm in zip(self.blocks, self.denses, self.norms):
            # dense(embed) gives (B, n); unsqueeze to (B, n, 1) for broadcasting.
            h = self.swish(block(norm(out + dense(time_embed)[:, :, None])))
            # Residual connection if shapes match.
            if h.shape == out.shape:
                out = h + out
            else:
                out = h
        
        out = self.final(out)  # (B, 4, L)
        out = out.permute(0, 2, 1)  # (B, L, 4)
        
        # Normalization
        out = out - out.mean(dim=-1, keepdim=True)
        return out


class MLPModel(nn.Module):
    def __init__(
        self, input_dim: int = 128, time_dim: int = 1, hidden_dim=128, length=500):
        super().__init__()
        self.input_dim = input_dim
        self.time_dim = time_dim
        self.hidden_dim = hidden_dim

        self.time_embedding = nn.Linear(1, time_dim)
        self.token_embedding = torch.nn.Embedding(self.input_dim, hidden_dim)

        self.swish = Swish()

        self.main = nn.Sequential(
            self.swish,
            nn.Linear(hidden_dim * length + time_dim, hidden_dim),
            self.swish,
            nn.Linear(hidden_dim, hidden_dim),
            self.swish,
            nn.Linear(hidden_dim, hidden_dim),
            self.swish,
            nn.Linear(hidden_dim, self.input_dim * length),
        )

    def forward(self, x, t):
        '''
        x shape (B,L)
        t shape (B,)
        '''
        t = self.time_embedding(t.unsqueeze(-1))
        x = self.token_embedding(x)

        B, N, d = x.shape
        x = x.reshape(B, N * d)
        
        h = torch.cat([x, t], dim=1)
        h = self.main(h)

        h = h.reshape(B, N, self.input_dim)

        return h

class CNNModel(nn.Module):
    """U-Net-style score model for PepDFM/AMP-DFM with optional conditioning.

    If ``cond_dim`` is provided, a linear projection of the conditioning vector
    is added to the time embedding pathway to steer generation.
    """

    def __init__(self, alphabet_size=4, embed_dim=256, hidden_dim=256, cond_dim: Optional[int] = None):
        """
        Args:
            embed_dim (int): Dimensionality of the token and time embeddings.
            cond_dim (int | None): Size of conditioning vector. When ``None`` or 0,
                the model behaves identically to the unconditional variant.
        """
        super().__init__()
        self.alphabet_size = alphabet_size
        
        self.token_embedding = nn.Embedding(self.alphabet_size, embed_dim)

        self.time_embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim)
        )

        # Optional conditioning projection
        self.cond_proj = None
        if cond_dim is not None and cond_dim > 0:
            self.cond_proj = nn.Linear(cond_dim, embed_dim)

        self.swish = Swish()

        n = hidden_dim  # channel width
        
        self.linear = nn.Conv1d(embed_dim, n, kernel_size=9, padding=4)
        
        # Six convolutional blocks for greater capacity
        self.blocks = nn.ModuleList([
            nn.Conv1d(n, n, kernel_size=9, padding=4),
            nn.Conv1d(n, n, kernel_size=9, padding=4),
            nn.Conv1d(n, n, kernel_size=9, dilation=4, padding=16),
            nn.Conv1d(n, n, kernel_size=9, dilation=16, padding=64),
            nn.Conv1d(n, n, kernel_size=9, dilation=64, padding=256),
            nn.Conv1d(n, n, kernel_size=9, padding=4),
        ])
        
        self.denses = nn.ModuleList([Dense(embed_dim, n) for _ in range(6)])
        self.norms  = nn.ModuleList([nn.GroupNorm(1, n) for _ in range(6)])

        self.final = nn.Sequential(
            nn.Conv1d(n, n, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(n, self.alphabet_size, kernel_size=1)
        )

    def forward(self, x, t, cond_vec=None):
        """
        Args:
            x: LongTensor (B, L) of token indices.
            t: FloatTensor (B,) of time steps in [0, 1].
            cond_vec: Optional FloatTensor (B, C) of conditioning bits/features.
        Returns:
            Logits tensor of shape (B, L, vocab).
        """
        x = self.token_embedding(x)

        time_embed = self.swish(self.time_embed(t))  # (B, embed_dim)
        if self.cond_proj is not None and cond_vec is not None:
            cond_embed = self.cond_proj(cond_vec.float())
            fused = time_embed + cond_embed
        else:
            fused = time_embed
        
        out = x.permute(0, 2, 1)
        out = self.swish(self.linear(out))
        
        for block, dense, norm in zip(self.blocks, self.denses, self.norms):
            h = out + dense(fused)[:, :, None]
            h = norm(h)
            h = self.swish(block(h))
            if h.shape == out.shape:
                out = h + out
            else:
                out = h
        
        out = self.final(out)
        out = out.permute(0, 2, 1)
        out = out - out.mean(dim=-1, keepdim=True)
        return out

class CNNModel_Large(nn.Module):
    """A time-dependent score-based model built upon U-Net architecture."""

    def __init__(self, alphabet_size=4, embed_dim=256, hidden_dim=256):
        """
        Args:
            embed_dim (int): Dimensionality of the token and time embeddings.
        """
        super().__init__()
        self.alphabet_size = alphabet_size
        
        self.token_embedding = nn.Embedding(self.alphabet_size, embed_dim)

        self.time_embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim)
        )

        self.swish = Swish()
        
        n = hidden_dim
        
        self.linear = nn.Conv1d(embed_dim, n, kernel_size=9, padding=4)
        
        self.blocks = nn.ModuleList([
            nn.Conv1d(n, n, kernel_size=9, padding=4),
            nn.Conv1d(n, n, kernel_size=9, padding=4),
            nn.Conv1d(n, n, kernel_size=9, dilation=4, padding=16),
            nn.Conv1d(n, n, kernel_size=9, dilation=16, padding=64),
            nn.Conv1d(n, n, kernel_size=9, dilation=64, padding=256),
            nn.Conv1d(n, n, kernel_size=9, padding=4),
            nn.Conv1d(n, n, kernel_size=9, padding=4),
            nn.Conv1d(n, n, kernel_size=9, dilation=4, padding=16),
            nn.Conv1d(n, n, kernel_size=9, dilation=16, padding=64),
            nn.Conv1d(n, n, kernel_size=9, dilation=64, padding=256),
            nn.Conv1d(n, n, kernel_size=9, padding=4),
            nn.Conv1d(n, n, kernel_size=9, padding=4),
            nn.Conv1d(n, n, kernel_size=9, dilation=4, padding=16),
            nn.Conv1d(n, n, kernel_size=9, dilation=16, padding=64),
            nn.Conv1d(n, n, kernel_size=9, dilation=64, padding=256),
            nn.Conv1d(n, n, kernel_size=9, padding=4),
            nn.Conv1d(n, n, kernel_size=9, padding=4),
            nn.Conv1d(n, n, kernel_size=9, dilation=4, padding=16),
            nn.Conv1d(n, n, kernel_size=9, dilation=16, padding=64),
            nn.Conv1d(n, n, kernel_size=9, dilation=64, padding=256)
        ])
        
        self.denses = nn.ModuleList([Dense(embed_dim, n) for _ in range(20)])
        self.norms = nn.ModuleList([nn.GroupNorm(1, n) for _ in range(20)])

        self.final = nn.Sequential(
            nn.Conv1d(n, n, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(n, self.alphabet_size, kernel_size=1)
        )

    def forward(self, x, t):
        """
        Args:
            x: Tensor of shape (B, L) containing DNA token indices.
            t: Tensor of shape (B,) containing the time steps.
        Returns:
            out: Tensor of shape (B, L, 4) with output logits for each DNA base.
        """
        x = self.token_embedding(x) # (B, L) -> (B, L, embed_dim)
        time_embed = self.swish(self.time_embed(t))  # (B, embed_dim)
        
        out = x.permute(0, 2, 1)    # (B, L, embed_dim) -> (B, embed_dim, L)
        out = self.swish(self.linear(out))  # (B, n, L)
        
        # Process through convolutional blocks, adding time conditioning via dense layers.
        for block, dense, norm in zip(self.blocks, self.denses, self.norms):
            # dense(embed) gives (B, n); unsqueeze to (B, n, 1) for broadcasting.
            h = self.swish(block(norm(out + dense(time_embed)[:, :, None])))
            # Residual connection if shapes match.
            if h.shape == out.shape:
                out = h + out
            else:
                out = h
        
        out = self.final(out)  # (B, 4, L)
        out = out.permute(0, 2, 1)  # (B, L, 4)
        
        # Normalization
        out = out - out.mean(dim=-1, keepdim=True)
        return out
class CNNModelPep(nn.Module):
    """A time-dependent score-based model built upon U-Net architecture."""

    def __init__(self, alphabet_size=4, embed_dim=256, hidden_dim=256):
        """
        Args:
            embed_dim (int): Dimensionality of the token and time embeddings.
        """
        super().__init__()
        self.alphabet_size = alphabet_size
        
        self.token_embedding = nn.Embedding(self.alphabet_size, embed_dim)
        # self.esm = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
        # self.esm.eval()
        # for param in self.esm.parameters():
        #     param.requires_grad = False

        self.time_embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim)
        )

        self.swish = Swish()
        
        n = hidden_dim
        
        self.linear = nn.Conv1d(embed_dim, n, kernel_size=9, padding=4)
        
        # Added sixth convolutional block to match wider architecture
        self.blocks = nn.ModuleList([
            nn.Conv1d(n, n, kernel_size=9, padding=4),
            nn.Conv1d(n, n, kernel_size=9, padding=4),
            nn.Conv1d(n, n, kernel_size=9, dilation=4, padding=16),
            nn.Conv1d(n, n, kernel_size=9, dilation=16, padding=64),
            nn.Conv1d(n, n, kernel_size=9, dilation=64, padding=256),
            nn.Conv1d(n, n, kernel_size=9, padding=4),  # new sixth block
        ])
        
        self.denses = nn.ModuleList([Dense(embed_dim, n) for _ in range(6)])
        self.norms  = nn.ModuleList([nn.GroupNorm(1, n) for _ in range(6)])

        self.final = nn.Sequential(
            nn.Conv1d(n, n, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(n, self.alphabet_size, kernel_size=1)
        )

    def forward(self, x, t):
        """
        Args:
            x: Tensor of shape (B, L) containing DNA token indices.
            t: Tensor of shape (B,) containing the time steps.
        Returns:
            out: Tensor of shape (B, L, 4) with output logits for each DNA base.
        """
        x = self.token_embedding(x) # (B, L) -> (B, L, embed_dim)
        # with torch.no_grad():
        #     x = self.esm(input_ids=x).last_hidden_state
        time_embed = self.swish(self.time_embed(t))  # (B, embed_dim)
        
        out = x.permute(0, 2, 1)    # (B, L, embed_dim) -> (B, embed_dim, L)
        out = self.swish(self.linear(out))  # (B, n, L)
        
        # Process through convolutional blocks, adding time conditioning via dense layers.
        for block, dense, norm in zip(self.blocks, self.denses, self.norms):
            # Add time conditioning, apply GroupNorm, then convolution + activation.
            h = out + dense(time_embed)[:, :, None]
            h = norm(h)
            h = self.swish(block(h))
            # Residual connection if shapes match.
            if h.shape == out.shape:
                out = h + out
            else:
                out = h
        
        out = self.final(out)  # (B, 4, L)
        out = out.permute(0, 2, 1)  # (B, L, 4)
        
        # Normalization
        out = out - out.mean(dim=-1, keepdim=True)
        return out

class CNNModelOriginal(nn.Module):
    """A time-dependent score-based model built upon U-Net architecture."""

    def __init__(self, alphabet_size=4, embed_dim=256, hidden_dim=256):
        """
        Args:
            embed_dim (int): Dimensionality of the token and time embeddings.
        """
        super().__init__()
        self.alphabet_size = alphabet_size
        
        self.token_embedding = nn.Embedding(self.alphabet_size, embed_dim)
        # self.esm = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
        # self.esm.eval()
        # for param in self.esm.parameters():
        #     param.requires_grad = False

        self.time_embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim)
        )

        self.swish = Swish()
        
        n = hidden_dim
        
        self.linear = nn.Conv1d(embed_dim, n, kernel_size=9, padding=4)
        
        self.blocks = nn.ModuleList([
            nn.Conv1d(n, n, kernel_size=9, padding=4),
            nn.Conv1d(n, n, kernel_size=9, padding=4),
            nn.Conv1d(n, n, kernel_size=9, dilation=4, padding=16),
            nn.Conv1d(n, n, kernel_size=9, dilation=16, padding=64),
            nn.Conv1d(n, n, kernel_size=9, dilation=64, padding=256),
            # nn.Conv1d(n, n, kernel_size=9, padding=4),
            # nn.Conv1d(n, n, kernel_size=9, padding=4),
            # nn.Conv1d(n, n, kernel_size=9, dilation=4, padding=16),
            # nn.Conv1d(n, n, kernel_size=9, dilation=16, padding=64),
            # nn.Conv1d(n, n, kernel_size=9, dilation=64, padding=256),
            # nn.Conv1d(n, n, kernel_size=9, padding=4),
            # nn.Conv1d(n, n, kernel_size=9, padding=4),
            # nn.Conv1d(n, n, kernel_size=9, dilation=4, padding=16),
            # nn.Conv1d(n, n, kernel_size=9, dilation=16, padding=64),
            # nn.Conv1d(n, n, kernel_size=9, dilation=64, padding=256),
            # nn.Conv1d(n, n, kernel_size=9, padding=4),
            # nn.Conv1d(n, n, kernel_size=9, padding=4),
            # nn.Conv1d(n, n, kernel_size=9, dilation=4, padding=16),
            # nn.Conv1d(n, n, kernel_size=9, dilation=16, padding=64),
            # nn.Conv1d(n, n, kernel_size=9, dilation=64, padding=256)
        ])
        
        self.denses = nn.ModuleList([Dense(embed_dim, n) for _ in range(5)])
        self.norms = nn.ModuleList([nn.GroupNorm(1, n) for _ in range(5)])

        self.final = nn.Sequential(
            nn.Conv1d(n, n, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(n, self.alphabet_size, kernel_size=1)
        )

    def forward(self, x, t):
        """
        Args:
            x: Tensor of shape (B, L) containing DNA token indices.
            t: Tensor of shape (B,) containing the time steps.
        Returns:
            out: Tensor of shape (B, L, 4) with output logits for each DNA base.
        """
        x = self.token_embedding(x) # (B, L) -> (B, L, embed_dim)
        # with torch.no_grad():
        #     x = self.esm(input_ids=x).last_hidden_state
        time_embed = self.swish(self.time_embed(t))  # (B, embed_dim)
        
        out = x.permute(0, 2, 1)    # (B, L, embed_dim) -> (B, embed_dim, L)
        out = self.swish(self.linear(out))  # (B, n, L)
        
        # Process through convolutional blocks, adding time conditioning via dense layers.
        for block, dense, norm in zip(self.blocks, self.denses, self.norms):
            # dense(embed) gives (B, n); unsqueeze to (B, n, 1) for broadcasting.
            h = self.swish(block(norm(out + dense(time_embed)[:, :, None])))
            # Residual connection if shapes match.
            if h.shape == out.shape:
                out = h + out
            else:
                out = h
        
        out = self.final(out)  # (B, 4, L)
        out = out.permute(0, 2, 1)  # (B, L, 4)
        
        # Normalization
        out = out - out.mean(dim=-1, keepdim=True)
        return out