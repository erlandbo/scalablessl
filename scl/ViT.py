import torch
from torch import nn
from torch.nn import functional as F


class PatchEmbed(nn.Module):
    def __init__(self, imgsize, patch_dim, embed_dim, in_channels=3):
        super().__init__()
        self.N_patches = (imgsize // patch_dim) * (imgsize // patch_dim)
        self.convembed = nn.Conv2d(in_channels=in_channels, out_channels=embed_dim, kernel_size=patch_dim, stride=patch_dim)

    def forward(self, x):
        x = self.convembed(x)
        x = torch.flatten(x, start_dim=2).transpose(1,2)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.W_Q = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        self.W_K = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        self.W_V = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        self.W_O = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        self.h = num_heads

    def forward(self, Q, K, V):
        N,L,Ek = Q.shape
        N,S,Ek = K.shape
        N,S,Ev = V.shape
        Q = self.W_Q(Q).view(N, L, self.h, Ek//self.h).transpose(1,2)  # (N,h,L,dk)
        K = self.W_K(K).view(N, S, self.h, Ek//self.h).transpose(1,2)  # (N,h,S,dk)
        V = self.W_V(V).view(N, S, self.h, Ev//self.h).transpose(1,2)  # (N,h,S,dv)
        weights = Q @ K.transpose(-1,-2)   # (N,h,L,dk) @ (N,h,dk,S) -> (N,h,L,S)
        weights = weights * (1 / (Ek/self.h)**0.5)
        weights = F.softmax(weights, dim=-1)
        attn = weights @ V  # (N,h,L,S) @ (N,h,S,dv) -> (N,h,L,dv)
        attn = self.W_O(attn.transpose(1,2).contiguous().view(N,L,Ev)) # (N,h,L,dv) -> (N,L,h,dv) -> (N,L,Ev)
        self.weights = weights
        return attn


class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout, activation="relu"):
        super().__init__()
        self.attn = MultiHeadAttention(embed_dim=d_model, num_heads=nhead)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        x_n = self.ln1(x)
        x = x + self.attn(x_n, x_n, x_n)
        x = x + self.mlp(self.ln2(x))
        return x


class ViT(nn.Module):
    def __init__(self, out_dim, in_channels, imgsize, patch_dim, num_layers, d_model, nhead, d_ff_ratio, dropout=0.1, activation="relu"):
        super().__init__()
        self.embed = PatchEmbed(imgsize=imgsize, patch_dim=patch_dim, embed_dim=d_model, in_channels=in_channels)
        self.encoder = nn.ModuleList(
            [EncoderLayer(d_model=d_model, nhead=nhead, d_ff=d_model*d_ff_ratio, dropout=dropout, activation=activation)
             for _ in range(num_layers)]
        )
        self.fc = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, out_dim)) if out_dim > 0 else nn.Identity()
        self.cls_token = nn.Parameter(torch.rand(1, 1, d_model))
        self.posemb = nn.Parameter(torch.rand(1, self.embed.N_patches + 1, d_model))
        self.dropout = nn.Dropout(p=dropout)
        self.num_classes = out_dim

    def forward(self, x):
        x = self.embed(x)  # (N,3,H,W) -> (N,H'W',d_model)
        cls_token = self.cls_token.repeat(x.shape[0],1,1)
        x = torch.cat([cls_token, x], dim=1)  # (N,num_patches,d_model) -> (N,num_patches+1,d_model)
        x = self.dropout(x + self.posemb)
        for layer in self.encoder:
            x = layer(x)  # (N,num_patches+1,d_model)
        x = self.fc(x[:, 0])  # (N,d_model) -> (N,outdim)
        return x


if __name__ == "__main__":
    model = ViT(
        out_dim=128,
        in_channels=3,
        imgsize=32,
        patch_dim=4,
        num_layers=7,
        d_model=512,
        nhead=8,
        d_ff_ratio=4,
        dropout=0.1,
        activation="relu"
    )
    print(model)
    x = torch.rand((8, 3, 32, 32))
    out = model(x)
    print(out.shape)
