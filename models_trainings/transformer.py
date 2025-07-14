# models_trainings/transformer.py

import torch
import torch.nn as nn

class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        self.activation = nn.GELU()

    def forward(self, src):
        attn_output, attn_weights = self.self_attn(src, src, src, need_weights=True, average_attn_weights=False)
        src = self.norm1(src + self.dropout(attn_output))
        ff = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = self.norm2(src + self.dropout(ff))
        return src, attn_weights

class AdaptiveDropPatch(nn.Module):
    def __init__(self, input_dim, seq_len, pred_len, patch_size, embed_dim, target_dim, num_heads, num_layers):
        super().__init__()
        self.seq_len, self.pred_len, self.patch_size = seq_len, pred_len, patch_size
        self.num_patches = seq_len // patch_size
        self.patch_embed = nn.Linear(input_dim * patch_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))
        self.transformer_layers = nn.ModuleList([
            CustomTransformerEncoderLayer(embed_dim, num_heads) for _ in range(num_layers)
        ])
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, self.pred_len * target_dim),
            nn.Unflatten(-1, (self.pred_len, target_dim))  # Output shape: [B, pred_len, target_dim]
    )

    def forward(self, x):
        B, L, D = x.shape
        x = x.view(B, self.num_patches, self.patch_size, D)
        patch_var = x.var(dim=2).mean(dim=2)
        threshold = patch_var.quantile(0.2, dim=1, keepdim=True)
        mask = (patch_var >= threshold).float().unsqueeze(-1)
        patches_flat = x.view(B, self.num_patches, -1)
        patches_emb = self.patch_embed(patches_flat) * mask
        patches_emb += self.pos_embed
        attn_weights_all, out = [], patches_emb
        for layer in self.transformer_layers:
            out, attn_weights = layer(out)
            attn_weights_all.append(attn_weights)
        pooled = out.mean(dim=1)
        output = self.decoder(pooled)
        return output, mask, attn_weights_all
